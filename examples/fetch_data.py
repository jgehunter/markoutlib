"""Data download helpers for markoutlib examples."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import polars as pl
import requests

DATA_DIR = Path(__file__).parent / "data"


def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def fetch_binance_aggtrades(
    symbol: str = "BTCUSDT",
    date: str = "2024-01-01",
) -> pl.DataFrame:
    """Download Binance aggTrades from public data (data.binance.vision).

    Returns a DataFrame with columns:
        agg_trade_id, price, qty, first_trade_id, last_trade_id,
        timestamp, is_buyer_maker, is_best_match
    """
    cache = _ensure_data_dir() / f"binance_aggTrades_{symbol}_{date}.csv"
    if not cache.exists():
        url = (
            f"https://data.binance.vision/data/spot/daily/aggTrades"
            f"/{symbol}/{symbol}-aggTrades-{date}.zip"
        )
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            cache.write_bytes(zf.read(csv_name))
        print(f"Saved to {cache}")
    return pl.read_csv(
        cache,
        has_header=False,
        new_columns=[
            "agg_trade_id",
            "price",
            "qty",
            "first_trade_id",
            "last_trade_id",
            "timestamp",
            "is_buyer_maker",
            "is_best_match",
        ],
    )


def fetch_lobster_sample(
    symbol: str = "AAPL",
    level: int = 10,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Download LOBSTER sample data (2012-06-21).

    Returns (messages, orderbook) DataFrames.
    """
    cache_dir = _ensure_data_dir() / f"lobster_{symbol}_{level}"
    msg_path = cache_dir / "messages.csv"
    ob_path = cache_dir / "orderbook.csv"

    if not msg_path.exists():
        url = (
            f"https://data.lobsterdata.com/info/sample"
            f"/LOBSTER_SampleFile_{symbol}_2012-06-21_{level}.zip"
        )
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        cache_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                lower = name.lower()
                if "message" in lower:
                    msg_path.write_bytes(zf.read(name))
                elif "orderbook" in lower:
                    ob_path.write_bytes(zf.read(name))
        print(f"Extracted to {cache_dir}")

    messages = pl.read_csv(
        msg_path,
        has_header=False,
        new_columns=[
            "time",
            "event_type",
            "order_id",
            "size",
            "price",
            "direction",
        ],
    )
    ob_cols = []
    for i in range(1, level + 1):
        ob_cols.extend(
            [
                f"ask_price_{i}",
                f"ask_size_{i}",
                f"bid_price_{i}",
                f"bid_size_{i}",
            ]
        )
    orderbook = pl.read_csv(ob_path, has_header=False, new_columns=ob_cols)

    return messages, orderbook


def prepare_binance_trades_and_quotes(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convert Binance aggTrades to markoutlib trades + quotes.

    Since Binance public data has no BBO stream, we derive a synthetic
    mid from the last buy and sell trade prices. This is standard
    practice for crypto spot when order-book snapshots aren't available.

    Returns (trades, quotes).
    """
    # Auto-detect timestamp resolution: ms (13 digits) vs us (16 digits)
    ts_magnitude = df["timestamp"][0]
    if ts_magnitude > 1e15:  # microseconds
        ts_expr = pl.col("timestamp").cast(pl.Datetime("us"))
    else:  # milliseconds
        ts_expr = (pl.col("timestamp") * 1_000).cast(pl.Datetime("us"))

    raw = df.select(
        ts_expr.alias("timestamp"),
        # is_buyer_maker=True means the taker sold -> side = -1
        pl.when(pl.col("is_buyer_maker")).then(-1).otherwise(1).alias("side"),
        pl.col("price").cast(pl.Float64),
        pl.col("qty").cast(pl.Float64).alias("size"),
    ).sort("timestamp")

    # Derive a synthetic mid: rolling last-buy and last-sell prices
    last_buy = (
        pl.when(pl.col("side") == 1)
        .then(pl.col("price"))
        .otherwise(None)
        .forward_fill()
        .alias("_last_buy")
    )
    last_sell = (
        pl.when(pl.col("side") == -1)
        .then(pl.col("price"))
        .otherwise(None)
        .forward_fill()
        .alias("_last_sell")
    )
    raw = raw.with_columns(last_buy, last_sell)

    # Drop rows before we have both a buy and a sell
    raw = raw.filter(
        pl.col("_last_buy").is_not_null() & pl.col("_last_sell").is_not_null()
    )

    raw = raw.with_columns(
        ((pl.col("_last_buy") + pl.col("_last_sell")) / 2).alias("mid")
    )

    trades = raw.select("timestamp", "side", "price", "mid", "size")
    quotes = (
        raw.select("timestamp", "mid").unique(subset=["timestamp"]).sort("timestamp")
    )

    return trades, quotes


def prepare_lobster(
    messages: pl.DataFrame, orderbook: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convert LOBSTER data to markoutlib trades + quotes format."""
    base_date = datetime(2012, 6, 21)

    mid = (
        orderbook["ask_price_1"].cast(pl.Float64) / 10_000
        + orderbook["bid_price_1"].cast(pl.Float64) / 10_000
    ) / 2

    combined = messages.with_columns(
        (
            pl.lit(base_date)
            + pl.duration(microseconds=(pl.col("time") * 1_000_000).cast(pl.Int64))
        ).alias("timestamp"),
        mid.alias("mid"),
        (pl.col("price").cast(pl.Float64) / 10_000).alias("price_adj"),
    )

    # Trades: event types 4 (visible) and 5 (hidden)
    # LOBSTER direction is the passive order's side; negate to get taker side.
    trades = combined.filter(pl.col("event_type").is_in([4, 5])).select(
        "timestamp",
        (-pl.col("direction")).alias("side"),
        pl.col("price_adj").alias("price"),
        "mid",
        pl.col("size").cast(pl.Float64).alias("size"),
    )

    # Quotes: all events with unique timestamps
    quotes = (
        combined.select("timestamp", "mid")
        .unique(subset=["timestamp"])
        .sort("timestamp")
    )

    return trades, quotes
