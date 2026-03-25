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


def fetch_tardis_crypto(
    exchange: str = "binance",
    data_type: str = "trades",
    symbol: str = "BTCUSDT",
    year: int = 2024,
    month: int = 1,
    day: int = 1,
) -> pl.DataFrame:
    """Download Tardis.dev free first-of-month data."""
    cache = (
        _ensure_data_dir()
        / f"tardis_{exchange}_{data_type}_{symbol}_{year}{month:02d}{day:02d}.csv.gz"
    )
    if not cache.exists():
        url = (
            f"https://datasets.tardis.dev/v1/{exchange}/{data_type}"
            f"/{year}/{month:02d}/{day:02d}/{symbol}.csv.gz"
        )
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
        print(f"Saved to {cache}")
    return pl.read_csv(cache)


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
            f"https://lobsterdata.com/info/sample"
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


def prepare_tardis_trades(df: pl.DataFrame) -> pl.DataFrame:
    """Convert Tardis trades to markoutlib format."""
    return df.select(
        (pl.col("timestamp") * 1000).cast(pl.Datetime("us")).alias("timestamp"),
        pl.when(pl.col("side") == "buy").then(1).otherwise(-1).alias("side"),
        pl.col("price").cast(pl.Float64),
        pl.col("amount").cast(pl.Float64).alias("size"),
    )


def prepare_tardis_quotes(df: pl.DataFrame) -> pl.DataFrame:
    """Convert Tardis quotes to markoutlib format."""
    return df.select(
        (pl.col("timestamp") * 1000).cast(pl.Datetime("us")).alias("timestamp"),
        (
            (
                pl.col("ask_price").cast(pl.Float64)
                + pl.col("bid_price").cast(pl.Float64)
            )
            / 2
        ).alias("mid"),
    )


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
    trades = combined.filter(pl.col("event_type").is_in([4, 5])).select(
        "timestamp",
        pl.col("direction").alias("side"),
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
