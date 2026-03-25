# %% [markdown]
# # Crypto Markout Analysis: Binance BTCUSDT
#
# Markout curves on real cryptocurrency trade data.
# Data: Binance spot BTCUSDT, 2024-01-01, via [Tardis.dev](https://tardis.dev).

# %%
import sys

sys.path.insert(0, ".")

import numpy as np  # noqa: F401
import polars as pl
from fetch_data import (
    fetch_tardis_crypto,
    prepare_tardis_quotes,
    prepare_tardis_trades,
)

import markoutlib as mo

# %% [markdown]
# ## Download and prepare data

# %%
raw_trades = fetch_tardis_crypto("binance", "trades", "BTCUSDT", 2024, 1, 1)
raw_quotes = fetch_tardis_crypto("binance", "quotes", "BTCUSDT", 2024, 1, 1)

trades = prepare_tardis_trades(raw_trades)
quotes = prepare_tardis_quotes(raw_quotes)

# Add mid to trades via asof join
trades = trades.sort("timestamp").join_asof(
    quotes.sort("timestamp"), on="timestamp", strategy="backward"
)

print(f"Trades: {trades.shape[0]:,}")
print(f"Quotes: {quotes.shape[0]:,}")
print(f"Time range: {trades['timestamp'].min()} to {trades['timestamp'].max()}")

# %% [markdown]
# ## Compute markouts
#
# Wall-clock horizons from 1 second to 5 minutes, plus trade-clock.

# %%
result = mo.compute(
    trades=trades,
    quotes=quotes,
    horizons=mo.seconds(1, 5, 10, 30, 60, 300) + mo.trades(10, 50, 100, 500),
    unit="bps",
)

# %% [markdown]
# ## Markout decay curve

# %%
result.plot.curve()

# %% [markdown]
# ## Segment by trade size
#
# Bucket trades into small / medium / large by quantity.

# %%
q33 = trades["size"].quantile(0.33)
q67 = trades["size"].quantile(0.67)

trades_tagged = trades.with_columns(
    pl.when(pl.col("size") <= q33)
    .then(pl.lit("small"))
    .when(pl.col("size") <= q67)
    .then(pl.lit("medium"))
    .otherwise(pl.lit("large"))
    .alias("size_bucket")
)

result_sized = mo.compute(
    trades=trades_tagged,
    quotes=quotes,
    horizons=mo.seconds(1, 5, 10, 30, 60, 300),
    unit="bps",
)

result_sized.plot.curve(by="size_bucket")

# %% [markdown]
# ## Weighted vs unweighted comparison
#
# Does size-weighting change the markout picture?

# %%
result.compare(weight="size")

# %% [markdown]
# ## Distribution at 30 seconds

# %%
result.plot.distribution(horizon=mo.seconds(30))

# %% [markdown]
# ## Observations
#
# - [Fill in real conclusions after running with actual data]
# - Look for: do large trades show more adverse selection?
# - Does the markout curve flatten (information fully absorbed)?
# - Is there divergence between weighted and unweighted means?
