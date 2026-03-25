# %% [markdown]
# # Information Leakage Detection
#
# Pre-trade baselines reveal whether the mid is already moving
# before a trade executes. This is the signature of informed flow.
#
# Data: LOBSTER sample, AAPL, 2012-06-21.

# %%
import sys

sys.path.insert(0, ".")

import numpy as np  # noqa: F401
import polars as pl
from fetch_data import fetch_lobster_sample, prepare_lobster

import markoutlib as mo

# %% [markdown]
# ## Download and prepare data

# %%
messages, orderbook = fetch_lobster_sample("AAPL", level=10)
trades, quotes = prepare_lobster(messages, orderbook)

print(f"Trades: {trades.shape[0]:,}")
print(f"Quotes: {quotes.shape[0]:,}")

# %% [markdown]
# ## Compute pre-trade and post-trade markouts
#
# 61 horizons from -30 seconds to +30 seconds, stepping by 1 second.
# This captures the full picture: what happened before AND after each trade.

# %%
result = mo.compute(
    trades=trades,
    quotes=quotes,
    horizons=mo.seconds_range(-30, 30, 1),
    unit="bps",
)

# %% [markdown]
# ## The crossing-zero curve
#
# The x-axis crosses zero at the moment of the trade.
# Left of zero: pre-trade price movement.
# Right of zero: post-trade price movement (standard markout).
#
# If the curve is flat before zero and slopes after, the flow is uninformed.
# If the curve is already trending before zero, there's information leakage.

# %%
result.plot.curve()

# %% [markdown]
# ## Segment by trade size
#
# Are large trades preceded by more pre-trade price movement?

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
    horizons=mo.seconds_range(-30, 30, 1),
    unit="bps",
)

result_sized.plot.curve(by="size_bucket")

# %% [markdown]
# ## Quantify pre-trade slope
#
# Compare the mean markout at -30s, -10s, and 0 for each size bucket.

# %%
curve = result_sized.curve(by="size_bucket")
pre_trade = curve.filter(pl.col("horizon_value").is_in([-30.0, -10.0, 0.0]))
print(pre_trade.select("size_bucket", "horizon_value", "markout_mean", "n_obs"))

# %% [markdown]
# ## Heatmap: size bucket x horizon

# %%
result_sized.plot.heatmap(by="size_bucket")

# %% [markdown]
# ## Observations
#
# - [Fill in after running with real data]
# - Is there a visible pre-trade trend for large trades?
# - How does the pre-trade slope compare between size buckets?
# - At what negative horizon does the trend become detectable?
# - This analysis would not be possible without negative-offset support.
