# %% [markdown]
# # markoutlib Quickstart
#
# Basic API walkthrough using synthetic data.

# %%
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import markoutlib as mo

# %% [markdown]
# ## Create synthetic data
#
# 500 trades over ~80 minutes, alternating buy/sell.
# Quotes every 100ms with a slow upward drift.

# %%
rng = np.random.default_rng(42)
base = datetime(2024, 1, 15, 10, 0, 0)
n_trades = 500

trades = pl.DataFrame(
    {
        "timestamp": [
            base + timedelta(seconds=i * 10 + rng.uniform(0, 5))
            for i in range(n_trades)
        ],
        "side": [1 if rng.random() > 0.5 else -1 for _ in range(n_trades)],
        "price": [
            100.0 + 0.001 * i + rng.normal(0, 0.005) for i in range(n_trades)
        ],
        "mid": [100.0 + 0.001 * i for i in range(n_trades)],
        "size": rng.lognormal(5, 1, n_trades).tolist(),
        "counterparty": [f"LP_{rng.integers(0, 4)}" for _ in range(n_trades)],
    }
).cast({"timestamp": pl.Datetime("us")})

n_quotes = 50_000
quotes = pl.DataFrame(
    {
        "timestamp": [
            base + timedelta(milliseconds=100 * i) for i in range(n_quotes)
        ],
        "mid": [
            100.0 + 0.001 * (i / 10) + rng.normal(0, 0.002)
            for i in range(n_quotes)
        ],
    }
).cast({"timestamp": pl.Datetime("us")})

print(f"Trades: {trades.shape}, Quotes: {quotes.shape}")
trades.head()

# %% [markdown]
# ## Compute markouts

# %%
result = mo.compute(
    trades=trades,
    quotes=quotes,
    horizons=mo.seconds(1, 5, 10, 30, 60) + mo.trades(5, 10, 50),
    unit="bps",
)

# %% [markdown]
# ## Markout curve

# %%
result.curve()

# %%
result.plot.curve()

# %% [markdown]
# ## Segmentation by counterparty

# %%
result.curve(by="counterparty")

# %%
result.plot.curve(by="counterparty")

# %% [markdown]
# ## Weighted vs unweighted

# %%
result.compare(weight="size")

# %% [markdown]
# ## Statistical tests

# %%
result.test("counterparty")

# %% [markdown]
# ## Half-life estimation

# %%
hl = result.half_life()
if hl.converged:
    print(f"Half-life: {hl.half_life:.1f}s, Terminal: {hl.terminal_markout:.2f} bps")
else:
    print("Decay fit did not converge (expected with synthetic random data)")

# %% [markdown]
# ## Raw data export

# %%
result.to_polars().head(10)
