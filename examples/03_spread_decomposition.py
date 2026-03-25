# %% [markdown]
# # Spread Decomposition: AAPL (NASDAQ)
#
# Effective spread = realized spread + price impact.
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
trades.head()

# %% [markdown]
# ## Compute markouts at multiple horizons

# %%
result = mo.compute(
    trades=trades,
    quotes=quotes,
    horizons=mo.seconds(1, 5, 10, 30, 60),
    unit="bps",
)

# %% [markdown]
# ## Spread decomposition at 5 seconds

# %%
result.spread_decomposition(horizon=mo.seconds(5))

# %% [markdown]
# ## Verify the identity

# %%
decomp = result.spread_decomposition(horizon=mo.seconds(5))
for row in decomp.iter_rows(named=True):
    eff = row["effective_spread_mean"]
    real = row["realized_spread_mean"]
    imp = row["price_impact_mean"]
    print(f"Effective: {eff:.4f}, Realized: {real:.4f}, Impact: {imp:.4f}")
    print(
        f"Identity check: {eff:.4f} = {real + imp:.4f}"
        f" (diff: {abs(eff - real - imp):.2e})"
    )

# %% [markdown]
# ## Decomposition across horizons
#
# How does the split between realized spread and price impact
# change as the horizon increases?

# %%
rows = []
for h in [1, 5, 10, 30, 60]:
    d = result.spread_decomposition(horizon=mo.seconds(h))
    for r in d.iter_rows(named=True):
        rows.append(
            {
                "horizon_s": h,
                "effective_spread": r["effective_spread_mean"],
                "realized_spread": r["realized_spread_mean"],
                "price_impact": r["price_impact_mean"],
            }
        )

decomp_df = pl.DataFrame(rows)
print(decomp_df)

# %% [markdown]
# ## Segment by trade size

# %%
q50 = trades["size"].quantile(0.5)
trades_tagged = trades.with_columns(
    pl.when(pl.col("size") <= q50)
    .then(pl.lit("small"))
    .otherwise(pl.lit("large"))
    .alias("size_bucket")
)

result_sized = mo.compute(
    trades=trades_tagged,
    quotes=quotes,
    horizons=mo.seconds(5),
    unit="bps",
)

result_sized.spread_decomposition(horizon=mo.seconds(5), by="size_bucket")

# %% [markdown]
# ## Observations
#
# - [Fill in after running with real data]
# - Do larger trades have higher price impact?
# - At what horizon does the realized spread approach zero?
# - Is the effective spread consistent with AAPL's typical spread in 2012?
