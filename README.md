# markoutlib

Markout P&L analysis for Python. Polars-native. Instrument-agnostic. Statistically rigorous.

## Why this library?

1. **There's no good open-source implementation.** Markout computation is conceptually simple -- an asof join and some arithmetic. But a correct implementation accumulates edge cases fast: session boundaries, stale quotes, auction states, null propagation when forward data is unavailable. This library makes those decisions explicit and documented.

2. **Three clock domains matter.** Wall-clock, trade-clock, and tick-clock markouts answer different questions. Comparing across all three on the same dataset with consistent segmentation is where you actually learn something about flow characteristics.

3. **Point estimates aren't enough.** A markout curve without confidence intervals is a number, not a conclusion. Block bootstrap CIs, permutation tests for segment differences, and decay half-life estimation are built in.

4. **The gap between weighted and unweighted markout is a signal.** When equal-weighted and size-weighted markouts diverge, that tells you something about the relationship between trade size and information content. The library makes this comparison trivial.

5. **No magic.** You provide the mid. You provide the side. The library never infers, constructs, or guesses. When forward data isn't available, you get null -- not a silent fallback.

## Quickstart

```python
import polars as pl
import markoutlib as mo

result = mo.compute(
    trades=trades,
    quotes=quotes,
    horizons=mo.seconds(1, 5, 30),
)

result.curve()
result.plot.curve()
```

## Full example

```python
import markoutlib as mo

# Multiple horizon types in one pass
result = mo.compute(
    trades=trades,
    quotes=quotes,
    horizons=mo.seconds(1, 5, 30) + mo.trades(10, 50) + mo.ticks(100),
    unit="bps",
)

# Markout curve with bootstrap CIs and Newey-West t-stats
result.curve()

# Segment by any column in your trades DataFrame
result.curve(by="counterparty")

# Size-weighted markout
result.curve(weight="size")

# Weighted vs unweighted side-by-side
result.compare(weight="size")

# Exponential decay fit -> half-life in horizon units
result.half_life()

# Permutation test: does markout differ across counterparties?
result.test("counterparty")

# Pairwise tests with Benjamini-Hochberg correction
result.test("counterparty", pairwise=True)

# Pre-trade baselines — detect information leakage
result = mo.compute(
    trades=trades,
    quotes=quotes,
    horizons=mo.seconds_range(-30, 30, step=1),
)
result.plot.curve()  # Full crossing-zero curve

# Spread decomposition: effective = realized + price impact
result.spread_decomposition(horizon=mo.seconds(5))
result.effective_spread()
result.realized_spread(horizon=mo.seconds(5))

# Visualization
result.plot.curve()
result.plot.curve(by="counterparty")
result.plot.heatmap(by="symbol")
result.plot.distribution(horizon=mo.seconds(5).single())
result.plot.comparison(by="counterparty")
result.plot.scatter(x="size", horizon=mo.seconds(5).single())

# Export
df = result.to_polars()
pdf = result.to_pandas()  # requires pip install markoutlib[pandas]
```

## Concepts

### Horizon types

**Wall-clock** (`mo.seconds(1, 5, 30)`) -- measure the mid N seconds after each trade. Uses an asof join against the quote stream. Stale quote protection nulls out matches where the nearest quote is more than 2x the horizon away. Use wall-clock when you care about real-time P&L impact.

**Trade-clock** (`mo.trades(10, 50)`) -- measure the mid at the Nth subsequent trade. No quote data required. Use trade-clock when you want to control for activity rate -- a 10-trade markout means the same thing in a liquid name and an illiquid one.

**Tick-clock** (`mo.ticks(100)`) -- measure the mid at the Nth subsequent quote update. Use tick-clock when you want to normalize by information arrival rate rather than calendar time.

All three can be combined in a single `compute()` call via `+` on horizon sets.

### Sign convention

Positive markout = price moved in the direction of the analyzed party's trade.

- `side=1` (buy): markout is positive when the mid rose after the trade
- `side=-1` (sell): markout is positive when the mid fell after the trade

Formula: `side * (future_mid - mid)`, scaled to bps by default.

### Units

- `unit="bps"` (default): `side * (future_mid - mid) / mid * 10000`
- `unit="price"`: `side * (future_mid - mid)`

### Negative offsets and pre-trade baselines

Horizon values can be negative. A horizon of `-5` seconds looks up the mid 5 seconds *before* each trade, giving you a pre-trade baseline. This is the standard technique for detecting information leakage: if the markout curve is already non-zero before time zero, someone is trading on stale information or the signal is arriving before the trade timestamp suggests.

Use `seconds_range(-30, 30, step=1)` to generate a full crossing-zero curve. The `half_life()` method automatically filters out negative horizons since decay fitting only makes sense on the post-trade portion.

### Spread decomposition

The Huang-Stoll identity decomposes the effective half-spread into realized spread and price impact:

`effective_spread = realized_spread + price_impact`

The effective spread measures execution cost at the moment of the trade. The realized spread measures what the liquidity provider actually earns after prices move. The price impact measures the permanent information content of the trade. When price impact dominates, the flow is informed. When realized spread dominates, the liquidity provider is profiting from transient effects.

### The mid column contract

You provide the mid. The library never constructs it from bid/ask, NBBO, or any other source. This is deliberate -- mid calculation varies by instrument, venue, and use case. Bring your own.

Required columns on `trades`: `timestamp`, `side`, `price`, `mid`.
Required columns on `quotes`: `timestamp`, `mid`.

Any additional columns (e.g. `counterparty`, `symbol`, `size`) pass through and are available for segmentation and weighting.

## API reference

### `mo.compute(trades, quotes, *, horizons, unit="bps", by=None)`

Core computation. Returns a `MarkoutResult`. Accepts Polars or pandas DataFrames (pandas is converted internally).

| Parameter | Type | Description |
|-----------|------|-------------|
| `trades` | `DataFrame` | Trade records with `timestamp`, `side`, `price`, `mid` |
| `quotes` | `DataFrame \| None` | Quote records with `timestamp`, `mid`. Required for wall/tick horizons |
| `horizons` | `HorizonSet` | Built via `mo.seconds()`, `mo.trades()`, `mo.ticks()`, composable with `+` |
| `unit` | `str` | `"bps"` or `"price"` |
| `by` | `str \| list[str] \| None` | Partition column(s) for per-symbol or per-venue computation |

### Horizon constructors

| Function | Clock domain | Requires quotes |
|----------|-------------|-----------------|
| `mo.seconds(*values)` | Wall-clock | Yes |
| `mo.trades(*values)` | Trade-clock | No |
| `mo.ticks(*values)` | Tick-clock | Yes |

Range constructors generate uniform grids with inclusive stop:

```python
mo.seconds_range(start=1, stop=60, step=5)    # 1, 6, 11, ..., 56
mo.seconds_range(start=-30, stop=30, step=5)   # Crosses zero
mo.trades_range(start=1, stop=100, step=10)
mo.ticks_range(start=1, stop=50, step=5)
```

### `MarkoutResult` methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.curve(by=, weight=)` | `DataFrame` | Mean, median, quantiles, bootstrap CI, Newey-West t-stat per horizon |
| `.half_life(by=)` | `DecayFitResult \| DataFrame` | Exponential decay fit with half-life, time constant, R-squared |
| `.test(column, pairwise=)` | `DataFrame` | Permutation test for markout differences across segments |
| `.compare(weight=)` | `DataFrame` | Weighted vs unweighted mean side-by-side |
| `.plot.curve(by=)` | `Figure` | Markout decay curve with CIs |
| `.plot.heatmap(by=)` | `Figure` | Horizon x segment heatmap |
| `.plot.distribution(horizon=, by=)` | `Figure` | Markout distribution at a single horizon |
| `.plot.comparison(by=)` | `Figure` | Segment comparison chart |
| `.plot.scatter(x=, horizon=)` | `Figure` | Markout vs continuous variable |
| `.effective_spread(by=)` | `DataFrame` | Effective half-spread per trade |
| `.realized_spread(horizon=, by=)` | `DataFrame` | Realized half-spread at given horizon |
| `.price_impact(horizon=, by=)` | `DataFrame` | Price impact component at given horizon |
| `.spread_decomposition(horizon=, by=)` | `DataFrame` | Full Huang-Stoll decomposition |
| `.to_polars()` | `DataFrame` | Raw per-trade markout data |
| `.to_pandas()` | `pd.DataFrame` | Pandas export (requires `markoutlib[pandas]`) |

## Examples

See the `examples/` directory for runnable notebooks:

- `01_quickstart.py` — Synthetic data API walkthrough
- `02_crypto_markouts.py` — Binance BTCUSDT markout analysis (Tardis.dev data)
- `03_spread_decomposition.py` — AAPL effective/realized spread (LOBSTER data)
- `04_information_leakage.py` — Pre-trade baseline detection (LOBSTER data)

Open in VS Code or JupyterLab as interactive notebooks (percent format).

## Non-goals

- **Constructing the mid.** You know your instrument better than this library does.
- **Trade classification.** Side must be provided. Use Lee-Ready, bulk volume classification, or whatever applies to your data.
- **Tick data storage or retrieval.** This is a computation library, not a data pipeline.
- **Real-time / streaming computation.** Batch only. If you need streaming markouts, you need different architecture.
- **Multi-asset portfolio markout.** This operates on a flat table of trades. Portfolio-level aggregation is your responsibility.

## Roadmap

- [ ] Rust-accelerated tick-clock joins via `pyo3-polars`
- [ ] Async-aware session boundary handling
- [ ] Configurable stale quote tolerance
- [ ] Additional bootstrap methods (BCa, studentized)
- [ ] LaTeX report generation
- [x] Markout attribution decomposition (spread vs alpha components)

## Installation

```bash
pip install markoutlib           # v0.2
```

With pandas support:

```bash
pip install markoutlib[pandas]
```

Requires Python 3.11+. Polars is the only required dependency.

## License

MIT
