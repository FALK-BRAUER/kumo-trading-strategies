# tech-universe-daily-2025-2026

QC27 — tech momentum with inverse-volatility allocation — on Alpaca SIP daily bars, 2025-01-02 → 2026-08-10.
The set IS the data: `bars.parquet` (537 symbols: the 534-name US Technology universe from `sectors.parquet` plus
SPY/QQQ/GLD; raw OHLCV plus the provider's split/dividend-adjusted close as `close_adj`), `sectors.parquet` (the
sector snapshot the universe filter reads), `half_spreads.json` (the measured half-spread cost model; it covers 26 %
of the tech universe, the rest pays the default median spread). `manifest.json` records the extract.

`report.html` compares the variations side by side; each variation is its `parameters.json` — `run.py` is identical:

| variation | what | total return | Sharpe | max DD | round trips |
|---|---|---|---|---|---|
| `monthly` | QC27 as specified: monthly rebalance, momentum on `close_adj`, no view | +193.5 % | 1.71 | −33.5 % | 168 |
| `daily-no-view` | the selected cadence and price field, view off | +222.1 % | 1.79 | −23.0 % | 3,018 |
| `best` | the selected configuration (`live.py`): daily, `close`, 50-day index-vs-MA view EXIT_ONLY | +202.0 % | 1.71 | −23.1 % | 2,990 |

Read the numbers with the caveats the strategy README states: the sector snapshot is current, not point-in-time
(survivorship measured at ~1.9 %); one window with no holdout; daily cadence pays ~29 % of capital in costs over
the window against monthly's 8 %, buying the drawdown difference. The view's price on this window (−20 pp for no
drawdown change) is the expected shape — a bear-market mechanism in a window with little bear market in it.

Reproduce: `python variations/<variant>/run.py` from a checkout with the package installed; it rewrites that
variation's `results/` (eight files, provenance hashes every input) and regenerates `report.html`.
