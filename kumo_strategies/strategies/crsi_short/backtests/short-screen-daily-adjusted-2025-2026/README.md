# short-screen-daily-adjusted-2025-2026

CRSISHORT — the mean-reversion short: ConnorsRSI > 90 on a liquid, volatile name at the close, a short limit 3 %
above that close resting through the next session, covered on reversal or flatness — on Alpaca SIP daily bars with
`adjustment=all`, 2024-06-03 → 2026-09-08 (trading 2025-01-01 →; the first 100 sessions warm the indicators).

The set IS the data: `bars.parquet` — 291 symbols, split- and dividend-adjusted OHLCV. Adjusted, not raw, because a
short books a reverse split on raw prices as a −1000 % loss; the raw store inverted this strategy's verdict once
(the strategy README tells that story). `run.py` rebuilds the strategy's feature panel from these bars with the
strategy's own code and hands the panel to the runner, so the backtest evaluates the code the lane trades.

Why 289 symbols: the screen runs over ~11,000 manageable names, but only the names that ever produced an entry
signal (or carried one overnight) in the window can touch the book. The set carries exactly those, plus SPY, QQQ and GLD as benchmarks — the
full-panel run and this set agree on return, drawdown, Sharpe and trade count. `manifest.json` records the extract.

| variation | what | total return | Sharpe | max DD | trades | borrow paid |
|---|---|---|---|---|---|---|
| `best` | the frozen acceptance candidate: 20 slots, hold-through, 25 bps a side, 20 %/yr borrow, no borrow-fee gate | +80.8 % | 2.35 | −8.1 % | 231 | $33,772 |
| `costs-200bps` | the same at 200 bps a side | +22.1 % | 0.83 | −14.0 % | 231 | $27,599 |

Read with the strategy README's caveats: no locate data in the window (the borrow gate is OFF here, stated in the
parameters), the fill model is `max(limit, open)` when the session's high reaches the limit,
one window, no holdout. The acceptance script's mutation arms (CRSI > 99, no volatility filter, market entry,
exits off) all move the result; they are tests, not variations, and live in the strategy's `tests/`.

Reproduce: `python variations/<variant>/run.py` from a checkout with the package installed; it rewrites that
variation's `results/` and regenerates `report.html`.
