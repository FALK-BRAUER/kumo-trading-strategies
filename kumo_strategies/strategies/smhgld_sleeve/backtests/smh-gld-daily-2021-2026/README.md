# smh-gld-daily-2021-2026

The SMH/GLD sleeve — 48 % semiconductors, 52 % gold, rebalanced at the close on every session the drift leaves a
0.25 % band — on split-adjusted daily closes, 2021-01-04 → 2026-09-10. The set IS the data: `bars.parquet` holds
the two legs' closes (the sleeve reads one price field) plus SPY and QQQ closes as benchmarks; `manifest.json` records the snapshot.

The sleeve family of the one runner reproduces the lab's fixed-weight walk on both pinned windows inside the
harness's tolerance (CAGR ± 1.5, Sharpe ± 0.08, max DD ± 1.0) — `tests/test_the_set_reproduces_the_research_on_both_windows.py`
is that gate, computed with the harness's own formulas from the tracked `equity.csv`:

| variation | window · book | CAGR | Sharpe | max DD | rebalances | research said |
|---|---|---|---|---|---|---|
| `best` | 2024-01-02 → · $1 M | 46.3 % | 1.74 | −15.0 % | 260 | 45.9 / 1.73 / −14.8 |
| `since-2021` | 2021-01-04 → · $1 M — the window with 2022 in it | 25.1 % | 1.16 | −27.8 % | 551 | 25.8 / 1.19 / −27.6 |
| `book-20k` | 2024-01-02 → · $20 k book | 46.1 % | 1.74 | −14.8 % | 307 | — |

(CAGR above is the harness's formula, sessions / 252; `metrics.json` carries the Report's calendar CAGR, a little higher.)

What `book-20k` shows: on a $20 k book a share of SMH is ~1 % of the sleeve, above the drift band, so
integer rounding itself triggers rebalances — 307 against 260 on the same window. The result barely moves at
10 bps; it is the fill count that changes, and that is what the lane pays for.

Not modelled here, stated rather than implied: the lane's declared market view (50-day index-vs-MA, EXIT_ONLY)
— the research walk never read it and the sleeve family does not either, so these are the no-view numbers.
Costs are a flat 10 bps a side; a market-on-close fill sized twenty minutes earlier is the residual
`live_notes()` in `live.py` names. Two legs, one window, no holdout.

Reproduce: `python variations/<variant>/run.py` from a checkout with the package installed.
