# liquid-top100-daily-2024-2026

QC345 — monthly momentum rotation, top-down universe selection, trail between rebalances — on Alpaca SIP daily
bars, 2024-01-02 → 2026-08-10 (the first 252 sessions are momentum warm-up; trading starts 2025).
The set IS the data: `bars.parquet` (266 symbols; raw OHLCV, the provider's adjusted close as `close_adj`, and the
rebuilt `close_split` / `close_split_dividend` series for the 165 names that ever qualified — the field the momentum
reads), `assets.parquet` (exchange / status / name, the fundamental-like filter's input), `instruments.json`
(venue metadata for sizing), `half_spreads.json` (the measured half-spread cost model, 58 % coverage here).

Why 266 symbols and not the whole market: the universe is chosen per month as the top 50 by price × dollar volume
among the 100 most liquid names. The set carries every name that ever entered a monthly top-100 liquidity set on the
full 16,474-symbol panel (212), so the ranking is unchanged — the full-panel run and this set agree to the cent.
`manifest.json` records the extract and the source hashes.

| variation | what | total return | Sharpe | max DD | round trips |
|---|---|---|---|---|---|
| `best` | package defaults, fully invested — the family's recorded result | +176.7 % | 1.55 | −33.2 % | 85 |
| `cash-buffer-0.75` | the same with a 25 % cash buffer (75 % invested) | +121.9 % | 1.54 | −25.7 % | 87 |

Caveats the strategy README states apply: `market_cap_mode=price_x_dv` is a liquidity proxy for market cap, not
shares outstanding; the split/dividend series exist only for names that qualified in the rebuild, so a name outside
the 165 cannot be selected even when it ranks; one window, no holdout; 48 trials corrected for in the deflated Sharpe.

Reproduce: `python variations/<variant>/run.py` from a checkout with the package installed; it rewrites that
variation's `results/` and regenerates `report.html`.
