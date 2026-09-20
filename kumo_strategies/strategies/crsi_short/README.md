# crsi_short — CRSISHORT

The pure decision layer for the overbought/high-volatility short book (#123): ConnorsRSI(3,2,100) > 90
on liquid, >100%-vol US equities, entered with a sell-short limit 3% above the signal close and
covered on a reversal or on a return to entry.

- `config.py` — `CrsiShortConfig`, frozen. Every swept field lives here, including the entry limit.
- `indicators.py` — ConnorsRSI and its parts, plus annualised log volatility. Per-symbol series only.
- `engine.py` — `build_feature_panel` (features as of the prior close, split-adjusted prices required)
  and `decide` (entries, limit prices, borrow refusals). No exits come out of `decide`.
- `screen.py` — `screen_universe`: the daily screen the lane must be handed (every name eligible in the
  last N sessions), asked of the same gates the backtest uses. A list of past trades is not a universe.
- `exits.py` — the reversal and flat covers, short side. The give-back arithmetic is shared with the
  long lanes in `strategies/give_back.py`.

Belongs here: decision logic that a backtest and the live runner must share, with no I/O.
Does not belong here: order submission, price fetching, Nautilus types, database access, a clock.
