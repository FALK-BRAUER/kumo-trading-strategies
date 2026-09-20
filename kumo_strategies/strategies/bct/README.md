# bct

BCTROT — the BCT rotation: the momentum rotation deciding three times a session (09:35, 12:00, 15:40) instead of
once. It is a SCHEDULE of `momentum_rotation`, not a second strategy — same candidate pool, same scoring, same exits,
same reconciliation — so this folder holds only what differs: the lane (`nautilus.py`, a subclass of the momentum
lane that passes its own `decision_slots` and identity) and its tests.

Backtest evidence lives where the decision lives: `momentum_rotation/backtests/ledger-book-daily-2024-2026/variations/bctrot`
(BCTROT's clock and gap rule on the same ledger book — a not-yet-successful variation of momentum, measured beside
`baseline`). There is deliberately no `backtests/` here; a second copy of the momentum set would be the drift this
package's docstring warns about.
