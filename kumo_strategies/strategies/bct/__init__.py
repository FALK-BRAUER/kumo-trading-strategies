"""BCTROT — the BCT rotation. A SCHEDULE of the momentum rotation, not a second strategy.

The decision, the pool, the exits and the reconciliation are `strategies.momentum_rotation`'s; this
package holds only the lane that runs them three times a session (`nautilus.py`) and its tests.
Its backtest evidence is the `bctrot` variation of `momentum_rotation/backtests/ledger-book-daily-2024-2026`.
"""
