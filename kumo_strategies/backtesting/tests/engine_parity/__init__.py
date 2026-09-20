"""The second derivation, kept as a TEST (#270, fold 3).

`backtesting/runner.py` pushed the real `MomentumRotationStrategy` through a Nautilus
`BacktestEngine`. It was not a simulator; its value was disagreement with the pandas runner — the
shape that found #40's liquidity look-ahead and CRSISHORT's four engine defects, none by inspection.
Falk's ruling is one runner, so it is not a runner any more; what survives is the comparison, run in
its own process (a second BacktestEngine per interpreter aborts natively) by `replay.py`, and
asserted by `test_engine_parity.py`.
"""
