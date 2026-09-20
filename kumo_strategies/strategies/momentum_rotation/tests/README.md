# tests/strategies/momentum_rotation

Deterministic tests for the momentum-rotation decision engine.

Each test pins a behaviour whose absence produced a false research result — splits, leveraged
products, score ties, pool leakage. They are regressions against real bugs, not hypotheticals.

Does NOT hold: backtest runs, data fixtures pulled from disk, or anything needing a live feed.
