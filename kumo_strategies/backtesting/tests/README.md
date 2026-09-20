# tests/backtesting

Tests for the backtest machinery: FIFO round-trip matching, cost apportionment, KPI computation,
the deflated Sharpe, and the measured-spread cost model.

Each test pins a way a number has been wrong or could silently lie — untraded sessions diluting
Sharpe, cost not split across partial exits, a flat cost model hiding that the open is 3× dearer
than midday. Add a test here whenever a reported figure turns out to have been misleading.

Does NOT hold: strategy-rule tests (→ `tests/strategies`), Nautilus adapter tests (→ `tests/runtime`).
