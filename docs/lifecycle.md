# Strategy Lifecycle

A strategy moves through these gates:

1. Research candidate in `research-lab` or prior evidence in `kumo-qc`.
2. Pure decision implementation in `kumo_strategies/strategies/`.
3. Fixture-backed tests for decision behavior.
4. Replay/backtest runner using the same contracts.
5. Performance ledger with explicit fees, slippage, and liquidation assumptions.
6. Nautilus adapter smoke test.
7. Paper-trading gate through Cockpit.
8. Live gate with reconciliation and rollback notes.

A result is not portable until the decision contract and the replay runner use the same inputs, outputs, and identity fields expected by Cockpit.
