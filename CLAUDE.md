# CLAUDE.md — kumo-trading-strategies

The strategy library for `kumo-trading-platform`: every strategy as its pure decision, its NautilusTrader lane, its
tests and its backtest sets in one folder (`strategies/<name>/`); the shared runtime the platform imports
(`runtime/`); the ONE backtest runner every strategy's evidence is produced by (`backtesting/`). Python import name:
`kumo_strategies`. Backtest data lives in each set (parquet through Git LFS); exploratory research does not live here.

## Rules

**Check Nautilus first — every time.** Before writing an adapter, a cache, a reconciliation step or an
order-state machine, read what NautilusTrader already provides. Out-of-tree code exists only where
Nautilus has no mechanism, and the comment at the top of that code says which mechanism was checked
and why it did not fit.

**Every bug gets a test.** The test is written before the fix and seen failing. A test written after
the fix is a guess about what it covers. Revert the fix, watch the test go red, restore. For a result
rather than a code path, break the input deliberately and confirm the number moves.

**Agreement is not connection.** Two components that agree on a value are not thereby connected. A
test that both sides pass independently proves nothing about the seam. Test the seam: call the real
thing with the real argument and assert the real effect.

**A fallback is a silent wrong answer.** `x or default` cannot tell "unset" from "zero". A branch
nobody can reach is worse than no branch. When a value is absent, raise or report — never substitute.
Reserve fallbacks for cases where the fallback is genuinely as correct as the primary, and say so in a
comment, because the next reader cannot tell a considered fallback from a lazy one.

**Absence is not permission.** A gate that defaults open because its config is missing is not a gate.
Every gate defaults closed. A detector that recognises its subject by a property the defect destroys
can never fire. Enforce properties at the value, not by enumerating callers.

**Structured reports.** Status, defects and decisions are structured, not prose: `What · Cause ·
Impact now · Impact if ignored · Fix · Risk · Decision needed`. Drop any line without content. Lead
with the problem, not the investigation. Numbers over adjectives. Never present a menu without a
recommendation.

**Verify by disagreement.** Two derivations of one fact are a detector. Identical when they should
differ means a dead mechanism; differing when they should match means a live defect. Print both.

## Conventions

- Python 3.13, `uv` for environments, `pytest` for tests, `ruff` for lint. Tests live next to the
  code they test or under `tests/` mirroring the source tree.
- Conventional Commits. No secrets, account ids, hostnames or personal identifiers anywhere in the
  tree — `bin/check-public-tree.sh` enforces it.
- Every non-hidden directory has a `README.md`: what it holds, what goes in, what does not.
- Comments in English; they say why, not what.
