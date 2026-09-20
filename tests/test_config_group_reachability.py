"""#26, widened — a config GROUP that nothing reads at all.

The audit so far found fields that were wired but unreachable in some callers: `max_correlation` and
`inverse_vol_sizing` (no `corr`/`vol` passed), the whole of `IntradayConfig` (no `overlay`),
`ExitConfig` in the Nautilus adapters (no `evaluate_exits` call). Each of those has at least one path
where it works.

`ExecutionConfig` has none. `grep -rn "\\.execution" src/ tests/ research/` returns nothing. Its only
two mentions anywhere are its own class statement and the `field(default_factory=ExecutionConfig)`
that instantiates it. Three fields, zero reads:

  `fill = "next_open"`         The backtest DOES fill at the next open — via `runner_verified.run`'s
                               own `entry: str = "next_open"` argument. The config field is a second
                               copy of that decision that nothing consults, so setting
                               `execution.fill = "close"` changes nothing and reports no error.
                               That is a look-ahead trap in the shape #10 warns about: the field
                               that appears to control fill timing does not.

  `cost_bps = 10.0`            `runner_verified` charges from the measured per-symbol `CostModel`,
                               not a flat rate. The docstring — "the measured result survived
                               40bps" — reads as a live sensitivity claim about a number that is not
                               connected to anything.

  `gap_through_stop`           Documents what happens when a bar opens through a stop. **There is no
                               stop mechanism in the backtest at all**; `grep -n stop
                               runner_verified.py` is empty. This is not an unwired flag, it is a
                               config field describing a feature that was never built.

That last one is a different failure from the rest of the audit, and worse for a reader: anyone
opening `MomentumRotationConfig` would reasonably conclude the backtest handles stops.

NOT FIXED HERE. Deleting or wiring `ExecutionConfig` is a public API change — Cockpit imports this
module (`kumo-trading-platform/backend/strategies/momentum.py:224`), and #27 is open on how config surface is
declared and versioned across that boundary. Recorded and pinned, same treatment as `max_weight`.
"""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

# One folder per strategy (ks#211): tests, backtests and studies live INSIDE the package tree now.
# An audit over package SOURCE must not read them as source.
_NONSRC = {"tests", "backtests", "studies", "__pycache__"}


def _src_only(paths):
    return [p for p in paths if not (_NONSRC & set(p.parts))]


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SRC = ROOT / "kumo_strategies"

#: group attribute on MomentumRotationConfig -> is it read ANYWHERE in the package?
#:
#: `False` is a finding, not a target. Flip a row only alongside the code change that earns it.
EXPECTED_REACHABLE = {
    "score": True,
    "intraday": True,        # reachable, though `overlay` gates most of it — see the wiring audit
    "portfolio": True,
    "exits": True,           # reachable via evaluate_exits in 2 of 4 drivers — see the wiring audit
    "gates": True,
    "execution": True,       # was DEAD; `fill` is now consulted by runner_verified
    "market_view": True,     # read by run_sessions and both verified runners (#147)
}


def _src_files() -> list[Path]:
    return [p for p in _src_only(SRC.rglob("*.py")) if "__pycache__" not in p.parts]


def _group_is_read(group: str) -> bool:
    """Does any source file read `<config holder>.<group>`?

    AST rather than a regex, because both obvious regexes are wrong in opposite directions:

      `\\.exits\\.`   misses `evaluate_exits(cfg.exits, ...)`, where the group is passed WHOLE and
                    never field-accessed. That is how `exits` first read as dead here.
      `\\.exits\\b`   matches `plan.exits` — the ExitPlan's own attribute, an unrelated object — and
                    would report the group live on the strength of a name collision.

    So: match any attribute access whose attribute name is the group AND whose base expression
    mentions a config holder (`cfg`, `_cfg`, `self.cfg`, `config`). That accepts the group being
    passed whole and rejects same-named attributes on other objects.
    """
    for p in _src_files():
        if p.name == "config.py":
            continue
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Attribute) and node.attr == group:
                base = ast.unparse(node.value).lower()
                if "cfg" in base or "config" in base:
                    return True
    return False


def test_every_config_group_is_accounted_for():
    """A new group added to MomentumRotationConfig must be classified here, or this fails."""
    declared = {f.name for f in fields(MomentumRotationConfig)
                if f.name not in ("candidate_source", "candidate_params")}
    assert declared == set(EXPECTED_REACHABLE), (
        f"config groups changed.\n  unclassified: {sorted(declared - set(EXPECTED_REACHABLE))}\n"
        f"  stale rows: {sorted(set(EXPECTED_REACHABLE) - declared)}")


@pytest.mark.parametrize("group", sorted(EXPECTED_REACHABLE))
def test_config_group_reachability_matches_the_audit(group):
    assert _group_is_read(group) is EXPECTED_REACHABLE[group], (
        f"`{group}` reachability changed. If it is now read, wire the row to True in the same "
        f"commit; if it stopped being read, that is a regression of the #26 class.")


def test_the_matcher_would_actually_detect_a_read():
    """Guard on the guard. `_group_is_read` returning False must mean absence, not a broken regex.

    Without this, a typo in the pattern would report every group dead and the parametrised test
    would fail loudly — but `test_execution_config_is_still_entirely_dead` would pass for the wrong
    reason, which is the failure mode this suite keeps finding in itself.
    """
    assert _group_is_read("portfolio"), "matcher cannot see a field-accessed group"
    assert _group_is_read("exits"), (
        "matcher cannot see a group passed WHOLE — `evaluate_exits(cfg.exits, ...)` never "
        "field-accesses it, and a `\\.exits\\.` regex reported it dead for exactly that reason")
    assert not _group_is_read("no_such_group_xyz"), "matcher reports a nonexistent group as read"
