from __future__ import annotations

import pytest

from kumo_strategies.contracts import StrategyIdentity
from pathlib import Path


def test_strategy_identity_derives_strategy_name() -> None:
    identity = StrategyIdentity(
        account_id="paper",
        client_id="alpaca-paper",
        instrument_id="AAPL.NASDAQ",
        strategy_id="MOMENTUM-002",
    )

    assert identity.strategy_name == "MOMENTUM"


def test_strategy_identity_requires_cycle_for_attribution() -> None:
    identity = StrategyIdentity(
        account_id="paper",
        client_id="alpaca-paper",
        instrument_id="AAPL.NASDAQ",
        strategy_id="MANUAL-001",
    )

    with pytest.raises(ValueError, match="cycle_id is required"):
        identity.require_cycle()

    assert identity.with_cycle("cycle-1").require_cycle().cycle_id == "cycle-1"


def test_NO_component_has_a_default_strategy_identity():
    """There is nothing left to drift, which is a stronger guarantee than agreeing.

    THE ORIGINAL DEFECT: the tag moved 001 -> 002 and only some defaults followed. The journal kept
    writing under MOMENTUM-001 while position claims and Nautilus positions were under MOMENTUM-002.
    Each side was self-consistent, so nothing raised — the audit trail simply described a strategy
    that, by id, held nothing. This test used to assert the three defaults AGREED.

    Agreeing was never the property that mattered. On 2026-08-22 all three agreed perfectly, on
    "MOMENTUM-002", and `pgrunner`'s BUY path did not pass one — so every entry from BCTROT-004 and
    QC345-003 was built claiming to be MOMENTUM-002. Three defaults in perfect agreement about the
    wrong lane. The guard was green throughout.

    A DEFAULT IDENTITY IS A WRONG ANSWER WITH GOOD MANNERS (kumo-trading-platform's phrase, 2026-08-22): it is
    a real, live, position-holding lane, well-formed everywhere it is read, byte-identical in every
    derived artifact. Nothing downstream can tell it was defaulted. So the fix is not a better
    default, it is no default — a `TypeError` or `ValueError` at construction is the one failure mode
    that cannot be mistaken for correct behaviour.

    `PgSessionRunner` validates rather than omitting because its `strategy_id` sits after fields that
    do have defaults, and a non-default field cannot follow one; reordering would silently change
    what every positional caller means.
    """
    import dataclasses

    from kumo_strategies.runtime.executor.broker import OrderRequest
    from kumo_strategies.runtime.executor.pgjournal import PgJournal
    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner

    for cls in (PgJournal, OrderRequest):
        field = cls.__dataclass_fields__["strategy_id"]
        assert field.default is dataclasses.MISSING, (
            f"{cls.__name__}.strategy_id has default {field.default!r} — a lane id as a default is "
            f"undetectable downstream")

    assert PgSessionRunner.__dataclass_fields__["strategy_id"].default == "", (
        "PgSessionRunner.strategy_id must default to the empty sentinel, never a lane")


def test_a_runner_built_without_a_lane_REFUSES_to_exist():
    """It would not merely mislabel orders — it would read another lane's book.

    `self.strategy_id` drives `PositionState.strategy_id == self.strategy_id` and
    `StrategyState.strategy_id == self.strategy_id`, so a runner that defaulted to MOMENTUM-002 would
    query MOMENTUM's positions and lifecycle, then act on them. Confirmed against cockpit's live
    journal on 2026-08-22: four distinct ids appear (BCTROT-004 x22, MOMENTUM-002 x134, QC345-003 x1,
    TECHIVOL-005 x9), so no construction site relies on a default today. This is what keeps that true.
    """
    import pytest as _pytest

    from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner

    with _pytest.raises(ValueError, match="strategy_id"):
        PgSessionRunner(pool=None, journal=None, lifecycle=None, cfg=None)


def test_a_journal_built_without_a_lane_REFUSES_to_exist():
    """Every audit row carries this id. A defaulted one files another lane's history under MOMENTUM."""
    import pytest as _pytest

    from kumo_strategies.runtime.executor.pgjournal import PgJournal

    with _pytest.raises(TypeError):
        PgJournal(None)                                          # type: ignore[call-arg]


def test_this_repo_does_not_ship_its_own_operator_ui() -> None:
    """The operator surface is the COCKPIT. There must not be a second one here.

    There was: a hand-written page plus its own HTTP server lived in this package, 922 lines that
    nothing imported. It duplicated the cockpit, spoke a different design language, and had to be put
    on a tailnet before it was reachable from a phone at all — while carrying an unauthenticated
    control that could liquidate the book. It also drifted: it reported DISABLED for a TRADING
    strategy for a whole session because its strategy_id was never moved from MOMENTUM-001.

    The pool/lifecycle DOMAIN logic stays here — pins, excludes, halt, liquidate. What must not come
    back is a second place to look at it.
    """
    import kumo_strategies.runtime.executor as executor

    root = Path(executor.__file__).parent
    forbidden = [p.name for p in (root / "api.py", root / "serve.py", root / "web") if p.exists()]
    assert not forbidden, (
        f"a second operator UI reappeared in kumo-trading-strategies: {forbidden}. "
        f"Pool and lifecycle surfaces belong in kumo-trading-platform (GET /pool, the Pool tile)."
    )
