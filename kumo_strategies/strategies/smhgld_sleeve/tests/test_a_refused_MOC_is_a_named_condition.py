"""A rebalance the venue refused must not read like a lane that decided nothing (#177, platform issue 965).

THE LANE FILLS MARKET-ON-CLOSE and a MOC has a CUTOFF — IB ~15:50 ET, Alpaca 15:45 ET, both
documented rather than measured. The decision slot is `close-20m` (15:40), which clears both by
+10m and +5m. That margin is the whole safety, and it is computed against a NORMAL session.

ON A HALF DAY THE SLOT MOVES AND THE CUTOFF MAY NOT. `close-20m` resolves to 12:40 against a 13:00
close, correctly; whether either venue shifts its MOC cutoff by the same three hours is UNKNOWN.
Roughly nine sessions a year, all low-volume, and at a 0.25pp band a missed rebalance is caught by
the next session — so the cost is bounded and the coordinator has accepted it.

WHAT IS NOT ACCEPTED IS MISSING IT SILENTLY, and that is what this file is for. "The venue refused
the MOC" and "the lane decided nothing" must be DIFFERENT READINGS. Today they are the same one: a
rejection is forgotten and nothing distinguishes the order type that was refused, so nine sessions a
year could drift unrebalanced and every surface would show a lane behaving normally.

"Nobody would notice" is the reason to make it noticeable, not a reason to accept it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _Host:
    """The REAL methods off `SmhGldSleeveStrategy`, bound to a narrow host.

    `Actor.log` is a read-only Cython attribute, so a constructed strategy cannot have its logger
    replaced — the repo's established pattern. Every entry below is the attribute itself, so
    renaming or deleting one fails these tests rather than exercising a copy.
    """

    from kumo_strategies.strategies.smhgld_sleeve.nautilus import SmhGldSleeveStrategy as _S

    on_order_rejected = _S.on_order_rejected
    _name_a_refused_close = _S._name_a_refused_close
    _forget = _S._forget
    _claim_qty_after_event = _S._claim_qty_after_event
    _CLOSING_TIF = _S._CLOSING_TIF
    id = "SMHGLD-903"
    del _S


def _lane():
    lane = _Host()
    lane.said = []
    lane._held = set()
    lane._pending = {}
    lane.cache = SimpleNamespace(positions_open=lambda **kw: [])
    lane._runner = None
    lane._loop = None
    lane.log = SimpleNamespace(info=lambda *a, **k: None,
                               warning=lambda m, *a, **k: lane.said.append(("warning", m)),
                               error=lambda m, *a, **k: lane.said.append(("error", m)))
    return lane


def _rejection(sym, *, tif="AT_THE_CLOSE", reason="MOC order received after cutoff"):
    return SimpleNamespace(
        instrument_id=SimpleNamespace(symbol=sym),
        client_order_id=f"SMHGLD-903-{sym}",
        order_side=SimpleNamespace(name="BUY"),
        time_in_force=SimpleNamespace(name=tif),
        reason=reason)


def test_the_DEFAULT_decision_slot_clears_BOTH_documented_MOC_cutoffs():
    """`close-20m`, not `close-10m`. Measured on a live instance resolver: `close-10m` is 15:50, which is
    EXACTLY IB's documented cutoff and five minutes PAST Alpaca's — zero margin, and a rejection at
    15:50 leaves no session to retry in. The same mistake as `open-5m` against the OPG cutoff."""
    import inspect
    from datetime import datetime

    from kumo_strategies.strategies.smhgld_sleeve.nautilus import SmhGldSleeveStrategy
    from kumo_strategies.strategies.momentum_rotation.slots import resolve

    default = inspect.signature(SmhGldSleeveStrategy.__init__).parameters["decision_slots"].default
    assert default == ("close-20m",), default

    (_name, when), = resolve(default, datetime(2026, 9, 11, 9, 30), datetime(2026, 9, 11, 16, 0))
    assert when == datetime(2026, 9, 11, 15, 40)
    assert when < datetime(2026, 9, 11, 15, 45), "no margin against Alpaca's documented cutoff"
    assert when < datetime(2026, 9, 11, 15, 50), "no margin against IB's documented cutoff"


def test_a_refused_CLOSING_order_is_NAMED_not_merely_forgotten():
    """The condition the coordinator set. A generic 'rejected' row makes a missed rebalance look
    like an ordinary refusal, and an absent row makes it look like no decision."""
    lane = _lane()
    lane.on_order_rejected(_rejection("SMH"))
    said = " ".join(m for _lvl, m in lane.said)
    assert said, "the refusal was swallowed entirely"
    assert "SMH" in said
    assert "close" in said.lower() or "MOC" in said, (
        f"the message does not say the CLOSING order was refused: {said!r}")
    assert "rebalance" in said.lower() or "unrebalanced" in said.lower(), (
        f"the message does not say what was lost — that the rebalance did not happen: {said!r}")


def test_an_ORDINARY_rejection_is_not_dressed_up_as_a_missed_close():
    """THE CONTROL. If every rejection claimed the close was missed, the named condition would carry
    no information — which is the same defect as not naming it at all."""
    lane = _lane()
    lane.on_order_rejected(_rejection("GLD", tif="DAY", reason="insufficient buying power"))
    said = " ".join(m for _lvl, m in lane.said)
    assert "MOC" not in said and "cutoff" not in said.lower(), (
        f"a DAY rejection was reported as a missed close: {said!r}")


def test_the_two_readings_DIFFER():
    """Verify by disagreement. A refused MOC and a refused DAY order must not produce the same
    message — if they do, the naming is decorative."""
    moc, day = _lane(), _lane()
    moc.on_order_rejected(_rejection("SMH"))
    day.on_order_rejected(_rejection("SMH", tif="DAY", reason="insufficient buying power"))
    assert [m for _l, m in moc.said] != [m for _l, m in day.said]


def test_an_UNREADABLE_time_in_force_does_not_claim_the_close_was_missed():
    """Three states. An event whose TIF cannot be read is not evidence of a MOC rejection, and
    guessing would put a half-day incident in the log on an ordinary bad afternoon."""
    lane = _lane()
    ev = _rejection("SMH")
    ev.time_in_force = None
    lane.on_order_rejected(ev)
    said = " ".join(m for _lvl, m in lane.said)
    assert said, "an unreadable rejection was swallowed"
    # ASSERT ON OUR CLAIM, NOT ON THE ECHOED REASON. The venue's own text here says "MOC order
    # received after cutoff", so a bare `"MOC" not in said` fails on the string we are quoting
    # rather than on anything we asserted — the first version of this test did exactly that and
    # blamed correct code. The thing that must be absent is the sentence WE add.
    assert "THE REBALANCE DID NOT HAPPEN" not in said, (
        f"an unreadable TIF was reported as a missed close: {said!r}")
    assert lane.said[0][0] == "warning", "an unreadable TIF was escalated to error"
