"""`OrderRequest.strategy_id` is about to become an AUTHORISATION, not a label.

kumo-trading-platform issue 462: `NautilusBroker.exit` calls `feed.release_for_exit(iid, qty)` with no owner, so
cockpit resolves it internally as `self.id` on the feed — MANUAL-001. Every exit from every lane
therefore arrives as MANUAL-001, and cockpit cannot tell "MOMENTUM exiting its own stop" from
"MOMENTUM exiting BCTROT's". Those are the same call. That is the #437 incident, mechanically.

The proposed fix threads `strategy_id=req.strategy_id` through as the CANCELLER. Once it lands, this
field stops being a label on a journal row and starts deciding whose protective stop may be
cancelled.

WHICH MAKES THE DEFAULT A TRAP. `OrderRequest.strategy_id` defaults to `DEFAULT_STRATEGY_ID`, which
is the string "MOMENTUM-002" — a real, live, position-holding lane, not a sentinel. Cockpit's design
keeps a permissive "proxy" concession when the owner arrives as None, on the reasoning that None
means "the feed could not name the lane". THIS REPO CAN NEVER SEND NONE. A construction site that
forgets the argument sends a FALSE BUT ENTIRELY PLAUSIBLE claim to be MOMENTUM-002 — and under the
new rule that authorises cancelling MOMENTUM-002's protection.

So the failure mode changes from "over-permissive in a known way" to "one lane silently authorised to
cancel another lane's stops", which is the exact incident the change exists to prevent.

Every site names it correctly today. This is what keeps that true, since the defect it guards is
invisible: the wrong value is a valid lane id and nothing downstream can tell it was defaulted.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

# One folder per strategy (ks#211): tests, backtests and studies live INSIDE the package tree now.
# An audit over package SOURCE must not read them as source.
_NONSRC = {"tests", "backtests", "studies", "__pycache__"}


def _src_only(paths):
    return [p for p in paths if not (_NONSRC & set(p.parts))]


_SRC = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "kumo_strategies" / "runtime"


def _order_request_sites():
    """Every `OrderRequest(...)` construction under `runtime/`, discovered."""
    out = []
    for path in _src_only(sorted(_SRC.rglob("*.py"))):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name != "OrderRequest":
                continue
            out.append((str(path.relative_to(_SRC)), node.lineno,
                        {k.arg for k in node.keywords if k.arg}))
    return out


def test_there_are_order_request_sites_to_check():
    assert _order_request_sites(), "no OrderRequest construction found — this guard is looking in the wrong place"


@pytest.mark.parametrize("site", _order_request_sites(), ids=lambda s: f"{s[0]}:{s[1]}")
def test_every_order_request_NAMES_its_strategy(site):
    """Omitting it does not fail — it silently claims to be MOMENTUM-002.

    `strategy_id` also feeds `client_order_id`, which is the replay guard and the only thing tying a
    broker order back to its owner, so a defaulted value is wrong in the journal, wrong in the order
    id, and — once #462 lands — wrong about whose stop may be cancelled. Three consequences, one
    omission, none of them visible at the call.
    """
    path, line, kwargs = site
    assert "strategy_id" in kwargs, (
        f"{path}:{line} builds an OrderRequest without naming a strategy, so it inherits the "
        f"hardcoded default 'MOMENTUM-002' — a real lane, not a sentinel")


def test_two_lanes_buying_the_same_symbol_do_not_share_an_order_id():
    """The consequence of a defaulted `strategy_id`, stated as behaviour rather than as a call shape.

    `client_order_id` hashes `(strategy_id, session, symbol, side[, slot][, attempt])`. Two lanes
    that both defaulted to "MOMENTUM-002" and reached the same symbol in the same session and slot
    would produce the SAME id — and a duplicate id is denied by Nautilus LOCALLY, before it ever
    reaches the venue, so the second lane's entry simply does not happen.

    That is not hypothetical arithmetic. It has already cost a live position once, in the slot form:
    MOMENTUM-002's open+5m attempt for FSM and its open+130m retry hashed identically on 2026-08-19
    because neither `session` nor `attempt` had changed. And BCTROT-004 and MOMENTUM-002 currently
    hold SIX symbols in common (AEM, AMGN, BDX, CGAU, WHD, WPM), so the overlap this needs is not
    rare — only the shared slot is, and a slot is a settings edit away.
    """
    from kumo_strategies.runtime.executor.broker import OrderRequest

    a = OrderRequest("WHD", "BUY", 10, "2026-08-24", strategy_id="MOMENTUM-002", slot="open+5m")
    b = OrderRequest("WHD", "BUY", 10, "2026-08-24", strategy_id="BCTROT-004", slot="open+5m")
    assert a.client_order_id != b.client_order_id, (
        "two lanes buying one symbol in one slot hash to the same order id; the second is denied "
        "locally as a duplicate and never reaches the venue")


def test_an_order_request_CANNOT_be_built_without_naming_its_strategy():
    """The default is gone. Agreed with kumo-trading-platform 2026-08-22 for #462.

    Their words, and the reason: "a default that is a real position-holding lane is not a default, it
    is a wrong answer with good manners."

    Cockpit's design keeps a permissive `proxy` concession when the canceller arrives as None, on the
    reasoning that None means "the feed could not name the lane". That branch was UNREACHABLE from
    this repo while the default was "MOMENTUM-002" — a forgotten argument produced a well-formed
    claim to be a real lane instead of an absent one, and under #462 that authorises cancelling that
    lane's protective stops.

    A `TypeError` at construction is the whole point. It is the one failure mode that cannot be
    mistaken for a correct order, which is exactly what the old default could be.
    """
    import pytest as _pytest

    from kumo_strategies.runtime.executor.broker import OrderRequest

    with _pytest.raises(TypeError):
        OrderRequest("WHD", "BUY", 10, "2026-08-24")            # type: ignore[call-arg]


def test_a_defaulted_strategy_is_INDISTINGUISHABLE_from_a_deliberate_one():
    """Why the guard above has to be structural rather than a runtime check.

    Nothing downstream can tell a defaulted `strategy_id` from a correct one: the value is a real,
    live, position-holding lane id, and it is well-formed everywhere it is read. There is no
    validation that could catch it — which is why the assertion is on the CALL SITES.
    """
    from kumo_strategies.runtime.executor.broker import OrderRequest

    wrong = OrderRequest("WHD", "BUY", 10, "2026-08-24", strategy_id="MOMENTUM-002")
    right = OrderRequest("WHD", "BUY", 10, "2026-08-24", strategy_id="MOMENTUM-002")
    assert wrong.client_order_id == right.client_order_id, (
        "the value that used to arrive by default is byte-identical to one chosen deliberately — "
        "which is why no runtime check could ever have caught the entry-path defect, and why the "
        "guard has to be structural")
