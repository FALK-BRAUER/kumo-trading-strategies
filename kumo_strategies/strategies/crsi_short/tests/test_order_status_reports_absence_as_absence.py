"""The auction leg's status, with ABSENT kept distinct from every status (#131, #208).

`crsi_short.replacement_is_owed` decides at the post-cross slot whether the auction leg ended
WITHOUT taking the slot. It treats EXPIRED/CANCELED as "send the replacement" and everything else as
no — so "not in the cache" must not be folded into a status string: one direction invents a
replacement that would double the position, the other suppresses one that is owed.

!! AND THE ABSENT CASE DECIDES WHETHER THE DECOMPOSITION WORKS AT ALL. If Nautilus keeps a
venue-cancelled OPG order in the cache, the status is CANCELED and the replacement fires. If it
PURGES it, this returns None, `replacement_is_owed` answers False, and the replacement NEVER fires —
silently, every session, while the auction leg works and the rest of the intent is dropped. Nobody
in either repo knows which it is today, which is why the accessor logs a distinct line: the first
live session answers it by grep rather than by inference.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from nautilus_trader.model.identifiers import InstrumentId

from kumo_strategies.runtime.nautilus.broker import NautilusBroker
from kumo_strategies.strategies.crsi_short.nautilus import replacement_is_owed

AAPL = InstrumentId.from_str("AAPL.XNAS")


def _broker(order=None, *, raises=False):
    said: list[tuple[str, str]] = []

    def get(coid):
        if raises:
            raise RuntimeError("cache is gone")
        return order

    strat = SimpleNamespace(
        cache=SimpleNamespace(order=get),
        log=SimpleNamespace(info=lambda m: said.append(("info", m)),
                            warning=lambda m: said.append(("warning", m)),
                            error=lambda m: said.append(("error", m))))
    return NautilusBroker(strategy=strat, instrument_ids=[AAPL]), said


def _order(status_name):
    return SimpleNamespace(status=SimpleNamespace(name=status_name))


@pytest.mark.parametrize("name", ["EXPIRED", "CANCELED", "ACCEPTED", "FILLED",
                                  "PARTIALLY_FILLED", "REJECTED"])
def test_a_present_order_reports_its_status_VERBATIM(name):
    broker, _ = _broker(_order(name))
    assert broker.order_status("kumo-abc") == name


def test_an_ABSENT_order_is_None_and_NOT_a_status_string():
    """None is the only honest value. Any string would be read by `replacement_is_owed` as a venue
    answer it never gave."""
    broker, said = _broker(None)

    assert broker.order_status("kumo-abc") is None
    assert any("NOT IN CACHE" in m for _, m in said), said


def test_ABSENT_does_not_become_a_replacement():
    """The composition that matters: absent -> None -> no replacement. Refusing costs one session's
    entry; guessing costs an unbounded short."""
    broker, _ = _broker(None)
    assert replacement_is_owed(broker.order_status("kumo-abc")) is False


@pytest.mark.parametrize("name,owed", [("EXPIRED", True), ("CANCELED", True), ("CANCELLED", True),
                                       ("ACCEPTED", False), ("FILLED", False),
                                       ("PARTIALLY_FILLED", False), ("REJECTED", False),
                                       ("SOMETHING_NEW", False)])
def test_the_status_composes_with_replacement_is_owed_as_the_rule_states(name, owed):
    broker, _ = _broker(_order(name))
    assert replacement_is_owed(broker.order_status("kumo-abc")) is owed


def test_a_CACHE_THAT_CANNOT_ANSWER_is_logged_DIFFERENTLY_from_an_absent_order():
    """Both yield None — there is no safe third answer on the order path — but only one of them is a
    broken system, and an operator must be able to tell them apart."""
    broker, said = _broker(raises=True)

    assert broker.order_status("kumo-abc") is None
    errors = [m for lvl, m in said if lvl == "error"]
    assert errors and "could not read the cache" in errors[0], said
    assert not any("NOT IN CACHE" in m for _, m in said), (
        "a broken cache was reported with the same words as a missing order")


def test_it_NEVER_RAISES_because_it_runs_beside_other_symbols():
    broker, _ = _broker(raises=True)
    broker.order_status("kumo-abc")                      # must not raise


def test_an_order_with_NO_STATUS_is_absent_rather_than_an_empty_string():
    """An order object that carries no status cannot answer the question, and `""` would flow into
    `replacement_is_owed` as an unrecognised status — the same answer, reached dishonestly."""
    broker, _ = _broker(SimpleNamespace(status=None))
    assert broker.order_status("kumo-abc") is None


def test_it_reads_the_SAME_cache_seam_as_cancel():
    """Two lookup paths for one question drift. AST-bound on both methods."""
    import ast
    import inspect
    import textwrap

    def _calls(fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        return {n.func.attr for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}

    assert "order" in _calls(NautilusBroker.order_status)
    assert "order" in _calls(NautilusBroker.cancel)


# -- the pre-open slot, derived rather than flagged -----------------------------------------------

def test_has_pre_open_answers_from_the_SLOT_GRAMMAR_not_a_name_list():
    """`crsi_short._reaches_the_auction` records the first version of this mistake: a hand-written
    list of literal names that would have REFUSED `open-10m`, the very slot the decomposition
    needs."""
    from kumo_strategies.strategies.momentum_rotation.slots import has_pre_open

    assert has_pre_open(("open-10m", "open+5m")) is True
    assert has_pre_open(("open+5m",)) is False
    assert has_pre_open(("close-20m",)) is False
    assert has_pre_open(()) is False


def test_a_WALL_CLOCK_is_not_treated_as_pre_open():
    """`09:20` is before the open on an ordinary day and after it on a half-day with a shifted open,
    and this function has no calendar. Unclamping on it would be wrong on exactly the days that are
    already unusual."""
    from kumo_strategies.strategies.momentum_rotation.slots import has_pre_open

    assert has_pre_open(("09:20",)) is False


def test_an_UNPARSEABLE_spec_does_not_take_the_lane_down():
    """`validate` refuses a bad spec at construction. This helper runs on the arming path, where
    raising would stop a lane from scheduling — which looks exactly like a lane that decided to
    hold."""
    from kumo_strategies.strategies.momentum_rotation.slots import has_pre_open

    assert has_pre_open(("not-a-slot", "open-10m")) is True
    assert has_pre_open(("not-a-slot",)) is False


def test_the_pre_open_slot_RESOLVES_BEFORE_THE_OPEN_only_when_allowed():
    """The behaviour the derivation exists to get right: without it `open-10m` clamps FORWARD to the
    open, placing the auction leg after the OPG cutoff — a well-formed order the cross has already
    passed, failing nowhere."""
    from datetime import datetime

    from kumo_strategies.strategies.momentum_rotation.slots import has_pre_open, resolve

    specs = ("open-10m", "open+5m")
    open_at, close_at = datetime(2026, 9, 14, 9, 30), datetime(2026, 9, 14, 16, 0)

    clamped = dict(resolve(specs, open_at, close_at, allow_pre_open=False))
    assert clamped["open-10m"] == open_at, "the clamp this derivation exists to avoid is gone"

    allowed = dict(resolve(specs, open_at, close_at, allow_pre_open=has_pre_open(specs)))
    assert allowed["open-10m"] < open_at
    assert allowed["open+5m"] > open_at, "the post-cross slot moved too"


def test_BOTH_call_sites_derive_it_so_they_cannot_disagree():
    """`allow_pre_open` must reach `next_slot_fire` AND `elapsed_slots` or neither: a lane that arms
    on a pre-open slot while its missed-slot detector clamps that slot forward reports a slot it
    never had. AST-bound on both, because a flag passed at one site and forgotten at the other is
    silent."""
    import ast
    import inspect
    import textwrap

    from kumo_strategies.runtime.nautilus.contract import RegistrationMixin
    from kumo_strategies.strategies.crsi_short.nautilus import CrsiShortStrategy

    def _kwargs_of(fn, callee):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == callee):
                return {kw.arg for kw in node.keywords}
        return None

    arming = _kwargs_of(CrsiShortStrategy._arm, "next_slot_fire")
    assert arming is not None and "allow_pre_open" in arming, arming

    owner = next(k for k in CrsiShortStrategy.__mro__ if "report_missed_on_start" in k.__dict__)
    missed = _kwargs_of(owner.__dict__["report_missed_on_start"], "elapsed_slots")
    assert missed is not None and "allow_pre_open" in missed, missed
