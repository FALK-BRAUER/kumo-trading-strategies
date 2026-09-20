"""A lane must NOTICE a position on the side it cannot manage, not only refuse to exit one (#66).

`sides.closing_quantity` and `refusal_reason` already refuse a wrong-sided position — but only on
the exit path, and only when the lane is already trying to exit that name. WHD sat in the book for
twelve hours as `+28 BCTROT / −28 MOMENTUM` and nothing said so, because no lane was trying to exit
it. The refusal is correct and it is not a detector.

WHY IT CANNOT BE REPAIRED BY RECONCILIATION, which is what makes reporting the whole job here: the
broker has ONE net position per symbol and no opinion about whose it is. CLAUDE.md states it — "broker
net = the only hard reconciliation anchor; per-strategy split is unverified by the broker". WHD's
detector reported no drift and was RIGHT to: +28 and −28 net to 0, and the broker agreed. The split
beneath it is unanchored and always will be. So a wrong-sided position cannot be fixed automatically
and must not be traded on; the honest response is to say so, loudly, every session, until a human
resolves it.

DISCOVERED, NOT LISTED. `#138` is open because a hand-written lane list silently dropped BCTROT — it
inherits its handler rather than defining one. Every lane here is found by walking the package.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from kumo_strategies.strategies import _layout
from kumo_strategies.runtime.nautilus.contract import report_wrong_sided_positions
from kumo_strategies.runtime.nautilus.sides import LONG



def _lane_classes():
    seen, out = set(), []
    for mod in _layout.lane_modules():                      # ks#211: the ONE discovery point
        for _n, cls in inspect.getmembers(mod, inspect.isclass):
            if (cls.__module__.startswith("kumo_strategies.strategies.")
                    and hasattr(cls, "POSITION_SIDE") and cls not in seen):
                seen.add(cls)
                out.append(cls)
    return out


LANES = _lane_classes()


def test_the_discovery_found_the_lanes():
    """Guards the guard. An empty scan makes every parametrised test below vacuous."""
    assert len(LANES) >= 6, [c.__name__ for c in LANES]
    assert any("BCT" in c.__name__ for c in LANES), "BCTROT inherits; it went missing once already"


def _pos(symbol: str, signed: int):
    return SimpleNamespace(instrument_id=SimpleNamespace(symbol=symbol),
                           quantity=abs(signed), signed_qty=signed)


class _Journal:
    def __init__(self):
        self.rows = []

    async def write(self, kind, summary, *, session, detail=None, **kw):
        self.rows.append((kind, summary, detail or {}))


def _lane(cls, positions):
    said, journal = [], _Journal()
    lane = SimpleNamespace(
        id=f"{cls.__name__}-001", POSITION_SIDE=getattr(cls, "POSITION_SIDE", LONG),
        cache=SimpleNamespace(positions_open=lambda **kw: list(positions)),
        log=SimpleNamespace(info=lambda *a, **k: None,
                            warning=lambda m="", *a, **k: said.append(str(m)),
                            error=lambda m="", *a, **k: said.append(str(m))),
        _loop=object(),
        session_journal=lambda: journal,
        fire_and_report=lambda coro, loop, what: _drain(coro),
    )
    lane._said, lane._journal = said, journal
    return lane


def _drain(coro):
    try:
        coro.send(None)
    except StopIteration:
        pass


@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_a_wrong_sided_position_is_REPORTED_even_though_nothing_is_exiting_it(cls):
    """THE WHD CASE. The lane is not trying to exit; it simply holds something it cannot manage."""
    side = getattr(cls, "POSITION_SIDE", LONG)
    wrong = -28 if side == LONG else 28
    lane = _lane(cls, [_pos("WHD", wrong)])
    found = report_wrong_sided_positions(lane, session="2026-09-11")
    assert found == ["WHD"], found
    assert lane._journal.rows, "held a position it cannot manage and wrote no durable row"
    kind, summary, detail = lane._journal.rows[0]
    assert kind == "risk", kind
    assert "WHD" in summary
    assert detail["symbol"] == "WHD" and detail["signed_qty"] == wrong


@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_a_correctly_sided_position_is_silent(cls):
    """A log line per held symbol per session buries the one that matters."""
    side = getattr(cls, "POSITION_SIDE", LONG)
    right = 28 if side == LONG else -28
    lane = _lane(cls, [_pos("AAPL", right)])
    assert report_wrong_sided_positions(lane, session="2026-09-11") == []
    assert lane._journal.rows == []


@pytest.mark.parametrize("cls", LANES, ids=lambda c: c.__name__)
def test_a_FLAT_position_is_silent(cls):
    lane = _lane(cls, [_pos("AAPL", 0)])
    assert report_wrong_sided_positions(lane, session="2026-09-11") == []
    assert lane._journal.rows == []


def test_the_message_says_what_an_operator_must_DO():
    """"wrong side" is not actionable. The row has to say that nothing in this lane could have
    opened it, so it came from reconciliation, a manual order or a phantom, and that it must be
    resolved outside the strategy — which is `sides.refusal_reason`'s wording, reused rather than
    written twice."""
    lane = _lane(LANES[0], [_pos("WHD", -28 if getattr(LANES[0], "POSITION_SIDE", LONG) == LONG else 28)])
    report_wrong_sided_positions(lane, session="2026-09-11")
    _kind, summary, detail = lane._journal.rows[0]
    assert "DOUBLE" in detail["reason"], detail
    assert "outside the strategy" in detail["reason"], detail


def test_it_reads_only_THIS_lane_s_positions():
    """The account's book is not ours. Reading it here would report another lane's short as our
    unmanageable position — which is #66's own defect, inverted: TECHIVOL proposed exiting eight
    names it did not own by reading the account as its own."""
    src = inspect.getsource(report_wrong_sided_positions)
    call = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "positions_open")
    assert any(k.arg == "strategy_id" for k in call.keywords), (
        "positions_open() without strategy_id reads the ACCOUNT's book, not this lane's")


def test_a_lane_with_no_journal_still_says_it_in_the_log():
    """Two lanes carry no `RegistrationMixin`, so `session_journal` does not exist on them. A
    position nobody can manage must not go unmentioned because the durable path is unreachable."""
    lane = _lane(LANES[0], [_pos("WHD", -28 if getattr(LANES[0], "POSITION_SIDE", LONG) == LONG else 28)])
    del lane.session_journal
    found = report_wrong_sided_positions(lane, session="2026-09-11")
    assert found == ["WHD"]
    assert any("WHD" in m for m in lane._said), lane._said


def test_every_lane_with_a_SESSION_PATH_reports_wrong_sided_positions():
    """COVERAGE, not existence — the rule `test_foreign_order_events` and #138 both exist for.

    A lane that computes a session and never asks whether it holds something it cannot manage is
    the WHD case waiting to happen again. Asserted on the AST of every module that defines a session
    coroutine, so a lane added tomorrow fails here rather than shipping without the check.

    BOTH CALLEE SHAPES are collected: `report_wrong_sided_positions(self, ...)` is an `ast.Name`,
    `self.report_wrong_sided_positions(...)` an `ast.Attribute`. Gathering only one is exactly the
    hole that let a regression ship green in #133.
    """
    missing = []
    for path in _layout.nautilus_sources():
        tree = ast.parse(path.read_text())
        if not any(isinstance(n, ast.AsyncFunctionDef) and n.name == "_session_coro"
                   for n in ast.walk(tree)):
            continue
        called = {(n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None))
                  for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if "report_wrong_sided_positions" not in called:
            missing.append(_layout.rel(path))
    assert not missing, (
        f"these run a session and never check whether they hold a position they cannot manage: "
        f"{missing}")


def test_the_coverage_scan_actually_FOUND_session_paths():
    """Guards the guard: a scan that matches nothing passes the test above for free."""
    found = [_layout.rel(p) for p in _layout.nautilus_sources()
             if any(isinstance(n, ast.AsyncFunctionDef) and n.name == "_session_coro"
                    for n in ast.walk(ast.parse(p.read_text())))]
    assert len(found) >= 5, found
