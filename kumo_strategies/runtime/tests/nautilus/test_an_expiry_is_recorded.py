"""A DAY order that expires at the close must write a terminal row (#165).

MEASURED ON THE PAPER PIN by the cockpit session: terminal rows come from FOUR handlers —
`on_order_filled`, `on_order_denied`, `on_order_rejected`, `on_order_canceled`. There is no
`on_order_expired`. Every lane sends MARKET DAY orders, and a DAY order not fully filled by the
close EXPIRES at the venue.

WORSE THAN A WRONG ROW, WHICH IS WHY IT IS ITS OWN TICKET rather than part of #164. The hybrid case
at least leaves an `ok=False` row to be wrong about. An expiry leaves the journal reading "still in
flight" FOREVER: `attempts_for` sees no failure, `_resume` sees no terminal outcome and treats the
symbol as accepted-but-unanswered (deliberately left alone, never retried), and cockpit's readback
— which expects exactly one terminal row per order — cannot see the order end at all.

FOUR STATES, THE FOURTH SILENT: filled / refused / cancelled / EXPIRED. The counter's three-state
problem in #164 was two states where there were three; this is three where there are four, in the
writer rather than the reader.

Not yet observed: the cockpit cache holds 35 lane orders, 34 FILLED and 1 REJECTED, none EXPIRED.
Reachable by construction — DAY orders at the close, a partial fill late in the session.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from types import SimpleNamespace

import pytest

from kumo_strategies.strategies import _layout



def _handlers(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("on_order_")}


def _adapters_that_record_terminals() -> list[pathlib.Path]:
    """DISCOVERED, NOT LISTED — #138's rule, which exists because a hand-written lane list silently
    dropped BCTROT. Any module that handles a terminal order event is in scope."""
    out = []
    for path in _layout.nautilus_sources():                  # ks#211: shared layer + every lane
        h = _handlers(path)
        if "on_order_filled" in h:
            out.append(path)
    return out


LANES = _adapters_that_record_terminals()


def test_THE_SCAN_FOUND_THE_ADAPTERS():
    """Guards the guard: an empty scan makes every check below vacuous."""
    assert len(LANES) >= 3, [_layout.rel(p) for p in LANES]


@pytest.mark.parametrize("path", LANES, ids=_layout.rel)
def test_EVERY_ADAPTER_THAT_HANDLES_A_CANCEL_ALSO_HANDLES_AN_EXPIRY(path):
    """An expiry is a terminal venue answer exactly as a cancel is, and both leave the lane holding
    less than it asked for. An adapter that records one and not the other has a silent fourth
    state.

    Keyed on `on_order_canceled` rather than asserted unconditionally — but the exemption is
    VERIFIED, not asserted, because "deliberately incomplete" was my own wording for three adapters
    and only one of them had earned it:

      crsi_short        ORDER_PATH_COMPLETE = False, refuses to construct unless shadow_only. Real.
      penny_gap         handles ONLY on_order_filled — no reject, deny, cancel or expire. Its
                        terminal coverage is far worse than this ticket, and it is the PARKED #40
                        binding rather than a live lane. Noted on #165, not fixed here.
      template_rotation was missing cancel AND expiry — and it is the file new adapters are COPIED
                        from, so a gap there is a gap generator. Fixed, which is why it no longer
                        skips.
    """
    h = _handlers(path)
    if "on_order_canceled" not in h:
        pytest.skip(f"{_layout.rel(path)}: no cancel handler either — see the docstring, the exemption is "
                    f"per-adapter and verified")
    assert "on_order_expired" in h, (
        f"{_layout.rel(path)} records a cancel but not an EXPIRY. Every lane sends MARKET DAY orders and a "
        f"DAY order unfilled at the close expires — the journal then reads 'still in flight' "
        f"forever and nothing ever retries it.")


@pytest.mark.parametrize("path", LANES, ids=_layout.rel)
def test_THE_EXPIRY_HANDLER_RECORDS_A_TERMINAL_ROW(path):
    """Existence is not enough — a handler that only forgets the intent would pass the check above
    while still writing nothing. Bound to the AST of the handler itself."""
    h = _handlers(path)
    if "on_order_expired" not in h:
        pytest.skip(f"{_layout.rel(path)} has no expiry handler (covered by the check above)")
    if "_record_terminal" not in path.read_text():
        # VERIFIED EXEMPTION, not a convenience: `template_rotation` records NO terminal rows for
        # ANY event — it has no `_record_terminal` and no runner to hand one to. That is a wider
        # gap than this ticket and belongs on its own; what #165 asks of the template is that it
        # HANDLE the event, which the check above enforces, so a copied adapter starts with the
        # handler present rather than absent.
        pytest.skip(f"{_layout.rel(path)} records no terminal rows for any event — wider than #165")
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "on_order_expired")
    called = {(c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", None))
              for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert "_record_terminal" in called, (
        f"{_layout.rel(path)}.on_order_expired does not record a terminal row — the handler exists and the "
        f"state is still silent")


@pytest.mark.parametrize("path", LANES, ids=_layout.rel)
def test_AN_EXPIRY_IS_NOT_RECORDED_AS_A_SUCCESS(path):
    """`ok=True` would make an expiry suppress the symbol's attempt count in `attempts_for` exactly
    as a fill does — a name that never filled would look done for the session."""
    h = _handlers(path)
    if "on_order_expired" not in h:
        pytest.skip("no expiry handler")
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "on_order_expired")
    for c in ast.walk(fn):
        if isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "_record_terminal":
            ok = next((k.value for k in c.keywords if k.arg == "ok"), None)
            assert isinstance(ok, ast.Constant) and ok.value is False, (
                f"{_layout.rel(path)} records an expiry with ok={ast.unparse(ok) if ok else 'missing'}")


@pytest.mark.parametrize("path", LANES, ids=_layout.rel)
def test_A_FOREIGN_EXPIRY_IS_IGNORED(path):
    """Same rule as every sibling handler. Cockpit's protective stops are foreign orders on the
    same instruments; recording their expiry as ours would attribute another owner's outcome to
    this lane and `_forget` would clear an in-flight marker for an order still working."""
    h = _handlers(path)
    if "on_order_expired" not in h:
        pytest.skip("no expiry handler")
    src = path.read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "on_order_expired")
    names = {getattr(c.func, "id", None) for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert "is_foreign" in names, f"{_layout.rel(path)}.on_order_expired does not check is_foreign"


# -- behaviour, through the real handler ---------------------------------------------------------------

def _driven(coid="kumo-abc"):
    """Drive the REAL handler with the narrowest host that can answer it.

    `is_foreign` reads the CLIENT ORDER ID PREFIX, not a strategy id — so a lane identity is not
    needed and cannot be set anyway (`Component.id` is read-only). Binding the real unbound method
    to a stub rather than reimplementing it: a reimplementation tests the copy.
    """
    from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy

    rows, forgotten = [], []

    class _Host:
        def _forget(self, event):
            forgotten.append(event)

        def _record_terminal(self, event, *, ok, detail):
            rows.append((ok, detail))

    ev = SimpleNamespace(
        client_order_id=coid,
        instrument_id=SimpleNamespace(symbol=SimpleNamespace(value="AEM")))
    MomentumRotationStrategy.on_order_expired(_Host(), ev)
    return rows, forgotten


def test_AN_EXPIRY_WRITES_EXACTLY_ONE_TERMINAL_ROW():
    """The AST checks say the shape is right; this says it does the thing. One row, ok=False,
    naming the expiry — not zero, and not two."""
    rows, forgotten = _driven()
    assert len(rows) == 1, f"expected exactly one terminal row, got {rows}"
    ok, detail = rows[0]
    assert ok is False
    assert "expir" in detail.lower(), detail
    assert len(forgotten) == 1, "the pending intent was not cleared — the lane stays blocked"


def test_A_FOREIGN_EXPIRY_WRITES_NOTHING_AND_CLEARS_NOTHING():
    """Cockpit's protective stops are foreign orders on the same instruments, cancelled and re-armed
    on a 60s reconciler tick. Recording their expiry would attribute another owner's outcome to this
    lane, and `_forget` — which drops the pending intent BY SYMBOL — would clear the in-flight
    marker for an order still working."""
    rows, forgotten = _driven(coid="PROT-SELL-AEM-XNAS-3c182de6")
    assert rows == [], f"a foreign expiry was recorded as ours: {rows}"
    assert forgotten == [], "a foreign expiry cleared this lane's pending intent"


@pytest.mark.parametrize("path", LANES, ids=_layout.rel)
def test_THE_DETAIL_SAYS_EXPIRED_RATHER_THAN_CANCELED(path):
    """An operator reading the journal must tell a venue EXPIRY from a cancel someone issued — they
    have different causes and different responses.

    BOUND TO THE AST, not to the source text. The first version of this asserted
    `"expired" in inspect.getsource(...)`, which this repo has a standing guard against and which
    duly failed it: a substring check on source is satisfied by writing the word in a COMMENT and
    broken by deleting one. The string that reaches the journal is the `detail=` argument, so that
    is what is read.
    """
    h = _handlers(path)
    if "on_order_expired" not in h or "_record_terminal" not in path.read_text():
        pytest.skip("no expiry handler that records")
    fn = next(n for n in ast.walk(ast.parse(path.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "on_order_expired")
    details = [k.value.value for c in ast.walk(fn) if isinstance(c, ast.Call)
               for k in c.keywords
               if k.arg == "detail" and isinstance(k.value, ast.Constant)]
    assert details, "the expiry records no literal detail — nothing names the state in the journal"
    assert any("expir" in d.lower() for d in details), details
    assert not any("cancel" in d.lower() for d in details), (
        f"an expiry is journalled as a cancel: {details}")
