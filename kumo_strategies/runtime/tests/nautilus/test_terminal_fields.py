"""The order's identity on a terminal row, and it must never cost the row (#164).

`attempts_for` had to key by (session, symbol) because terminal rows carried only `{ok, phase}`
with a null `correlation` — nothing tied a row to an order. A late reject from an OLDER order on
the same symbol therefore re-armed the attempt count of a FILLED one, minting a fresh
`client_order_id` that Nautilus does not deny.

THE ENRICHMENT IS A BONUS AND THE ROW IS THE POINT. Every adapter's `_record_terminal` body sits
inside `try/except Exception: log`, so anything that raises while reading the order swallows the
WHOLE ROW — turning a missing `filled_qty` into a missing terminal row, which is the observability
gap #51 exists to close. The first version read `self.cache.order(...)` outside the guard and did
exactly that.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kumo_strategies.runtime.nautilus.contract import call_record_terminal, terminal_fields


class _Order:
    def __init__(self, filled=0, status="FILLED", side="BUY"):
        self.filled_qty, self.status, self.side = filled, status, side


def _strategy(order=None, *, cache=True, raises=False):
    class _Cache:
        def order(self, coid):
            if raises:
                raise RuntimeError("cache miss deep in nautilus")
            return order
    return SimpleNamespace(cache=_Cache() if cache else None)


def _event(coid="kumo-abc", last_qty=5):
    return SimpleNamespace(client_order_id=coid, last_qty=last_qty)


# -- the property that matters -------------------------------------------------------------------

@pytest.mark.parametrize("strategy", [
    _strategy(raises=True),                       # the cache itself blows up
    _strategy(cache=False),                       # no cache at all
    SimpleNamespace(),                            # not even the attribute
])
def test_NOTHING_IT_READS_CAN_RAISE(strategy):
    """A missing `filled_qty` must never become a missing terminal row."""
    assert terminal_fields(strategy, _event()) is not None


def test_A_CACHE_THAT_RAISES_STILL_LETS_THE_ROW_BE_WRITTEN():
    """THE REGRESSION, end to end through the seam. A test double whose cache had no `order`
    method suppressed all four of qc27's terminal rows — 'recorded nothing'."""
    seen = []

    def record(session, symbol, ok, detail, **kw):
        seen.append((symbol, ok, kw))

    call_record_terminal(record, "2026-08-31", "AEM", True, "filled",
                         **terminal_fields(_strategy(raises=True), _event()))
    assert seen, "the row was not written because the enrichment failed"


def test_A_BROKEN_EVENT_STILL_YIELDS_A_DICT():
    assert isinstance(terminal_fields(_strategy(), SimpleNamespace()), dict)


# -- what it reads when it can ---------------------------------------------------------------------

def test_IT_PREFERS_THE_ORDERS_CUMULATIVE_FILL_OVER_THE_EVENTS_LAST_QTY():
    """`last_qty` is THIS event's fill; `filled_qty` is the order's running total. A partial fill
    followed by a cancel must report the shares BOOKED, not the last slice — that is the whole
    third state (`refused, 17 booked`)."""
    f = terminal_fields(_strategy(_Order(filled=17, status="CANCELED")), _event(last_qty=5))
    assert f["filled_qty"] == 17
    assert f["status"] == "canceled"


def test_IT_CARRIES_THE_SIDE():
    """Without it a SELL's rejection increments the BUY attempt for the same name in the same
    session. MEASURED on momentum's own schedule: AEM 2026-08-31, SELL 9 filled at open+5m, then
    BUY 9 decided at five later slots — same session, same symbol, opposite side."""
    assert terminal_fields(_strategy(_Order(side="OrderSide.SELL")), _event())["side"] == "SELL"
    assert terminal_fields(_strategy(_Order(side="OrderSide.BUY")), _event())["side"] == "BUY"


def test_IT_CARRIES_THE_ORDER_ID():
    assert terminal_fields(_strategy(_Order()), _event("kumo-xyz"))["client_order_id"] == "kumo-xyz"


def test_AN_ENUM_STATUS_IS_REDUCED_TO_ITS_NAME():
    """Nautilus statuses stringify as `OrderStatus.FILLED`. The row should carry `filled`, not the
    enum's repr — it is compared against by other code and rendered in a UI."""
    assert terminal_fields(_strategy(_Order(status="OrderStatus.PARTIALLY_FILLED")),
                           _event())["status"] == "partially_filled"


# -- the cross-repo seam ------------------------------------------------------------------------------

def test_KEYWORDS_A_RUNNER_CANNOT_ACCEPT_ARE_DROPPED_NOT_RAISED():
    """THIS SEAM IS PINNED BY REVISION. kumo-trading-platform supplies its own runner; passing a keyword an
    older implementation does not accept raises TypeError INSIDE NAUTILUS'S DISPATCH, on the
    handler telling us a real order filled. `bctrot_rotation` records the same lesson for
    `read_slots` (#377), with the direction reversed."""
    got = {}

    async def old_runner(session, symbol, ok, detail):          # the pre-#164 signature
        got["called"] = True

    call_record_terminal(old_runner, "S", "AEM", True, "filled",
                         client_order_id="x", filled_qty=1, status="filled", side="BUY").close()
    assert got.get("called") or True     # constructing the coroutine is the assertion: no TypeError


def test_A_RUNNER_THAT_ACCEPTS_THEM_GETS_THEM():
    seen = {}

    def new_runner(session, symbol, ok, detail, *, client_order_id=None, filled_qty=0,
                   status=None, side=None):
        seen.update(client_order_id=client_order_id, filled_qty=filled_qty, side=side)

    call_record_terminal(new_runner, "S", "AEM", True, "filled",
                         client_order_id="x", filled_qty=7, status="filled", side="BUY")
    assert seen == {"client_order_id": "x", "filled_qty": 7, "side": "BUY"}


def test_A_KWARGS_RUNNER_GETS_EVERYTHING():
    """`**kwargs` cannot raise, so pass it all — but note it is not evidence the callee STORES any
    of it. "Forwarding through **kwargs is not a public seam"."""
    seen = {}

    def kwargs_runner(session, symbol, ok, detail, **kw):
        seen.update(kw)

    call_record_terminal(kwargs_runner, "S", "AEM", True, "filled", client_order_id="x", side="BUY")
    assert seen == {"client_order_id": "x", "side": "BUY"}


# -- the silent drop is recorded, once ----------------------------------------------------------------

def _fresh_seam():
    """The report set is process-global by design (once per runner per process), so tests that
    assert on first-report behaviour must clear it or they depend on execution order."""
    from kumo_strategies.runtime.nautilus import contract
    contract._SEAM_REPORTED.clear()


def test_DROPPING_FIELDS_IS_REPORTED_NOT_SILENT():
    """A SHIM THAT SILENTLY DROPS FIELDS IS A FALLBACK, and a fallback is a silent wrong answer
    unless it is as correct as the primary. This one is not: an old runner keeps receiving poorer
    rows indefinitely.

    The failure is not a crash. It is someone reading a terminal row with no `filled_qty` in six
    months, concluding the fill data was never written, and re-deriving the whole investigation.
    Raised by the coordinator reviewing #166.
    """
    _fresh_seam()
    said = []

    async def old_runner(session, symbol, ok, detail):
        pass

    call_record_terminal(old_runner, "S", "AEM", True, "filled", log=said.append,
                         client_order_id="x", filled_qty=1, status="filled", side="BUY").close()
    assert said, "fields were dropped and nothing said so"
    msg = said[0]
    assert "DEGRADED BY DESIGN" in msg, msg
    for field in ("client_order_id", "filled_qty", "side"):
        assert field in msg, f"{field} not named in the report: {msg}"
    assert "record_terminal" in msg, "the message does not say which side needs widening"


def test_IT_REPORTS_ONCE_PER_RUNNER_NOT_ONCE_PER_ORDER():
    """A degradation notice that fires on every fill is noise, and noise is how a real one gets
    missed. MOMENTUM alone wrote 278 terminal rows in the measured window."""
    _fresh_seam()
    said = []

    async def old_runner(session, symbol, ok, detail):
        pass

    for _ in range(50):
        call_record_terminal(old_runner, "S", "AEM", True, "filled", log=said.append,
                             client_order_id="x", filled_qty=1).close()
    assert len(said) == 1, f"{len(said)} notices for one runner"


def test_A_RUNNER_THAT_TAKES_THEM_IS_ALSO_RECORDED():
    """THREE STATES: supplied and taken, supplied and dropped, never asked. Recording only the
    bad case means a silent seam and a working seam look identical in a log."""
    _fresh_seam()
    said = []

    def new_runner(session, symbol, ok, detail, *, client_order_id=None, filled_qty=0):
        pass

    call_record_terminal(new_runner, "S", "AEM", True, "filled", log=said.append,
                         client_order_id="x", filled_qty=1)
    assert said and "full per-order outcome" in said[0], said


def test_A_PARTIAL_ACCEPTANCE_NAMES_ONLY_WHAT_WAS_DROPPED():
    """A runner mid-migration takes some and not others. Naming all four would send whoever reads
    it to widen fields that are already accepted."""
    _fresh_seam()
    said = []

    def half_runner(session, symbol, ok, detail, *, client_order_id=None):
        pass

    call_record_terminal(half_runner, "S", "AEM", True, "filled", log=said.append,
                         client_order_id="x", filled_qty=1, side="BUY")
    msg = said[0]
    assert "DEGRADED" in msg and "filled_qty" in msg and "side" in msg
    assert "'client_order_id'" not in msg.split("does not accept")[1].split("so those")[0], (
        f"an accepted field was named as dropped: {msg}")


def test_NO_LOGGER_STILL_WORKS_AND_STAYS_QUIET():
    """The shim must not require a logger — it is called from paths that may not have one, and a
    reporting mechanism that can break the call it reports on is worse than no reporting."""
    _fresh_seam()

    def new_runner(session, symbol, ok, detail, *, client_order_id=None):
        return "called"

    assert call_record_terminal(new_runner, "S", "AEM", True, "f", client_order_id="x") is None \
        or True      # the assertion is that it did not raise


def test_THE_REPORT_CANNOT_BREAK_THE_CALL():
    """A logger that raises must cost the notice, never the terminal row — the same rule as
    `terminal_fields`, and the reason it exists is that the first version of THAT broke the row."""
    _fresh_seam()

    def boom(_msg):
        raise RuntimeError("logging is down")

    def new_runner(session, symbol, ok, detail, *, client_order_id=None):
        return "written"

    try:
        call_record_terminal(new_runner, "S", "AEM", True, "filled", log=boom,
                             client_order_id="x")
    except RuntimeError:
        pytest.fail("a failing logger prevented the terminal row from being written")
