"""An attempt belongs to an ORDER, not to a (session, symbol) pair (#164, platform issue 551 blocker).

MEASURED ON THE PAPER JOURNAL, 2026-09-11, by the cockpit session:

    MOMENTUM-002  2026-08-31  AEM   1 fill + 5 budget rejects
    MOMENTUM-002  2026-08-31  VCTR  1 + 5     NOW 1 + 3     HALO 2 + 2
    BCTROT-004    2026-09-02  APA   1 + 1 (budget)
    TECHIVOL-005  2026-09-01  TOST  1 + 1 (connection timeout)

08-31 ran five slots. AEM's order FILLED at the first; the same symbol's later submissions were
refused five times, each writing an `ok=False` terminal row. `attempts_for` keys by
(session, symbol), so it reports **5 attempts for a symbol whose order filled**.

WHY THAT IS NOT MERELY A WRONG NUMBER. `attempt` feeds `client_order_id`
(`OrderRequest.client_order_id` appends `:{attempt}` when nonzero) and `_resume` passes it —
pgrunner:1297 for exits, :1496 for entries, both present on the currently pinned 4d28488. So a
nonzero count mints a FRESH id, which Nautilus does not deny as a duplicate.

PRECISELY, because the first version of this note overstated it: the local duplicate guard IS
engaged whenever the count has not moved — attempt=0 reproduces the original id byte-for-byte
(measured: kumo-2f80de5fb8d9a2d3965d at 0 against kumo-325c30e8388d576da269 at 5), and paper's
Nautilus cache is Redis-backed with `flush_on_start=False`, so that denial survives a restart. The
guard disengages exactly when the count moves, which is normally when a retry is legitimate.

SO THE FAILING SCENARIO IS NARROWER AND SHARPER THAN "the counter is wrong": a fresh id is minted
only if a NEW terminal ok=False row lands BETWEEN the fill and the resume — and because terminal
rows carry no order id, A LATE REJECT FROM ANY OLDER ORDER ON THE SAME SYMBOL counts as that row.
That is the keying gap made concrete, and it is what the tests below drive.

`held_qty` is the second guard and it is the one that has been holding: it is built from the ACCOUNT
position intersected with the claims ledger, not from the journal, so a filled entry is normally in
it and excluded from the replay. The doubling therefore needs TWO faults — this one, plus a claim
that does not reflect the fill. That second fault is not hypothetical here (#829: a 260-share LAND
claim for a position that never existed; #830: a stale VCTR 16), and `sync_claim` is best-effort
from a live event handler.

This file removes the first fault. THREE STATES WHERE THE COUNTER HAS TWO: absent /
refused-nothing-booked / refused-shares-booked — and "filled" is not a state it can represent at
all. Fourth instance of that argument in this package after `MarketState`, `Verdict` and
`Assessment`.
"""

from __future__ import annotations

import pytest

from kumo_strategies.runtime.executor.retry import attempts_for

SESSION = "2026-08-31"


def _row(symbol, ok, *, coid=None, filled=0, status=None, session=SESSION, phase="terminal",
         side=None):
    """A journal row in the shape `record_terminal` writes."""
    detail = {"phase": phase, "ok": ok}
    if coid is not None:
        detail |= {"client_order_id": coid, "filled_qty": filled, "status": status, "side": side}
    return {"session": session, "symbol": symbol, "detail": detail,
            "correlation": coid}


# -- the fixture's own property, asserted before anything is asserted ABOUT it ------------------------

def test_THE_FIXTURE_ROWS_CARRY_AN_ORDER_ID():
    """Guards the guard. Every test below distinguishes orders; if the fixture cannot tell two
    orders apart, they are all asserting on a single-order world and pass for free."""
    rows = _aem_rows()
    coids = {r["detail"].get("client_order_id") for r in rows}
    assert len(coids) == len(rows), f"rows do not identify distinct orders: {coids}"
    assert None not in coids


def _aem_rows():
    """THE MEASURED CASE: one fill, then five refusals at later slots, one session."""
    return [
        _row("AEM", True, coid="AEM-s1", filled=12, status="filled"),
        *[_row("AEM", False, coid=f"AEM-s{i}", filled=0, status="denied") for i in range(2, 7)],
    ]


# -- the defect ---------------------------------------------------------------------------------------

def test_A_FILLED_ORDER_IS_NOT_FIVE_ATTEMPTS():
    """THE AEM CASE. The counter reported 5 for a symbol that filled."""
    assert attempts_for(_aem_rows(), SESSION, ["AEM"]).get("AEM", 0) == 0, (
        "a symbol whose order FILLED still reads attempts — a resume mints a fresh client order "
        "id for it, which Nautilus will not deny")


@pytest.mark.parametrize("symbol,fills,rejects", [
    ("AEM", 1, 5), ("VCTR", 1, 5), ("NOW", 1, 3), ("HALO", 2, 2),
    ("APA", 1, 1), ("TOST", 1, 1),
])
def test_EVERY_MEASURED_GRAIN_READS_ZERO(symbol, fills, rejects):
    """All six instances the paper journal actually contains, not just the worst one."""
    rows = ([_row(symbol, True, coid=f"{symbol}-f{i}", filled=10, status="filled")
             for i in range(fills)]
            + [_row(symbol, False, coid=f"{symbol}-r{i}", filled=0, status="denied")
               for i in range(rejects)])
    assert attempts_for(rows, SESSION, [symbol]).get(symbol, 0) == 0


def test_REFUSALS_WITHOUT_A_FILL_STILL_COUNT():
    """The counter must keep doing its job. This is the case it was written for — #51's VCTR SELL,
    accepted at submit and rejected by the venue minutes later, retried because of this count."""
    rows = [_row("VCTR", False, coid=f"VCTR-{i}", filled=0, status="rejected") for i in range(3)]
    assert attempts_for(rows, SESSION, ["VCTR"])["VCTR"] == 3


def test_THE_SAME_ORDER_REPORTING_TWICE_IS_ONE_ATTEMPT():
    """Terminal handlers can fire more than once for one order — a reject followed by a cancel of
    the same id. Counting rows rather than ORDERS would inflate the attempt and mint a fresh id."""
    rows = [_row("NOW", False, coid="NOW-1", filled=0, status="rejected"),
            _row("NOW", False, coid="NOW-1", filled=0, status="canceled")]
    assert attempts_for(rows, SESSION, ["NOW"])["NOW"] == 1


# -- the hybrid: shares booked, then refused ----------------------------------------------------------

def test_A_PARTIAL_FILL_THEN_CANCEL_IS_NOT_AN_ATTEMPT():
    """THE THIRD STATE. `ok=False` today means both "refused, nothing done" and "refused, 17 shares
    booked", and a resume that treats the second as an attempt re-sends the WHOLE order.

    Not yet observed on paper — reachable by construction: Alpaca cancels partially filled orders,
    and cockpit's own exit path cancels resting ones.
    """
    rows = [_row("APA", False, coid="APA-1", filled=17, status="partially_filled")]
    assert attempts_for(rows, SESSION, ["APA"]).get("APA", 0) == 0, (
        "a cancel that booked 17 shares counted as a failed attempt — the replay would re-send "
        "the full quantity on top of the shares already held")


def test_ONLY_ZERO_FILLED_FAILURES_ARE_ATTEMPTS():
    """The rule stated directly, so the two cases above cannot both be satisfied by an accident."""
    rows = [_row("X", False, coid="X-1", filled=0, status="denied"),
            _row("X", False, coid="X-2", filled=5, status="partially_filled")]
    assert attempts_for(rows, SESSION, ["X"])["X"] == 1


# -- what must not change -----------------------------------------------------------------------------

def test_ROWS_WITHOUT_AN_ORDER_ID_STILL_COUNT_BY_SYMBOL():
    """BACKWARD COMPATIBILITY, and it is a safety property rather than politeness. Every terminal
    row already in the paper journal predates this change and carries no `client_order_id`. If
    those stopped counting, a resume would read 0 attempts for a symbol that genuinely failed
    three times and retry it past its cap.

    Degrading to the old per-symbol behaviour for rows that cannot do better is the conservative
    direction: it over-counts rather than under-counts, and over-counting only ever refuses a
    retry.
    """
    rows = [_row("OLD", False) for _ in range(3)]           # no coid, as written before #164
    assert attempts_for(rows, SESSION, ["OLD"])["OLD"] == 3


def test_A_LEGACY_FILL_ROW_STILL_SUPPRESSES_LEGACY_FAILURES():
    """The AEM case as it appears in rows that PREDATE the fix — one ok=True terminal row and five
    ok=False, none carrying an id. The fill is still visible, so the suppression still applies."""
    rows = [_row("AEM", True)] + [_row("AEM", False) for _ in range(5)]
    assert attempts_for(rows, SESSION, ["AEM"]).get("AEM", 0) == 0


def test_ANOTHER_SESSION_IS_NOT_COUNTED():
    rows = [_row("AEM", False, coid="AEM-1", session="2026-08-28")]
    assert attempts_for(rows, SESSION, ["AEM"]) == {}


def test_A_SYMBOL_NOT_ASKED_ABOUT_IS_NOT_RETURNED():
    rows = [_row("ZZZ", False, coid="ZZZ-1")]
    assert attempts_for(rows, SESSION, ["AEM"]) == {}


def test_NON_TERMINAL_ROWS_ARE_IGNORED():
    """`phase="result"` is submit-time — Nautilus accepting into its local lifecycle, not the venue
    answering. Counting those broke resume once: the first live run refused all 8 entries for
    missing instrument definitions and resume then reported "already decided"."""
    rows = [_row("AEM", False, coid="AEM-1", phase="result") for _ in range(4)]
    assert attempts_for(rows, SESSION, ["AEM"]) == {}


# -- the guard that actually stopped 08-31, pinned before anything can erode it -----------------------

from kumo_strategies.strategies.momentum_rotation.runner import _MAX_SUBMIT_ATTEMPTS  # noqa: E402
from kumo_strategies.runtime.executor.retry import already_attempted        # noqa: E402


def _result(symbol, ok, session=SESSION):
    return {"session": session, "symbol": symbol, "detail": {"phase": "result", "ok": ok}}


def test_A_CONFIRMED_FILL_KEEPS_THE_SYMBOL_OUT_OF_THE_REPLAY_LIST():
    """THE GUARD THAT ACTUALLY HELD ON 08-31, ahead of `held_qty` and well ahead of the budget gate.

    Raised by the cockpit session while reviewing #164, and the warning is the reason this test
    exists at all: `attempts_for` being SYMBOL-grain is the defect, and `attempted` being
    SYMBOL-grain is the fix. Converting both to per-order in one pass would have repaired the
    counter and deleted the guard covering for it — and no test exercised one-fill-plus-N-rejects
    through this path, so the suite would very plausibly have stayed green.

    The AEM shape exactly: one terminal fill, five terminal rejects, one symbol, one session.
    """
    rows = ([_result("AEM", True)]
            + [_row("AEM", True, coid="AEM-s1", filled=12, status="filled")]
            + [_row("AEM", False, coid=f"AEM-s{i}", filled=0, status="denied") for i in range(2, 7)])
    attempted = already_attempted(rows, SESSION, max_attempts=_MAX_SUBMIT_ATTEMPTS)
    assert "AEM" in attempted, (
        "a symbol with a CONFIRMED FILL is back in the replay list — the fill must win over any "
        "number of later rejects on the same symbol, or a resume re-sends a filled entry")


def test_THE_FILL_WINS_EVEN_WHEN_THE_REJECTS_EXHAUST_THE_BOUND():
    """Order of evaluation, asserted rather than assumed: `terminal_fill` is tested BEFORE
    `terminal_reject`. Swapping them would route the AEM shape down the bounded-retry branch."""
    rows = ([_result("AEM", False)] * _MAX_SUBMIT_ATTEMPTS
            + [_row("AEM", True, coid="AEM-s1", filled=12, status="filled")]
            + [_row("AEM", False, coid="AEM-s2", filled=0, status="denied")])
    assert "AEM" in already_attempted(rows, SESSION, max_attempts=_MAX_SUBMIT_ATTEMPTS)


def test_A_REJECTED_SYMBOL_WITH_ROOM_LEFT_IS_STILL_RETRIED():
    """The guard must not become "never replay anything". One rejection inside the bound stays
    retryable — that is #51's VCTR case, the reason terminal rows are read at all."""
    rows = [_result("VCTR", True), _row("VCTR", False, coid="VCTR-1", filled=0, status="rejected")]
    assert "VCTR" not in already_attempted(rows, SESSION, max_attempts=_MAX_SUBMIT_ATTEMPTS)


def test_A_SUPPRESSED_ENTRY_STAYS_TERMINAL_FOR_THE_SESSION():
    """Carried through the extraction unchanged: retrying a suppressed entry would defeat the
    guard, and after a mid-session config fix would submit against a decision taken under the old
    configuration."""
    rows = [{"session": SESSION, "symbol": "SUP", "detail": {"suppressed": True}}]
    assert "SUP" in already_attempted(rows, SESSION, max_attempts=_MAX_SUBMIT_ATTEMPTS)


# -- the behaviour change this fix causes, said out loud ----------------------------------------------

def test_THE_FIX_MAKES_A_LOCAL_DUPLICATE_DENIAL_ROUTINE_AND_THAT_IS_CORRECT():
    """A BEHAVIOUR CHANGE IN THE DENIAL PATH, stated in a test so the next person reading a
    "duplicate denied" line knows it is the design and not an incident.

    A symbol that reads attempt=5 today reads 0 after this fix. On a resume it therefore mints the
    attempt=0 id — which MAY ALREADY EXIST from the original submit, and Nautilus will then deny it
    LOCALLY as a duplicate. That denial is the correct outcome: it is the idempotency guard doing
    exactly its job, and it is strictly better than the old behaviour, which minted a fresh id and
    reached the venue.

    Asserted on the id derivation itself rather than described, because the whole point of the
    change is which id gets minted.
    """
    from kumo_strategies.runtime.executor.broker import OrderRequest

    def coid(attempt):
        return OrderRequest("AEM", "BUY", 12, SESSION, strategy_id="MOMENTUM-002",
                            slot="open+5m", attempt=attempt).client_order_id

    assert coid(5) != coid(0), "attempt does not reach the id — the premise of #164 is wrong"
    assert coid(0) == coid(0)
    # and attempt=0 is byte-identical to omitting it, so no id this repo has ever minted moves
    same = OrderRequest("AEM", "BUY", 12, SESSION, strategy_id="MOMENTUM-002", slot="open+5m")
    assert same.client_order_id == coid(0), (
        "the zero case is not byte-identical — every historical id would shift, and a replay would "
        "no longer collide with the order it is replaying")



# -- side-awareness: LIVE ON MOMENTUM TODAY, not a QC345-era concern ----------------------------------

def test_A_REJECTED_SELL_DOES_NOT_ARM_THE_BUY_FOR_THE_SAME_NAME():
    """MEASURED ON MOMENTUM'S OWN SCHEDULE, which is why this is not a port-era concern.

    AEM, 2026-08-31: SELL 9 at open+5m, then BUY 9 decided at open+90m, +130m, +258m, +322m and
    +335m — same session, same symbol, OPPOSITE SIDE. It did not bite only because the SELL FILLED
    (terminal ok=true, no increment). A REJECTED exit followed by a re-entry of the same name in
    the same session hands the entry a fresh client order id off the exit's failure count.

    I first scoped side-awareness as latent until QC345 arrived with the port. That was wrong: the
    cockpit session measured the exit-then-re-enter shape on the lane running today. The mutation
    that removes the side filter killed nothing until this test existed.
    """
    rows = [_row("AEM", False, coid="AEM-sell-1", filled=0, status="rejected", side="SELL")]
    assert attempts_for(rows, SESSION, ["AEM"], side="SELL")["AEM"] == 1, "the exit's own retry"
    assert attempts_for(rows, SESSION, ["AEM"], side="BUY").get("AEM", 0) == 0, (
        "a rejected SELL armed the BUY for the same name — the entry gets a fresh client order id "
        "off the exit's failure count, and Nautilus will not deny it")


def test_EACH_SIDE_KEEPS_ITS_OWN_COUNT():
    rows = [_row("X", False, coid="X-s1", side="SELL"), _row("X", False, coid="X-s2", side="SELL"),
            _row("X", False, coid="X-b1", side="BUY")]
    assert attempts_for(rows, SESSION, ["X"], side="SELL")["X"] == 2
    assert attempts_for(rows, SESSION, ["X"], side="BUY")["X"] == 1


def test_NO_SIDE_FILTER_COUNTS_EVERYTHING():
    """`side=None` is the pre-#164 behaviour, kept so an existing caller is unchanged."""
    rows = [_row("X", False, coid="X-s1", side="SELL"), _row("X", False, coid="X-b1", side="BUY")]
    assert attempts_for(rows, SESSION, ["X"])["X"] == 2


def test_A_ROW_THAT_RECORDS_NO_SIDE_COUNTS_UNDER_EITHER_FILTER():
    """Same backward-compatibility rule as the missing order id: every row already in the paper
    journal predates this and records no side. Counting it under both sides OVER-counts, and
    over-counting only ever refuses a retry — the safe direction."""
    rows = [_row("OLD", False)]
    assert attempts_for(rows, SESSION, ["OLD"], side="BUY")["OLD"] == 1
    assert attempts_for(rows, SESSION, ["OLD"], side="SELL")["OLD"] == 1


def test_A_FILLED_SELL_DOES_NOT_SUPPRESS_THE_BUY():
    """THE OTHER HALF, and the trap in the suppression rule: AEM's SELL filled on 08-31 and the
    lane then legitimately re-entered. If a filled exit suppressed the entry's count, a genuinely
    failing re-entry would never retry."""
    rows = [_row("AEM", True, coid="AEM-sell-1", filled=9, status="filled", side="SELL"),
            _row("AEM", False, coid="AEM-buy-1", filled=0, status="denied", side="BUY")]
    assert attempts_for(rows, SESSION, ["AEM"], side="BUY")["AEM"] == 1, (
        "a FILLED SELL suppressed the BUY's attempt count — the re-entry would never be retried")
    assert attempts_for(rows, SESSION, ["AEM"], side="SELL").get("AEM", 0) == 0
