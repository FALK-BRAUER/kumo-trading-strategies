"""`attempts_for` decides a `client_order_id`, so a wrong count is a wrong order (kumo-trading-platform issue 383).

    TOO LOW   the retry regenerates an id already used. Nautilus denies it LOCALLY before it reaches
              the venue and journals identically to a real rejection — and `submit()` reads the order
              back BY that id, so it returns the EARLIER order and reports a stale answer as this
              one's.
    TOO HIGH  a fresh id for an order that may still be live at the venue: a SECOND POSITION.

Imported by kumo-trading-platform for QC345 rather than copied. Two copies of this predicate would not
disagree about a dashboard number; they would disagree about an order id.
"""

from __future__ import annotations

from kumo_strategies.runtime.executor.retry import attempts_for


def _row(sym, session="S1", phase="terminal", ok=False):
    return {"symbol": sym, "session": session, "detail": {"phase": phase, "ok": ok}}


def test_it_counts_TERMINAL_failures_only():
    """A `phase="result"` row is submit-time: `ok=True` there means Nautilus accepted the order into
    its local lifecycle, not that the venue did anything. Counting result rows as attempts is what
    broke resume once — the first live run refused 8 entries for missing instrument definitions, and
    after the fix resume still said "already decided" because those failures looked like attempts."""
    rows = [_row("AAA"), _row("AAA", phase="result"), _row("AAA", phase="intent")]
    assert attempts_for(rows, "S1", ["AAA"]) == {"AAA": 1}


def test_a_terminal_SUCCESS_is_not_an_attempt():
    """A fill is the end, not a try. Counting it would hand the next order a fresh id for a position
    that is already open."""
    assert attempts_for([_row("AAA", ok=True)], "S1", ["AAA"]) == {}


def test_another_SESSION_does_not_leak_in():
    """`client_order_id` is keyed on (strategy, session, symbol, side, slot, attempt). Yesterday's
    failures raising today's attempt would produce an id nothing has used — safe by luck, and wrong
    the moment a resume replays the earlier session."""
    assert attempts_for([_row("AAA", session="S0")], "S1", ["AAA"]) == {}


def test_a_symbol_not_being_traded_is_ignored():
    assert attempts_for([_row("ZZZ")], "S1", ["AAA"]) == {}


def test_repeated_failures_ACCUMULATE_because_each_one_needs_a_new_id():
    assert attempts_for([_row("AAA"), _row("AAA"), _row("AAA")], "S1", ["AAA"]) == {"AAA": 3}


def test_a_missing_or_empty_detail_does_not_raise():
    """Journal rows come from a database and a `detail` can be NULL. A raise here would take down the
    submit loop for a bookkeeping shape, not a trading one."""
    assert attempts_for([{"symbol": "AAA", "session": "S1"},
                         {"symbol": "AAA", "session": "S1", "detail": None},
                         _row("AAA")], "S1", ["AAA"]) == {"AAA": 1}


def test_it_is_SYNCHRONOUS_and_takes_rows_not_a_journal():
    """kumo-trading-platform calls it against whatever row source it already has, and this file needs no
    database. If it ever becomes a coroutine, every caller that forgot `await` gets a truthy object
    and silently counts nothing — the exact asymmetry that cost cockpit a live-looking bug tonight."""
    import inspect

    assert not inspect.iscoroutinefunction(attempts_for)
