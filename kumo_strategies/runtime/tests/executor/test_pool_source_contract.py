"""The rules `POST /pool/source/{name}` will be held to (kumo-trading-platform issue 486 step 5).

Cockpit owns the endpoint; this repo owns the WRITE, because `PoolSource` is defined here and
`PgSymbolPool.refresh_source` is its only writer. A second writer with REPLACE semantics does not
race — the loser's rows vanish.

Pure, so the rules are testable without Postgres. The write path itself needs a database and is
covered by the `@pg` tests; these are the decisions, which is where the defects live.
"""

from __future__ import annotations

import pytest

from kumo_strategies.runtime.executor.pgpool import (
    ShrinkRejected, UnknownSource, check_shrink, check_source_known)


def test_an_unknown_source_is_REFUSED_by_default():
    """`refresh_source("typo_watchlist", rows)` must not quietly invent a source.

    A typo produces a source that is fresh, real, and feeds nothing — which reads as healthy. Same
    shape as a default identity that is a live lane: well-formed, plausible, and wrong in a way
    nothing downstream can detect.
    """
    with pytest.raises(UnknownSource, match="typo_watchlist"):
        check_source_known("typo_watchlist", known=False, create=False)


def test_creating_a_source_is_ALLOWED_when_asked_for_explicitly():
    """A new source is a deliberate act, not a side effect of a first refresh."""
    check_source_known("ledger_book", known=False, create=True)
    check_source_known("ledger_book", known=True, create=False)


def test_the_DEFAULT_still_creates_because_the_stack_is_live():
    """`create` defaults to True in `refresh_source` and cockpit's endpoint passes False.

    Deliberate and temporary. The scheduled refreshers are running against a deployed stack the night
    before an open; flipping the default would make any source lacking a health row start failing,
    and `source_health()` treats a stale source as a HARD BLOCK on deciding. An unrun refresher does
    not degrade a lane, it stops it deciding at all.

    So the NEW surface gets the strict rule and the running one is untouched. Asserted rather than
    left as a comment, so the day someone tightens the default it is a decision and not a slip.
    """
    import inspect

    from kumo_strategies.runtime.executor.pgpool import PgSymbolPool

    assert inspect.signature(PgSymbolPool.refresh_source).parameters["create"].default is True


def test_a_TRUNCATED_feed_is_refused():
    """The existing guard, and it is stronger than an empty-check: a source returning 3 of 93
    symbols is the dangerous case, and an `allow_empty` flag would not see it at all.

    A pool departure is acted on as the followed trader's SELL SIGNAL — worth 29 points of return —
    so a truncated feed liquidates.
    """
    assert check_shrink("s", prev=40, now=2, max_shrink=0.5, reason=None) is not None
    assert check_shrink("s", prev=40, now=25, max_shrink=0.5, reason=None) is None


def test_an_EMPTY_refresh_of_a_populated_source_is_refused():
    """Replace-semantics means empty DELETES the set. A source that legitimately returns nothing and
    one whose fetch script silently failed are indistinguishable at this boundary."""
    assert check_shrink("s", prev=93, now=0, max_shrink=0.5, reason=None) is not None
    assert check_shrink("s", prev=1, now=0, max_shrink=0.5, reason=None) is not None


def test_a_LEGITIMATE_full_exit_is_possible_but_must_be_STATED():
    """The trap in the guard, which is not in the plan and which I would have shipped without.

    The shrink floor exists so a truncated feed cannot liquidate. It ALSO blocks a followed trader
    genuinely going flat — which is the single most significant thing that source can ever say, and
    the one the pool exists to act on. Refusing it forever means the one event that matters most is
    the one event that cannot happen.

    So the override is a REASON, not a boolean: a loud, recorded override rather than a refusal,
    for the same argument as the deploy guard. A rule with no escape gets bypassed under pressure,
    which removes the record rather than the case.
    """
    assert check_shrink("s", prev=93, now=0, max_shrink=0.5,
                        reason="ledger closed the book, confirmed in the 16:00 statement") is None
    with pytest.raises(ValueError, match="reason"):
        check_shrink("s", prev=93, now=0, max_shrink=0.5, reason="   ")


def test_a_NEW_source_may_start_empty():
    """No previous set means nothing can be lost. Refusing here would make an empty first import a
    failure rather than a fact."""
    assert check_shrink("s", prev=0, now=0, max_shrink=0.5, reason=None) is None
