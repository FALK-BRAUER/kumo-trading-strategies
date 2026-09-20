"""One intent, TWO orders: the opening-auction leg and its intraday replacement (#131).

CRSISHORT's entry is a sell-short LIMIT resting into the OPENING CROSS. **108 of its 231 accepted
trades fill at the opening print and carry 77.6% of the return** (+9.99%/trade against +2.53%), so
where the order rests is not an execution detail — it is most of the strategy.

IB cannot express that as one order. A limit-on-open (`AT_THE_OPEN` -> OPG) participates in the
cross and is CANCELLED by the venue if it does not fill there; an ordinary day limit never
participates in the cross at all. Reproducing the backtest therefore takes two orders for one
intent, and a lane that sent only the day limit would trade a different strategy while every
surface read normal — the shape `ORDER_PATH_COMPLETE = False` has been standing in front of since
the lane was written.

THE RULE IS PURE AND LIVES HERE; THE SUBMITTING LIVES IN COCKPIT. This package never calls a broker.
What it owns is WHICH ORDERS an intent becomes, and that is a rule a test can pin — the alternative
is the decomposition living as prose in a gateway nobody can test against the backtest.

THE SECOND LEG IS CONDITIONAL, and that is the whole hazard. Sent unconditionally, the lane rests
TWO orders for one slot and can fill BOTH — a double position on a short, which is the direction
that is unbounded. It exists only once the venue has told us the first is gone.
"""

from __future__ import annotations

import pytest

from kumo_strategies.runtime.executor.broker import OrderRequest


def _fn(name):
    import kumo_strategies.strategies.crsi_short.nautilus as mod
    fn = getattr(mod, name, None)
    if fn is None:
        pytest.fail(
            f"`{name}` does not exist: the limit-on-open decomposition is designed and not built, "
            f"so CRSISHORT can only rest an ordinary day limit and forgoes the opening cross where "
            f"77.6% of its return is made (#131)")
    return fn


#: `open-10m` is the ruled live slot: IB accepts OPG until ~09:28 ET, so `open-5m` has no
#: margin. Note there is no bare "open" in this repo's grammar — `slots._parse` requires an
#: offset, so the auction itself is spelled `open-0m`.
ARGS = dict(symbol="AAOI", qty=57, limit_px=12.34, session="2026-09-11",
            strategy_id="CRSISHORT-006", slot="open-10m")


# -- leg one: the auction --------------------------------------------------------------------------

def test_the_OPENING_leg_is_a_LIMIT_at_the_open():
    """Not a market-on-open. The limit is the strategy: `apply_gates` rests it 3% above the prior
    close and a market order would fill at whatever the cross prints."""
    leg = _fn("opening_leg")(**ARGS)
    assert isinstance(leg, OrderRequest)
    assert leg.time_in_force == "AT_THE_OPEN", (
        f"the opening leg is {leg.time_in_force!r}: a DAY limit never participates in the cross, "
        f"which is where 108 of 231 accepted trades fill")
    assert leg.limit_px == 12.34
    assert leg.side == "SELL" and leg.qty == 57


def test_the_opening_leg_REFUSES_a_sub_penny_limit():
    """`apply_gates` already rounds to the penny because "a limit at 12.3449 rests at 12.34, not
    12.35" and that changes which fills happen. Rounding again here would be a SECOND derivation of
    one number, and the second is where they disagree — so an unrounded price is refused rather
    than repaired."""
    with pytest.raises(ValueError, match="penny|round|tick"):
        _fn("opening_leg")(**{**ARGS, "limit_px": 12.3449})


def test_the_opening_leg_REFUSES_a_non_positive_quantity():
    """A zero-share order is not an order, and a negative one flips the side. `qty` here is a
    MAGNITUDE — the side says SELL."""
    for bad in (0, -57):
        with pytest.raises(ValueError):
            _fn("opening_leg")(**{**ARGS, "qty": bad})


# -- leg two: the replacement ----------------------------------------------------------------------

def test_the_REPLACEMENT_rests_at_the_SAME_price():
    """Same intent, same limit. A replacement at a different price is a different decision, and the
    backtest models one: `max(limit, open)`, the limit unchanged all session."""
    first = _fn("opening_leg")(**ARGS)
    second = _fn("replacement_leg")(first)
    assert second.limit_px == first.limit_px == 12.34
    assert second.time_in_force == "DAY"
    assert (second.side, second.qty, second.symbol) == (first.side, first.qty, first.symbol)


def test_the_two_legs_have_DIFFERENT_client_order_ids():
    """THE ID IS THE REPLAY GUARD. Nautilus denies a duplicate `client_order_id` LOCALLY, before the
    venue sees it — so if the two legs hashed the same, the replacement would be denied as a
    duplicate and `submit()` would read the FIRST order back out of the cache and report ITS answer
    as the second's. That exact confusion once reported a seven-hour-old rejection as the morning's.
    """
    first = _fn("opening_leg")(**ARGS)
    second = _fn("replacement_leg")(first)
    assert first.client_order_id != second.client_order_id, (
        "both legs hash to one id: the replacement would be denied locally as a duplicate and the "
        "auction leg's answer reported in its place")


def test_a_replacement_is_REFUSED_for_an_order_that_was_not_the_auction_leg():
    """Only an OPG leg has a replacement. Replacing a DAY limit would rest a second order for one
    slot — two live orders on one reservation, which `record_pending` refuses for the same reason."""
    day = OrderRequest(symbol="AAOI", side="SELL", qty=57, session="2026-09-11",
                       strategy_id="CRSISHORT-006", slot="open", limit_px=12.34,
                       time_in_force="DAY")
    with pytest.raises(ValueError, match="AT_THE_OPEN|auction|opening"):
        _fn("replacement_leg")(day)


# -- the hazard the conditionality exists for -------------------------------------------------------

def test_a_replacement_is_ONLY_owed_once_the_auction_leg_is_GONE():
    """THE DOUBLE-FILL. Both legs live at once means one slot resting two orders that can BOTH fill,
    and on a short that exposure is unbounded. The decision is a rule rather than a comment.
    """
    owed = _fn("replacement_is_owed")
    assert owed("EXPIRED") is True, "the auction ended without a fill — this is the ordinary path"
    assert owed("CANCELED") is True, "the venue pulled it; the intent stands"
    assert owed("FILLED") is False, "the entry is ON; a replacement would double the position"
    assert owed("ACCEPTED") is False, "still live in the auction — a second order would double it"
    assert owed("PARTIALLY_FILLED") is False, (
        "a partial fill took the slot. Topping up is a DIFFERENT decision from replacing an "
        "unfilled order and this lane does not make it")
    assert owed("REJECTED") is False, (
        "the venue refused the order. Resting the same intent again as a day limit would convert a "
        "rejection into a silent retry at a different time of day")


def test_an_UNKNOWN_order_status_does_NOT_owe_a_replacement():
    """Three states, and the third is not a yes. A status this package does not recognise means we
    cannot tell whether the auction leg is live — and resting a second order on a maybe is the
    double-fill. Refusing costs one session's entry; guessing costs an unbounded short."""
    for unknown in ("PENDING_UPDATE", "", None, "banana"):
        assert _fn("replacement_is_owed")(unknown) is False, (
            f"{unknown!r} was treated as grounds for a second order")


# -- the decomposition is useless if the DECISION happens after the auction -------------------------

def test_an_opening_leg_is_REFUSED_for_a_slot_that_cannot_reach_the_auction():
    """THE CONTRADICTION THAT WOULD OTHERWISE SHIP SILENTLY.

    `CrsiShortStrategy`'s `decision_slots` defaults to `("open+5m",)`, which is AFTER the opening
    cross. An order decided at 09:35 cannot participate in the 09:30 auction — so a lane wired that
    way would build a perfectly correct OPG order that the venue has nothing left to match it
    against, and the whole decomposition would be inert while every test above stayed green.

    The signal permits an earlier decision: the entry limit is computed from the PRIOR close
    (`asof_close`), so the decision is available before the open. What is missing is the
    declaration, and a refusal is how that stays visible rather than becoming a comment.
    """
    for late in ("open+5m", "open+150m", "close-20m"):
        with pytest.raises(ValueError, match="auction|before the open|slot"):
            _fn("opening_leg")(**{**ARGS, "slot": late})


def test_a_PRE_OPEN_slot_is_accepted():
    """The slots that can actually reach the cross — the real grammar, not a second one.

    `open-10m` is the coordinator's ruled slot (IB accepts OPG until ~09:28 ET, so `open-5m` has no
    margin). The first version of this guard invented its own vocabulary of literal names and would
    have REFUSED it, which is the two-derivations failure: `slots._parse` already defines what a
    slot spec means, and a second definition is where the two disagree.
    """
    for ok in ("open-10m", "open-30m", "open-0m", ""):
        leg = _fn("opening_leg")(**{**ARGS, "slot": ok})
        assert leg.time_in_force == "AT_THE_OPEN", ok


def test_a_WALL_CLOCK_slot_is_REFUSED_because_it_cannot_be_PROVEN_pre_open():
    """`09:20` is before the open on a normal day and after it on a half-day with a shifted open —
    and this function has no calendar. An unprovable pre-open slot is refused rather than assumed,
    because the failure it guards against is silent: a well-formed OPG order the auction has already
    passed."""
    for clock in ("09:20", "13:00"):
        with pytest.raises(ValueError, match="auction|open-relative|cannot"):
            _fn("opening_leg")(**{**ARGS, "slot": clock})


def test_the_REPLACEMENT_may_rest_at_any_slot():
    """The day limit is the leg that exists BECAUSE the auction is over, so the slot restriction
    above must not follow it — applying the auction rule to the replacement would forbid the only
    order this lane can still place."""
    first = _fn("opening_leg")(**ARGS)
    second = _fn("replacement_leg")(first)
    assert second.time_in_force == "DAY" and second.slot == first.slot
