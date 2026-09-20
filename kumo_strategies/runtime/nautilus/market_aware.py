"""Every lane answers all three `MarketAware` hooks, honestly (kumo-trading-platform issue 873, #147).

THE CONSUMER IS DEPLOYED AND THE PRODUCERS WERE THE GAP. Cockpit's poller runs on ibkr-paper asking
every registered lane every 60 seconds and reads `not_asked` for all of them, because nothing
implements the protocol. `not_asked` is the unknown-read-as-pass class sitting at the very top of
the safety protocol: cockpit cannot distinguish "no lane implements this" from "every lane says
nothing is wrong". The moment the lanes answer, the poller's three states mean what they say.

WHAT AN HONEST ANSWER IS, AND WHY A STUB IS NOT ONE
-----------------------------------------------------
A stub returning `Verdict(NO)` would turn every reading green and nothing would be true. Each hook
here answers from state the lane ALREADY HOLDS, and says what it computed the answer from:

  entries_blocked   a lane with a declared view answers from `market_state` over its own bars.
                    A lane declaring `signal=NONE` answers NO **with that as the reason** — it has
                    no window measurable on its own pool's usable history, which is a finding, not
                    an absence.
  emergency_exit    no lane has a trigger wired, so every lane answers NO **naming that**. Silence
                    and "no emergency" are different facts and only one of them is true here.
  self_assessment   lanes with a registered envelope place themselves in it; lanes without answer
                    UNKNOWN **with the reason**. UNKNOWN is the third state doing its job, and for
                    a newly registered lane it is the correct answer for weeks.

TWO CONSTRAINTS THAT ARE NOT STYLE
------------------------------------
A HOOK MUST NEVER RAISE. Cockpit counts faults and pages on them, so a hook that raises is a lane
reporting itself broken. Every read here is guarded, including `self._cfg` — a lane polled seconds
after registration may hold nothing at all, and that is the normal case rather than an error.

A HOOK MUST NEVER BLOCK. There is no timeout on the poller's side; a hook that reaches for data
stalls the timers of every lane behind it. Nothing here does I/O, awaits, or sleeps. Both
constraints are asserted by the suite rather than trusted to this docstring.

WHY A MIXIN AND NOT FIVE IMPLEMENTATIONS. Five copies is how the template's omissions propagated
into CRSISHORT twice in one day — the file that is copied from is the file nobody reviews. One
implementation reading per-lane DECLARATIONS keeps the honesty per lane while keeping the mechanism
in one place, and a lane that wants a different answer overrides the hook and says why.
"""

from __future__ import annotations

from kumo_strategies.strategies.market_view import (
    NO, RISK_OFF, UNKNOWN, YES, Assessment, MarketAction, MarketSignal, Verdict, market_state)

__all__ = ["MarketAwareMixin", "MarketViewUndeclared", "NO_TRIGGER", "VIEW_OWED", "bars_to_keep",
           "lane_name_of"]


class MarketViewUndeclared(RuntimeError):
    """A lane tried to register with no declared market view (#212).

    2026-09-13: "no strategy can opt out. Those are emergency rules." A lane whose config says
    `signal=NONE`, or carries no `market_view` at all, answers the three hooks with "declares no
    view" — which the poller reads as `not_asked`, and `not_asked` is indistinguishable from
    "nothing is wrong". Raised from `MarketAwareMixin.register`, so the lane never reaches the
    trader: a lane the platform cannot govern is not allowed to exist on the node.
    """


#: LANES THAT OWE A MEASURED VIEW, by lane name -> the ticket that owes it (#212).
#:
#: Measured on b575db1 by constructing each deployed lane's live config: 5 of 6 had no declared
#: view; only TECHIVOL-005 declares one. `market_view.py` says a window is MEASURED PER LANE, NEVER
#: INHERITED — TECHIVOL's 50 is TECHIVOL's number — so these four are owed a measurement, not a
#: copied constant. Each registers today with a WARNING naming its ticket; the class test
#: (`test_no_lane_opts_out_of_the_market_view.py`) asserts this dict equals its own xfails, so a
#: name cannot be added here without a red test there, and declaring a lane's view turns its
#: xfail red-by-design until both the entry and the mark come off.
#:
#: A NEW lane is not on this list and cannot be added without a ticket: it is refused.
VIEW_OWED: dict[str, str] = {
    "MOMENTUM": "#213",
    "BCTROT": "#214",
    "QC345": "#215",
    "CRSISHORT": "#216",
}


def bars_to_keep(need: int, cfg) -> int:
    """How many bars per name an adapter must RETAIN: its own warmup, or the declared view's window
    plus dwell, whichever is larger, plus the slack every adapter already carried (#212).

    THE CAP WAS THE VIEW'S SILENT KILLER. Every adapter builds `_bars` as
    `deque(maxlen=self._need + 5)`, and `_view_prices` reads `_bars`. SMHGLD's `_need` is 3, so its
    deque held 8 bars per leg — and a 50-session average needs 51 rows. With the view declared and
    the cap untouched, `entries_blocked` would have answered UNKNOWN forever, on any `history_days`,
    while every surface read "declared". TECHIVOL only works because its warmup (100) happens to
    exceed its window. Found by the paper owner's coverage review, measured in the deployed package.

    Derived from the config, never a literal, so a lane that changes its window changes its cap.
    A lane with no declared view keeps exactly what it kept before: `need + 5`.
    """
    view = getattr(cfg, "market_view", None)
    window = 0
    if view is not None and getattr(view, "signal", MarketSignal.NONE) is not MarketSignal.NONE:
        window = int(view.window) + int(getattr(view, "dwell", 1))
    return max(int(need), window) + 5


def lane_name_of(lane) -> str:
    """The lane NAME — `MOMENTUM` from `MOMENTUM-002` — off the Nautilus strategy id.

    The id, not the class: BCTROT is a subclass of MOMENTUM's adapter and registers under its own
    name, so keying an exemption by class would exempt (or refuse) the wrong lane.
    """
    raw = getattr(lane, "id", None)
    text = str(getattr(raw, "value", raw) or "")
    return text.rsplit("-", 1)[0] if "-" in text else text

#: Said by every lane, today, and it is the literal truth: the decision layer for an emergency
#: exists (`strategies/emergency.py`) and nothing produces the signal that would call it. The one
#: emergency-shaped input in the system is `daily_loss`, which yields `Action.HALT` — the lane stops
#: DECIDING and its positions stay held, which is not an exit.
#:
#: When a trigger lands, this constant is what the grep finds.
NO_TRIGGER = ("no emergency trigger is wired for this lane, so there is nothing that could raise "
              "one — this is the absence of a detector, not the absence of an emergency")


class MarketAwareMixin:
    """The three hooks, answered from what the lane already holds.

    Mixed in FIRST in the MRO of each adapter so it cannot be shadowed by Nautilus' `Strategy`, and
    so a lane overriding a hook does so deliberately.
    """

    # -- registration: no lane opts out ---------------------------------------------------------

    def register(self, *args, **kwargs):
        """Refuse to register a lane with no declared market view (#212), then register.

        THE SEAM IS NAUTILUS' OWN. `Trader.add_strategy` calls `strategy.register(...)` from
        Python, and this mixin precedes `Strategy` in every adapter's MRO, so this runs for every
        lane cockpit hands the trader — a NEW lane included, which is the point: the rule lives on
        the class, not in a lane's `__init__` that a copier can leave out.

        AN OWED LANE REGISTERS AND SAYS SO. The lanes in `VIEW_OWED` trade today and are refused
        nothing; each boot logs the ticket that owes its view, so the exemption is a standing debt
        on the operator's surface rather than a silence. The moment such a lane declares a view the
        warning stops — the exemption is inert for a declared lane, never a pass for the name.

        A RAISE HERE LEAVES ONE ORPHAN CLOCK, HARMLESSLY. `Trader.add_strategy` registers a
        per-component clock (trader.py:421) BEFORE calling this; on refusal that LiveClock sits in
        the kernel's component list with no timers and nothing scheduling on it — and the node does
        not survive the raise anyway (cockpit calls `add_strategy` directly and lets
        misconfiguration propagate), so the process holding the orphan exits.
        """
        cfg = self._market_view_cfg()
        declared = cfg is not None and getattr(cfg, "signal", MarketSignal.NONE) is not MarketSignal.NONE
        if not declared:
            name = lane_name_of(self)
            ticket = VIEW_OWED.get(name)
            if ticket is None:
                raise MarketViewUndeclared(
                    f"{self._lane_label()} declares no market view "
                    f"({'no market_view on its config' if cfg is None else 'signal=NONE'}) and no "
                    f"lane opts out (#212). Declare `market_view=MarketViewConfig(signal=..., "
                    f"window=<measured for THIS lane>, action=...)` on its live config. The "
                    f"{len(VIEW_OWED)} lanes that still owe a measured view are {sorted(VIEW_OWED)}, "
                    f"each under a ticket; a new lane is not added to that list, it declares.")
            log = getattr(self, "log", None)
            if log is not None:
                try:
                    log.warning(
                        f"{self._lane_label()}: registering with NO declared market view — "
                        f"{name} owes a measured one under {ticket} (#212). Its three hooks answer "
                        f"'declares no view' until then, which the poller reads as not_asked.")
                except Exception:                                       # noqa: BLE001
                    pass
        return super().register(*args, **kwargs)

    # -- may I open anything? --------------------------------------------------------------------

    def entries_blocked(self) -> Verdict:
        """Does this lane's own view of its own market say stop opening?

        THE LANE'S MARKET, NOT THE MARKET. `market_state` reads an equal-weight index of the names
        this lane actually ranks — a cap-weighted proxy would have a small-cap lane de-risking on
        four mega-caps it never holds.

        YES ONLY UNDER A DECLARED ACTION. `MarketState.blocks_entries` is False when `action` is
        None, which is the state of a lane that declared a signal and no action — and that
        combination is refused at config construction, so reaching it here means something
        upstream is wrong and the honest answer is still "not blocking".
        """
        cfg = self._market_view_cfg()
        if cfg is None or cfg.signal is MarketSignal.NONE:
            # A DEBT, NOT A DECISION (#212). This used to say "a declared NONE is a measurement
            # result, not a gap" — false prose beside a value that is now owed. The poller journals
            # this reason on every poll, so the ticket is visible per poll, not only in the boot
            # WARNING `register` emits.
            return Verdict(NO, (self._undeclared_reason(cfg),))
        prices = self._view_prices()
        if prices is None or getattr(prices, "empty", True):
            return Verdict(UNKNOWN, (
                f"{self._lane_label()} declares {cfg.signal.value} over {cfg.window} sessions and "
                f"holds no price history yet, so its view cannot be computed. UNKNOWN does not "
                f"block — a view we could not compute is not evidence the market is bad.",))
        state = self._market_state(prices, cfg)
        if state is None:
            return Verdict(UNKNOWN, (
                f"{self._lane_label()}: the view could not be computed from the bars on hand.",))
        if state.state == UNKNOWN:
            return Verdict(UNKNOWN, state.reasons or ("the view could not be computed",))
        if state.blocks_entries:
            return Verdict(YES, state.reasons + (
                f"action={cfg.action.value if cfg.action else None}",))
        return Verdict(NO, state.reasons or ("the view is risk-on",))

    # -- must I close what I hold? ----------------------------------------------------------------

    def emergency_exit(self) -> Verdict:
        """Is there a reason to flatten this lane's book NOW?

        NO LANE HAS A TRIGGER, AND EVERY LANE SAYS SO. Answering NO with an empty reason would make
        "nothing is wrong" and "nothing is watching" identical on the operator's surface, which is
        the failure this whole protocol is built against.

        A RISK-OFF VIEW IS NOT AN EMERGENCY UNLESS IT SAYS IT IS. A lane whose declared action is
        LIQUIDATE and whose view is risk-off is asking to be flattened, and that is what this hook
        reports. A lane on EXIT_ONLY is NOT: "today is not the day to buy" and "get out now" are
        different claims, and collapsing them is the #144 defect — a flag named for blocking that
        performed a liquidation, under which every figure published for one act measured the other.
        """
        cfg = self._market_view_cfg()
        if cfg is None or cfg.signal is MarketSignal.NONE:
            return Verdict(NO, (NO_TRIGGER,
                                f"{self._lane_label()} also declares no market view that could "
                                f"escalate to one."))
        if cfg.action is not MarketAction.LIQUIDATE:
            return Verdict(NO, (
                NO_TRIGGER,
                f"{self._lane_label()} declares action="
                f"{cfg.action.value if cfg.action else None}, which stops it OPENING and never asks "
                f"for the book to be flattened. Blocking entries and liquidating are different "
                f"acts and this lane has chosen the first.",))
        prices = self._view_prices()
        state = self._market_state(prices, cfg) if prices is not None and not getattr(
            prices, "empty", True) else None
        if state is None or state.state == UNKNOWN:
            return Verdict(UNKNOWN, (
                NO_TRIGGER,
                f"{self._lane_label()} declares LIQUIDATE on a view that cannot be computed yet, "
                f"so whether it would ask to be flattened is unknown. NOTHING ACTS ON UNKNOWN.",))
        if state.liquidates:
            return Verdict(YES, state.reasons + (
                f"{self._lane_label()} declares action=liquidate, so a risk-off view is a request "
                f"to flatten the book.",))
        return Verdict(NO, (NO_TRIGGER,) + (state.reasons or ("the view is risk-on",)))

    # -- how am I doing? ---------------------------------------------------------------------------

    def self_assessment(self) -> Assessment:
        """Is this lane doing what it said it would?

        A DIFFERENT QUESTION FROM ITS MARKET'S. "The market is falling" and "I am not working" want
        different responses, and a lane that cannot tell them apart de-risks for the wrong reason.

        UNKNOWN IS THE RIGHT ANSWER FOR MOST LANES TODAY and must not be dressed up as health.
        Answering IN_ENVELOPE without an envelope would be a health claim nothing supports — the
        exact unknown-read-as-pass shape this protocol exists to remove. A lane needs a registered
        envelope AND accumulated live windows before it can place itself in a distribution, and
        until then it says so.
        """
        env = self._registered_envelope()
        if env is None:
            return Assessment(UNKNOWN, (
                f"{self._lane_label()} has no registered envelope, so it has no pre-registered "
                f"distribution to place itself in. Nothing has been fitted for this lane, which is "
                f"a gap in evidence rather than a verdict about its health.",),
                evidence_sufficient=False)
        return Assessment(UNKNOWN, (
            f"{self._lane_label()} has a registered envelope and no accumulated LIVE windows to "
            f"compare against it. Envelope evidence starts at the lane's own coverage floor and "
            f"cannot be back-filled — no price fetch moves it — so this stays UNKNOWN until live "
            f"sessions accumulate.",), evidence_sufficient=False)

    # -- what the hooks read, each guarded ---------------------------------------------------------

    def _undeclared_reason(self, cfg) -> str:
        """Why a lane with no declared view answers NO: an owed lane names its ticket; any other
        undeclared lane is refused at registration (`register`) and can only be polled on a host."""
        how = "no market_view on its config" if cfg is None else "signal=NONE"
        ticket = VIEW_OWED.get(lane_name_of(self))
        if ticket is not None:
            return (f"{self._lane_label()} owes a MEASURED market view under {ticket} (#212; "
                    f"{how}) — it cannot block its own entries and reads not_asked until then")
        return (f"{self._lane_label()} declares no market view ({how}) and no lane opts out "
                f"(#212): such a lane is refused at registration, so this answer can only come "
                f"from a host outside the node")

    def _lane_label(self) -> str:
        """Never raises, never blocks. `self.id` is a Nautilus attribute that is absent on a lane
        that has not been through `Strategy.__init__`, which is exactly the shape the poller can
        meet just after registration."""
        for attr in ("id", "EXTERNAL_ID", "LABEL"):
            try:
                got = getattr(self, attr, None)
            except Exception:                                           # noqa: BLE001
                continue
            if got:
                return str(got)
        return type(self).__name__

    def _market_view_cfg(self):
        """The lane's declared view, or None if it declares nothing.

        A DIRECT ATTRIBUTE READ INSIDE THE GUARD, not `getattr(cfg, "market_view", None)`.

        Both are safe; only one is VISIBLE. `test_every_backtest_enforced_field_is_read_live_or_
        DECLARED_dead` walks the AST for reads of each enforced field, and a name passed as a
        string is not a read it can see — so the `getattr` form left `market_view` recorded as
        "enforced in the backtest and read by nothing live" while this module was reading it on
        every poll. The detector was right to keep failing: a dynamic read is exactly as invisible
        to a human auditor as it is to the walker.

        The exception path is load-bearing rather than defensive. CRSISHORT's config has no
        `market_view` field at all, and a lane polled before construction has no `_cfg`. Both are
        legitimate states that must produce an ANSWER rather than an exception.
        """
        try:
            cfg = self._cfg
        except AttributeError:
            return None
        if cfg is None:
            return None
        try:
            return cfg.market_view
        except AttributeError:
            # A lane that declares no view at all — CRSISHORT today. Not an error and not a gap:
            # `entries_blocked` turns it into "this lane never blocks its own entries", with the
            # reason.
            return None

    def _view_prices(self):
        """A date-indexed frame of closes per ticker, from the bars the lane is already holding.

        NO I/O. The lane subscribes to its own bars and accumulates them; this reshapes what is
        already in memory. A hook that asked the cache or the venue for prices would block the
        poller, which has no timeout.
        """
        try:
            import pandas as pd

            bars = getattr(self, "_bars", None)
            if not bars:
                return None
            rows = [row for sym in bars for row in bars[sym]]
            if not rows:
                return None
            frame = pd.DataFrame(rows)
            if not {"ticker", "date", "close"}.issubset(frame.columns):
                return None
            return frame.pivot_table(index="date", columns="ticker", values="close").sort_index()
        except Exception:                                               # noqa: BLE001
            return None

    def _market_state(self, prices, cfg):
        """`market_state` with its exceptions absorbed. It is pure and should not raise, and this
        hook is not the place to discover that it can."""
        try:
            return market_state(prices, cfg, asof=prices.index.max() + _one_day())
        except Exception:                                               # noqa: BLE001
            return None

    def _registered_envelope(self):
        """This lane's pre-registered envelope, or None.

        A lane declares its registry by setting `ENVELOPE_REGISTRY` and `envelope_key()`. Absent
        either, it has no envelope — which is an honest UNKNOWN, not an error.
        """
        try:
            registry = getattr(self, "ENVELOPE_REGISTRY", None)
            if not registry:
                return None
            key = self.envelope_key()
            return registry.get(key) if key else None
        except Exception:                                               # noqa: BLE001
            return None

    def envelope_key(self) -> str | None:
        """Which registered envelope describes THIS lane's configuration. None by default: a lane
        that has not said cannot be assumed into someone else's distribution."""
        return None


def _one_day():
    import pandas as pd
    return pd.Timedelta(days=1)
