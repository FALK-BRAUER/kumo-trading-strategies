"""The trading calendar CONTRACT, and two implementations of it. Spec: #189, kumo-trading-platform issue 628.

The venue that decides whether an order can be filled is the authority on when it is open, so a
weekday heuristic that disagrees with it is simply wrong. This matters on half-days (the close moves
to 13:00 ET) and on holidays, where a naive Mon-Fri scheduler would fire into a closed market and
record a session that never traded.

The cockpit's own session guard (engine_node.py:2743) is holiday-unaware by its own admission; this
deliberately is not.

WHO SUPPLIES IT IS NOT THIS MODULE'S BUSINESS. It was, and that was a defect: `AlpacaCalendar` was
the only thing that could satisfy `require_exchange=True`, so an IBKR-only instance had no way to
supply a holiday-aware calendar at all and could not boot — measured on ibkr-paper-retired 2026-08-28,
after #622 had already cleared the first Alpaca dependency out of the same boot path. The venue
answers this itself: IB's `Instrument.info` carries `liquidHours` with CLOSED days marked, weekends
and holidays identically, already in the Nautilus cache with no HTTP call. So a caller may now inject
one, and this module keeps only the contract:

    day(d)                        -> TradingDay | None, or raises OutsideCalendarWindow
    next_fire(after, offset_mins) -> (session, fire_at)
    is_trading_day(d)             -> bool

THREE STATES, NOT TWO. A venue calendar knows a ROLLING window — IB's is about six days — so
"the venue never described this date" is a real answer and is NOT the same as "closed". Closed is
silent by design; unknown must not be. See `OutsideCalendarWindow`.

Absence must not be readable as permission.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
_BASE = "https://paper-api.alpaca.markets"

#: How far forward a scheduling scan looks for the next trading day.
SCAN_DAYS = 45
#: How far `_ensure` warms around a date. THE FORWARD HALF MUST COVER `SCAN_DAYS`, and that is what
#: couples these two: `_ensure` is a RANGE check, not a cache check, so any date outside the warmed
#: span refetches — and the scan runs back ON THE EVENT LOOP once `warm()` has returned. A scan that
#: reached one day past the warmed span would put a blocking `urlopen` on the loop, which is the
#: defect `warm()` exists to remove, reached through the code path that removes it.
#:
#: It cannot happen today: the scan breaks at the first trading day, 0-4 days out, never 45. Named
#: rather than left as two independent literals so that widening the scan moves the span with it.
WARM_BACK_DAYS = 10
WARM_FORWARD_DAYS = SCAN_DAYS


@dataclass(frozen=True)
class TradingDay:
    session: date
    open_at: datetime          # tz-aware, ET
    close_at: datetime

    @property
    def is_half_day(self) -> bool:
        return (self.close_at - self.open_at) < timedelta(hours=6)


@dataclass
class AlpacaCalendar:
    """Trading days, cached. A fetch failure is fatal to scheduling on purpose — guessing whether
    the market is open is exactly the kind of assumption that fires an order into a closed venue."""

    key: str = field(default_factory=lambda: os.environ.get("APCA_API_KEY_ID", ""))
    secret: str = field(default_factory=lambda: os.environ.get("APCA_API_SECRET_KEY", ""))
    base: str = _BASE
    _cache: dict[date, TradingDay] = field(default_factory=dict, repr=False)
    _span: tuple[date, date] | None = field(default=None, repr=False)

    def _fetch(self, start: date, end: date) -> None:
        req = urllib.request.Request(
            f"{self.base}/v2/calendar?start={start}&end={end}",
            headers={"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret})
        with urllib.request.urlopen(req, timeout=15) as r:
            rows = json.loads(r.read())
        for d in rows:
            day = date.fromisoformat(d["date"])
            oh, om = (int(x) for x in d["open"].split(":"))
            ch, cm = (int(x) for x in d["close"].split(":"))
            self._cache[day] = TradingDay(
                day,
                datetime(day.year, day.month, day.day, oh, om, tzinfo=ET),
                datetime(day.year, day.month, day.day, ch, cm, tzinfo=ET))
        self._span = (start, end)

    def _ensure(self, around: date) -> None:
        if self._span and self._span[0] <= around <= self._span[1]:
            return
        self._fetch(around - timedelta(days=WARM_BACK_DAYS),
                    around + timedelta(days=WARM_FORWARD_DAYS))

    def day(self, d: date) -> TradingDay | None:
        self._ensure(d)
        return self._cache.get(d)

    def is_trading_day(self, d: date) -> bool:
        return self.day(d) is not None

    def next_fire(self, after: datetime, offset_minutes: int) -> tuple[date, datetime]:
        """The next (session, fire time) at `offset_minutes` past the open.

        Offset is from the OPEN, not a wall-clock time, so a half-day still fires correctly.
        """
        d = after.astimezone(ET).date()
        for _ in range(SCAN_DAYS):
            td = self.day(d)
            if td is not None:
                fire = td.open_at + timedelta(minutes=offset_minutes)
                if fire > after.astimezone(ET):
                    return td.session, fire
            d += timedelta(days=1)
        raise RuntimeError(f"no trading day found in the next {SCAN_DAYS} days — calendar looks wrong")


@dataclass
class WeekdayCalendar:
    """Fallback with no broker credentials: Mon-Fri, 09:30-16:00 ET, HOLIDAY-UNAWARE.

    Only for local development. It will happily schedule a session on Thanksgiving.
    """

    def day(self, d: date) -> TradingDay | None:
        if d.weekday() >= 5:
            return None
        return TradingDay(d, datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET),
                          datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET))

    def is_trading_day(self, d: date) -> bool:
        return self.day(d) is not None

    def next_fire(self, after: datetime, offset_minutes: int) -> tuple[date, datetime]:
        d = after.astimezone(ET).date()
        for _ in range(14):
            td = self.day(d)
            if td is not None:
                fire = td.open_at + timedelta(minutes=offset_minutes)
                if fire > after.astimezone(ET):
                    return td.session, fire
            d += timedelta(days=1)
        raise RuntimeError("no weekday found — impossible")


#: Everything a calendar must answer, whoever supplies it.
#:
#: `next_fire` is on this list and it is the one an injected calendar is most likely to omit: it is a
#: METHOD on the two calendars here, not a module helper, and only QC27 and QC345 call it —
#: `qc27_rotation.py:312`, `qc345_rotation.py:352`. A calendar satisfying only `day` passes every
#: test that goes through `warm` / `next_slot_fire` / `elapsed_slots` and then raises AttributeError
#: the first time TECHIVOL-005 or QC345-003 arms, on a path that retries quietly.
_CALENDAR_CONTRACT = ("day", "next_fire", "is_trading_day")


class OutsideCalendarWindow(RuntimeError):
    """The venue never described this date — as distinct from having said it is CLOSED.

    A venue-supplied calendar knows a ROLLING window; IB returns roughly six days. Beyond it there is
    no answer, and the whole point of this module is that a guess about whether the market is open is
    worse than not scheduling at all.

    Conflating unknown with closed is the dangerous direction, because closed is SILENT by design:

        elapsed_slots(cal, at, specs)  ->  td = cal.day(now.date())
                                           if td is None: return []          # "nothing was due"

    A day nobody described would report that nothing was due, which is indistinguishable from a real
    quiet day. That is the 2026-08-17 failure — a decision that never happens and leaves no trace —
    reached through the window instead of through a restart.

    Raised, not returned, so it cannot be dropped by a caller that only checks for None. Arming
    already retries at `ARM_RETRY_SECS`, so a lane that cannot yet see far enough simply tries again
    in a minute and succeeds once the venue extends its window.
    """


def _check_calendar_contract(cal: object) -> None:
    """Refuse an injected calendar that cannot answer, AT BUILD rather than at first arm.

    Structural, because the failure it prevents is structural. A calendar missing `next_fire` is
    perfectly usable by three of five call sites, so it boots, passes preflight and only fails when
    the two lanes that use it try to schedule — inside arming, which catches and retries, so the lane
    stays silently unarmed rather than crashing.
    """
    missing = [m for m in _CALENDAR_CONTRACT if not callable(getattr(cal, m, None))]
    if missing:
        raise TypeError(
            f"{type(cal).__name__} cannot be used as a calendar: missing {', '.join(missing)}. "
            f"The full contract is {', '.join(_CALENDAR_CONTRACT)} — `next_fire(after, "
            f"offset_minutes)` is the one usually forgotten, because only QC27 and QC345 call it "
            f"and they call it during arming, where the AttributeError would be caught and retried "
            f"rather than surfaced. `day()` must return a `TradingDay` (session, open_at, close_at, "
            f"tz-aware ET), None for a session the venue says is CLOSED, and raise "
            f"OutsideCalendarWindow for a date the venue never described.")


def build_calendar(require_exchange: bool = False, calendar: object | None = None):
    """The venue's own calendar if one is injected, else the broker's, else the weekday fallback.

    `require_exchange=True` refuses the fallback instead of returning it. Anything that will place
    real orders must pass it. The fallback is holiday-unaware by design and says so, but a silent
    downgrade turns that honesty into a trap: a misconfigured live environment gets a calendar that
    will happily schedule a session on Thanksgiving, and nothing in the logs distinguishes it from
    the real thing until the orders bounce.

    `calendar` SATISFIES that requirement rather than relaxing it (kumo-trading-platform issue 628). The distinction
    is the whole design: an IBKR-only instance could not supply a holiday-aware calendar at all, so
    the only way to boot it was to pass `require_exchange=False` — trading a visible outage for an
    invisible wrong trade. The three cockpit call sites keep their explicit `True` and change
    nothing; they gain a way to MEET it.

    An injected calendar also beats Alpaca when both are available. The instance's own venue is more
    authoritative about its sessions than a third party is, and a credential that happens to be
    present in the environment is a credential that WILL be used (kumo-trading-platform issue 573).

    This repo keeps the CONTRACT and stops being the place that knows which vendor supplies it —
    the same move as #622, where the layer that knows the rule stopped being the layer that knows
    the venue.
    """
    if calendar is not None:
        _check_calendar_contract(calendar)
        return calendar
    if os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY"):
        return AlpacaCalendar()
    if require_exchange:
        raise RuntimeError(
            "no APCA credentials and no calendar supplied, so only the HOLIDAY-UNAWARE "
            "WeekdayCalendar is available — refusing it here because this path can place orders and "
            "would schedule sessions on market holidays. An IBKR-only instance should pass "
            "`calendar=` built from the venue's own trading hours (kumo-trading-platform issue 628).")
    return WeekdayCalendar()


def warm(cal, at: datetime) -> None:
    """Populate `cal`'s cache for `at`'s date. THE ONLY BLOCKING CALL IN SCHEDULING.

    Isolated into one function for one reason: it can then be run OFF the event loop. `AlpacaCalendar`
    fetches lazily, so the first `.day()` anywhere in a scheduling path is a synchronous `urlopen`
    with a 15 second timeout — and every caller of `next_slot_fire` and `elapsed_slots` was reaching
    it from `on_start`, on the loop.

    Nothing here catches. A calendar that cannot be reached must still be fatal to SCHEDULING — the
    module's whole premise is that guessing whether the venue is open is worse than not scheduling.
    What changes is only WHERE that failure lands: on the caller's retry, not on the node's start.
    """
    cal.day(at.astimezone(ET).date())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def next_slot_fire(cal, after: datetime, specs: tuple[str, ...], *,
                   allow_pre_open: bool = False) -> tuple[date, str, datetime]:
    """The next (session, slot name, fire time) across ALL configured slots.

    `next_fire` answers "when is the next session's single decision". That question has no answer
    once a strategy decides more than once a day, and the live adapter was built around it: one
    alert, one offset, `override=True` so re-arming reuses the one slot. Multi-slot existed in the
    backtest runner and in the decision store's `(session, slot)` key while the thing that actually
    decides could still only fire once.

    Resolution goes through `momentum_rotation.slots.resolve`, the SAME function the backtest uses,
    rather than a second offset-from-open calculation here. Live and backtest disagreeing about when
    a decision happens is the recurring defect shape in this repo — the give-back rule, the ATR
    rules and the highs all diverged that way — and a slot resolved differently on the two sides
    would make every cadence result unverifiable rather than merely wrong.

    `allow_pre_open` IS PLUMBED INTO BOTH THIS AND `elapsed_slots`, OR NEITHER (#131). They call
    the same `resolve` deliberately, for the reason above; giving the flag to one recreates that
    divergence INSIDE the runtime. This one would arm 09:20 while `elapsed_slots` computed 09:30
    for the same name, so a process starting at 09:25 would believe the slot had not passed and
    decide twice, or start at 09:35 and never notice the pre-open slot was lost.

    Returns the slot NAME, not the offset, because that name is the idempotency key: `2026-08-14/
    close-20m` survives a restart or a retry, where a wall-clock timestamp cannot be told apart from
    a genuinely new decision.
    """
    from kumo_strategies.strategies.momentum_rotation.slots import resolve

    at = after.astimezone(ET)
    d = at.date()
    for _ in range(SCAN_DAYS):
        td = cal.day(d)
        if td is not None:
            for name, when in resolve(specs, td.open_at, td.close_at,
                                      allow_pre_open=allow_pre_open):
                if when > at:
                    return td.session, name, when
        d += timedelta(days=1)
    raise RuntimeError(f"no trading day found in the next {SCAN_DAYS} days — calendar looks wrong")


def elapsed_slots(cal, at: datetime, specs: tuple[str, ...], *,
                  allow_pre_open: bool = False) -> list[tuple[date, str, datetime]]:
    """Today's slots whose fire time has ALREADY PASSED at `at`.

    Exists because a restart inside the decision window silently loses the session. `_arm` asks for
    the NEXT fire, so a process that starts at 09:36 arms tomorrow and today simply never happens —
    no alert, no decision row, no log line. `missed_sessions` does not catch it either, because that
    is only appended when a session was already DUE, and a restart before the alert fires means it
    never became due.

    It cost a live session on 2026-08-17: the engine restarted at 09:35 ET, exactly the decision
    minute, armed forward to the 18th, and MOMENTUM-002 produced no decision row for the day. From
    the outside "skipped because we restarted" and "held because nothing ranked" are the same
    picture — which is the failure shape this codebase keeps paying for, arriving through the clock
    this time instead of a config flag.

    Returns the slots rather than a bool so the caller can name them: an operator needs to know WHICH
    decision was lost, not merely that something was.
    """
    from kumo_strategies.strategies.momentum_rotation.slots import resolve

    now = at.astimezone(ET)
    td = cal.day(now.date())
    if td is None:
        return []
    return [(td.session, name, when)
            for name, when in resolve(specs, td.open_at, td.close_at,
                                      allow_pre_open=allow_pre_open) if when <= now]
