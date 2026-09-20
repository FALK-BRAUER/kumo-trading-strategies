"""The trading calendar may come from the VENUE, not only from Alpaca (kumo-trading-platform issue 628).

THE FAILURE, measured on ibkr-paper-retired 2026-08-28 22:35 SGT, deploying #622 with no APCA_* set:

    RuntimeError: no APCA credentials, so only the HOLIDAY-UNAWARE WeekdayCalendar is available —
    refusing it here because this path can place orders and would schedule sessions on market holidays
      runtime/calendar.py:146   build_calendar(require_exchange=True)
      strategies/momentum.py:670  _build_rotation
      api/engine_node.py:6875     build_node()

THE REFUSAL IS CORRECT AND STAYS. `require_exchange=True` exists so a misconfigured live environment
cannot schedule a session on Thanksgiving, and relaxing it to obtain a boot would trade a visible
outage for an invisible wrong trade. What is wrong is that ALPACA is the only thing that can satisfy
it — an IBKR-only instance had no way to supply a calendar at all.

THE VENUE ALREADY ANSWERS THIS. Decoded from staging's real cached SPY instrument, not from docs:

    timeZoneId     US/Eastern
    liquidHours    20260826:0930-20260826:1600;...;20260829:CLOSED;20260830:CLOSED;...

`liquidHours` is the regular session with CLOSED days marked — weekends and holidays identically —
and it is already in the Nautilus cache with no HTTP call. Third instance of the same pattern in one
day: #622 (venue), #624 (asset type), this (calendar), all reachable from `Instrument.info` all along.

THIS FILE CHANGES NO DEFAULT. Injection is additive: an injected calendar wins, otherwise Alpaca when
credentials exist, otherwise the same refusal. Both tenants keep the behaviour they have.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from kumo_strategies.runtime.calendar import (
    ET, AlpacaCalendar, TradingDay, WeekdayCalendar, build_calendar,
)


class _VenueCalendar:
    """The narrowest thing satisfying the REAL calendar contract, standing in for cockpit's.

    `is_trading_day`, not `is_open` — the first version of this double invented the latter and the
    fixture-property test below caught it before anything was built on top. A double that cannot
    represent production is the bug, and this repo has shipped defects behind that exact mistake.
    """

    def __init__(self, open_days: set[date]):
        self._open = open_days

    def day(self, d: date):
        if d not in self._open:
            return None
        return TradingDay(d, datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET),
                          datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET))

    def is_trading_day(self, d: date) -> bool:
        return self.day(d) is not None

    def next_fire(self, after: datetime, offset_minutes: int):
        d = after.astimezone(ET).date()
        for _ in range(14):
            td = self.day(d)
            if td is not None:
                fire = td.open_at + timedelta(minutes=offset_minutes)
                if fire > after.astimezone(ET):
                    return td.session, fire
            d += timedelta(days=1)
        raise RuntimeError("no trading day within 14 days")


def test_the_double_matches_the_REAL_calendar_contract():
    """FIXTURE PROPERTY FIRST, and it earned its place immediately.

    The first version of the double exposed `is_open`; the real contract is `is_trading_day`. Every
    assertion below would have run against an object production could never accept — and it was this
    test, not any of them, that noticed.

    It also pins that the double can DISAGREE with the fallback: if the injected calendar answered
    the same as `WeekdayCalendar` for every date, injection could be ignored entirely and nothing
    here would fail.
    """
    d = date(2026, 8, 28)                       # a Friday — WeekdayCalendar says open
    assert WeekdayCalendar().is_trading_day(d)
    assert not _VenueCalendar(set()).is_trading_day(d), "the double cannot disagree with the fallback"


def test_an_INJECTED_calendar_satisfies_require_exchange_without_alpaca(monkeypatch):
    """THE DEFECT. An IBKR-only instance must be able to supply a holiday-aware calendar and boot."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    cal = _VenueCalendar({date(2026, 8, 28)})
    assert build_calendar(require_exchange=True, calendar=cal) is cal


def test_it_STILL_REFUSES_when_nothing_can_supply_one(monkeypatch):
    """THE GUARANTEE THAT MUST SURVIVE. No credentials AND no injected calendar means there is no
    holiday-aware answer available — and a path that places orders must not proceed on a guess."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="HOLIDAY-UNAWARE"):
        build_calendar(require_exchange=True)


def test_ALPACA_still_wins_when_no_calendar_is_injected(monkeypatch):
    """The trading tenant is UNCHANGED. Both tenants run this package on one pinned ref, so this must
    alter nothing for the one that is actually placing orders."""
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    assert isinstance(build_calendar(require_exchange=True), AlpacaCalendar)


def test_an_INJECTED_calendar_beats_alpaca_when_both_are_available(monkeypatch):
    """The instance's OWN venue is more authoritative about its sessions than a third party is.

    It also matters for the rule that made this necessary: staging must not consult Alpaca even when
    a credential happens to be present in the environment, because a credential that is present is a
    credential that WILL be used (#573).
    """
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    cal = _VenueCalendar({date(2026, 8, 28)})
    assert build_calendar(require_exchange=True, calendar=cal) is cal


def test_the_fallback_path_is_untouched(monkeypatch):
    """`require_exchange=False` still degrades to the weekday calendar. Backtests and research paths
    rely on it and they place no orders."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    assert isinstance(build_calendar(require_exchange=False), WeekdayCalendar)
