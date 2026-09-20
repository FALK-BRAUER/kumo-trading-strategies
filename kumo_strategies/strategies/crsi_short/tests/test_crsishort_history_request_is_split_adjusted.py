"""CRSISHORT's history request must SAY what it needs: split-adjusted bars, kept out of the shared
cache (kumo-trading-platform issue 1124, issue 261).

The constructor already refuses a lane built on anything but `price_adjustment=ADJUSTED`; this is
the other half of that fact — the request that fetches the history has to carry it, because on the
Alpaca path the REST default is `raw` and nothing downstream can tell an adjusted series from a raw
one by looking at it. 29 of the 130 acceptance names carry a split inside the 200-day warmup
window; on raw bars a split day reads as a -50% move and the screen and the exits are wrong by
construction (#123).

`disable_historical_cache` rides along because the 1-DAY bar type is shared with the chart and with
lanes measured on raw bars: an adjusted series overwriting their cache rows for the same timestamps
would be last-write-wins on a fact two consumers disagree about.

BOUND TO THE INSTALLED SIGNATURE, not to a fake. The recorder binds every call through
`inspect.signature(Strategy.request_bars)`, so a keyword Nautilus does not accept fails here rather
than at boot on the node — `params` is asserted as a KEYWORD, since a positional slip would land
in `end`/`limit` and the request would still "work".
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pandas as pd
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.strategies.crsi_short.nautilus import HISTORY_REQUEST_PARAMS, CrsiShortStrategy
from kumo_strategies.runtime.nautilus.held_seed import HeldSeedMixin
from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig

_SIG = inspect.signature(Strategy.request_bars)


class _Host(HeldSeedMixin):
    """The REAL `on_start` and `_bar_type`, on a host that records what was requested."""

    on_start = CrsiShortStrategy.on_start
    _bar_type = CrsiShortStrategy._bar_type
    id = "CRSISHORT-001"

    def __init__(self):
        self.requests: list[inspect.BoundArguments] = []
        self.subscribed: list = []
        self._history_days = 200
        self._need = CrsiShortConfig().warmup_sessions
        self._min_warm = 1
        self._slots = ("open-10m", "open+5m")
        self._suffix = "-1-DAY-LAST-EXTERNAL"
        self._iids = [InstrumentId.from_str("SOXS.ARCX"), InstrumentId.from_str("NVDA.XNAS")]
        self._held = set()
        self.cache = SimpleNamespace(positions_open=lambda **kw: [])
        self.clock = SimpleNamespace(utc_now=lambda: pd.Timestamp("2026-09-18T01:00:00Z"))
        self.log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                                   error=lambda *a, **k: None, debug=lambda *a, **k: None)
        self._loop = None
        self._account_topic = "broker.account"
        self._on_broker_account = lambda *a, **k: None
        self.msgbus = SimpleNamespace(subscribe=lambda topic, handler: None)

    # -- the seams `on_start` touches, each one recorded or inert ------------------------------
    def request_bars(self, *args, **kwargs):
        self.requests.append(_SIG.bind(self, *args, **kwargs))

    def subscribe_bars(self, bt):
        self.subscribed.append(bt)

    def request_instrument_if_missing(self, iid):
        pass

    def _resolve_symbols_if_needed(self):
        pass

    def reconcile_pending(self):
        pass

    def begin_arming(self):
        pass


def test_the_history_request_asks_for_split_adjusted_bars_outside_the_shared_cache():
    lane = _Host()
    CrsiShortStrategy.on_start(lane)
    assert len(lane.requests) == 2, "one history request per instrument"
    for bound in lane.requests:
        assert "params" in bound.kwargs, (
            f"`params` must travel as a keyword; got {bound.arguments!r}")
        assert bound.kwargs["params"] == {"adjustment": "split", "disable_historical_cache": True}
        assert bound.kwargs["params"] == HISTORY_REQUEST_PARAMS


def test_the_declared_params_are_the_two_facts_and_nothing_else():
    """A third key is a new decision, not a free rider: it would reach every venue adapter."""
    assert HISTORY_REQUEST_PARAMS == {"adjustment": "split", "disable_historical_cache": True}
