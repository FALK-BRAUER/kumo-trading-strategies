"""The price probe must ask about an instrument the lane actually TRADES (kumo-trading-platform issue 544).

Reported after cockpit's #532 removed the platform noise that was hiding it: THREE lanes report
`price: the strategy reported no such probe` and MOMENTUM-002 does not.

The cause is not that the probe fails. It is that the probe is never emitted, because it chose its
subject from the wrong set:

    symbols = list(getattr(self, "claimed_instruments", ()) or ())
    if symbols:
        observe("price", lambda: broker.last_price(symbols[0]))

`claimed_instruments` is `self._claims` — `external_order_claims`, documented as *"instruments this
strategy owns UNATTRIBUTED activity for"*, exclusive across the node. It is a registry of foreign
positions a lane adopts, NOT the universe it trades. MOMENTUM-002 is the only lane that has any, which
is exactly why it is the only lane with a price probe. QC345-003 watches 164 instruments and claims
none.

So a lane could be unable to price a single instrument it trades and still report no probe at all —
and "no such probe" is indistinguishable from "not applicable". An omitted probe is an absence read as
an all-clear, which is the failure shape this file exists to close.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kumo_strategies.runtime.nautilus.contract import RegistrationMixin


class _Host(RegistrationMixin):
    EXTERNAL_ID = "TEST-001"
    LABEL = "test"

    def __init__(self, *, iids=(), claims=(), price=101.5):
        self._iids = list(iids)
        self._claims = list(claims)
        self._need = 0
        self._armed_session = object()
        self._broker = SimpleNamespace(
            equity=lambda: 100_000.0,
            strategy_positions=lambda: {},
            last_price=lambda sym, **k: price,
        )

    @property
    def is_armed(self):
        return True


def _probes(host):
    return {p.name: p for p in host.preflight(host._broker)}


def test_a_lane_WITHOUT_external_claims_still_reports_a_price_probe():
    """QC345-003's shape: a full trading universe, zero external claims. It priced nothing and said
    nothing, on every boot, and cockpit could not tell that from "this lane has no prices to check"."""
    p = _probes(_Host(iids=["DELL.XNYS", "SNOW.XNAS"], claims=[]))
    assert "price" in p, (
        "a lane with 2 instruments and no external claims emitted NO price probe — absence reported "
        "as an all-clear, which is the defect")
    assert p["price"].value == 101.5


def test_the_probe_asks_about_an_instrument_the_lane_TRADES():
    """Not one it merely adopts unattributed activity for. If claims were still the subject, a lane
    holding both would be probed on the wrong symbol."""
    asked = []
    h = _Host(iids=["DELL.XNYS"], claims=["ZZZZ.XNAS"])
    h._broker.last_price = lambda sym, **k: (asked.append(sym) or 55.0)
    _probes(h)
    assert asked and asked[0] == "DELL", f"probed {asked} — that is the external-claims registry"


def test_a_lane_with_NO_instruments_says_so_rather_than_omitting_the_probe():
    """The case the old code silently produced for three lanes. "No instruments" is a FACT about the
    lane and belongs in the report; leaving the probe out makes it indistinguishable from never having
    been checked."""
    p = _probes(_Host(iids=[], claims=[]))
    assert "price" in p, "a lane with no instruments omitted the probe instead of reporting the fact"
    assert p["price"].value is None
    assert p["price"].error, "an absent subject must be stated, not left as a null with no reason"


def test_a_broker_that_cannot_price_is_reported_as_an_ERROR_not_a_silence():
    """The probe exists to catch this. It must survive a raising broker as a reported failure."""
    h = _Host(iids=["DELL.XNYS"])

    def _boom(sym, **k):
        raise RuntimeError("no market data")

    h._broker.last_price = _boom
    p = _probes(h)
    assert p["price"].value is None and "no market data" in p["price"].error
