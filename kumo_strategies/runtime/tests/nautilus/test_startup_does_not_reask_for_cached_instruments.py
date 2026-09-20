"""Start-up must not re-request what the adapter already put in the Cache (kumo-trading-platform issue 617).

THE COST, measured on ibkr-paper-retired 2026-08-28. Every lane loops its whole pool in `on_start` and
fires TWO historical requests per instrument — the definition and the bars:

    request_instrument(iid)
    request_bars(bt, start)

Across four lanes that is 2 x 372 = 744 requests in one burst:

    MOMENTUM-002   ~97   (BCT pool)          history 180 calendar days
    BCTROT-004     ~97   (same pool)         history 180
    QC345-003       164                      history 420
    TECHIVOL-005     14                      history 220

THAT ARITHMETIC IS THIS REPO'S, AND IT IS NOT THE #617 BUDGET. `request_instrument` becomes
`reqContractDetails`, which has its own IB allowance; `request_bars` becomes `reqHistoricalData`,
which carries the ~60-per-10-minutes pacing. Only the second is what starves warmup, and only the
second is what #617 paces. Skipping cached definitions therefore does NOT relieve that pacing.

The 744 above is also unattributed: it was a grep total over one 45-second boot window. Cockpit
derives ~742 for their UI feed ALONE, from seven granularities over ~106 instruments — so the two
numbers count different things and coincide by accident. Two derivations agreeing is not
corroboration when they are measuring different quantities.

What survives is the reason that never depended on the budget: since #622 the instruments are
already in the Cache by `on_start`, so this asks the venue for something we provably already hold.

HALF OF IT IS REDUNDANT. Since #622 the instruments are already in the Cache by `on_start`: the
kernel awaits `_await_engines_connected()` before starting the trader, and the IB data client pushes
every instrument the provider loaded into the Cache during that connect. Symbol resolution reads them
from exactly there — so a lane that resolved a symbol has, by construction, proved its definition is
cached.

WHAT MUST NOT BE LOST. The request is not decoration. `cache.instrument()` returning None is a hard
refusal at submit, and on the first live run every entry was priced and journalled and then rejected
with "no instrument definition cached". Bars alone do not populate it. So this GUARDS the request; it
does not remove it.
"""

from __future__ import annotations

import pytest

from nautilus_trader.model.identifiers import InstrumentId

from kumo_strategies.runtime.nautilus.contract import RegistrationMixin


class _Cache:
    def __init__(self, held: set[str]):
        self._held = held

    def instrument(self, iid):
        return object() if str(iid) in self._held else None


class _Lane(RegistrationMixin):
    def __init__(self, held: set[str]):
        self.cache = _Cache(held)
        self.requested: list[str] = []

    def request_instrument(self, iid):
        self.requested.append(str(iid))


_IID = InstrumentId.from_str("QQQ.XNAS")


def test_a_cached_instrument_is_NOT_re_requested():
    """THE SAVING. This is the common case after #622 and it is half the start-up burst."""
    lane = _Lane({"QQQ.XNAS"})
    asked = lane.request_instrument_if_missing(_IID)
    assert asked is False
    assert lane.requested == [], "asked the venue for a definition it already had"


def test_a_MISSING_instrument_IS_still_requested():
    """THE GUARANTEE THAT MUST SURVIVE, and the reason this guards rather than deletes.

    `cache.instrument()` returning None is a hard refusal at submit: every entry priced and
    journalled, then rejected with "no instrument definition cached". Bars do not populate it.
    """
    lane = _Lane(set())
    asked = lane.request_instrument_if_missing(_IID)
    assert asked is True
    assert lane.requested == ["QQQ.XNAS"], "a missing definition was not requested"


def test_the_two_cases_are_distinguishable_by_the_return():
    """A caller must be able to report how much it actually asked for rather than guessing — the
    request count is the quantity #617 is about."""
    assert _Lane({"QQQ.XNAS"}).request_instrument_if_missing(_IID) is False
    assert _Lane(set()).request_instrument_if_missing(_IID) is True


def test_a_HALF_CACHED_pool_asks_only_for_the_missing_half():
    """The realistic shape. A venue that loaded most of the pool but not all of it must produce
    exactly the missing requests — not none, and not all of them."""
    pool = [InstrumentId.from_str(f"{s}.XNAS") for s in ("AAA", "BBB", "CCC", "DDD")]
    lane = _Lane({"AAA.XNAS", "CCC.XNAS"})
    asked = [lane.request_instrument_if_missing(i) for i in pool]
    assert asked == [False, True, False, True]
    assert lane.requested == ["BBB.XNAS", "DDD.XNAS"]


def _lanes():
    """Every cockpit lane, discovered rather than listed."""
    import inspect

    from nautilus_trader.trading.strategy import Strategy

    from kumo_strategies.strategies import _layout

    out = []
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if (obj.__module__ == m.__name__ and issubclass(obj, Strategy)
                    and getattr(obj, "EXTERNAL_ID", None)):
                out.append(obj)
    return sorted(set(out), key=lambda c: c.__name__)


@pytest.mark.parametrize("cls", _lanes(), ids=lambda c: c.__name__)
def test_no_lane_asks_for_a_definition_unconditionally(cls):
    """AST-bound and aimed at every lane, because this was THREE copies of one comment and three
    copies of one call — the drift shape this repo keeps paying for.

    Bound to the parse tree rather than the source text: a grep for `request_instrument` is satisfied
    by the guarded name too, and by a comment mentioning either.
    """
    import ast
    import inspect
    import textwrap

    start = next((b.__dict__["on_start"] for b in cls.__mro__ if "on_start" in b.__dict__), None)
    try:
        src = inspect.getsource(start) if start is not None else None
    except (TypeError, OSError):
        src = None
    if src is None:
        pytest.skip(f"{cls.__name__} defines no Python `on_start`")

    called = {n.func.attr for n in ast.walk(ast.parse(textwrap.dedent(src)))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "request_instrument" not in called, (
        f"{cls.__name__}.on_start calls `request_instrument` unconditionally. Every instrument it "
        f"already has costs a request against a ~60-per-10-minute IBKR budget shared by four lanes, "
        f"and the overflow returns an empty array with no error (kumo-trading-platform issue 617).")
    if "request_bars" in called:
        assert "request_instrument_if_missing" in called, (
            f"{cls.__name__}.on_start requests bars but never requests a definition. "
            f"`cache.instrument()` returning None is a hard refusal at submit.")
