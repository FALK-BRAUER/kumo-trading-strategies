"""Every adapter's OWN trail-exit path honours an ATR-scaled exit rule, or says it cannot (#222).

`evaluate_exits` RAISES when `needs_atr(cfg)` is true and no `atr` is passed (exits.py:190) — on
purpose: silently arming without the input is the #26 shape, and this module already lost a rule to
it once. The runner path passes it (pgrunner.py:1085 → :676). The ADAPTERS' own `_trail_exits` —
the path a lane takes when it is built without a `session_runner` — did not:

    momentum_rotation.py:886           evaluate_exits(self._cfg.exits, prices, states)
    momentum_rotation_intraday.py:312  evaluate_exits(self._cfg.exits, prices, states)
    qc345_rotation.py:652              evaluate_exits(..., atr=atr, highs=highs)     <- the one that did

So with `take_profit_atr=4.5` — settable today via platform issue 1055 — MOMENTUM's adapter path raises
inside `_try_decide` instead of exiting. Dead for every deployed lane (all runner-wired,
momentum_rotation.py:564), a landmine for a shadow lane, a test host, or the standalone path the
template teaches. The honest fix is to PASS the input, as qc345 and pgrunner do, not to refuse
construction: the path is legitimate, it was just fed half its inputs.

Class-level: every adapter with a `_trail_exits` is driven on a narrow host with bars, a position
5 ATR in profit and `take_profit_atr=4.5` — it must not raise, and it must EXIT. Seen red on
1f5fe2e for momentum_rotation (raise) and momentum_rotation_intraday (raise); green for qc345.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pandas as pd
import pytest
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from kumo_strategies.strategies import _layout
from kumo_strategies.strategies.momentum_rotation.config import ExitConfig, MomentumRotationConfig
from kumo_strategies.strategies.qc345_rotation.config import QC345RotationConfig


def _adapters_with_trail_exits() -> list[type]:
    found: dict[str, type] = {}
    # Every lane module, from the ONE discovery point (strategies/_layout.py, ks#211). An import
    # failure raises here rather than being skipped: a lane that cannot import is not "absent".
    for m in _layout.lane_modules():
        for obj in vars(m).values():
            if (isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy
                    and "_trail_exits" in vars(obj)):
                found[obj.__name__] = obj      # by name: bctrot re-imports MomentumRotationStrategy
    return sorted(found.values(), key=lambda c: c.__name__)


ADAPTERS = _adapters_with_trail_exits()
#: The ones whose `_trail_exits` takes no arguments and reads the adapter's own bar deque — the
#: drivable shape. QC345's takes a `session` and reads a frame it builds; it already passes atr
#: (qc345_rotation.py:652) and is pinned by the source test below.
DRIVABLE = [c for c in ADAPTERS if list(inspect.signature(c._trail_exits).parameters) == ["self"]]


def test_the_discovery_found_the_three_adapters_that_evaluate_exits_on_their_own_path():
    names = {c.__name__ for c in ADAPTERS}
    assert {"MomentumRotationStrategy", "IntradayMomentumRotation", "QC345RotationStrategy"} <= names, names
    assert {c.__name__ for c in DRIVABLE} == {"MomentumRotationStrategy", "IntradayMomentumRotation"}
    assert "QC345RotationStrategy" in names       # driven too, through `_drive` with a session


# ------------------------------------------------------------------ a narrow host per adapter --

_DAY = 86_400_000_000_000
_T0 = 1_750_000_000_000_000_000


def _bar(sym: str, i: int, close: float, high: float | None = None) -> Bar:
    """A daily bar with a 2.0 range, so the 14-bar ATR is ~2.0 — a known unit to size 4.5 ATR by."""
    bt = BarType.from_str(f"{sym}.XNAS-1-DAY-LAST-EXTERNAL")
    hi = close + 1.0 if high is None else high
    return Bar(bar_type=bt, open=Price.from_str(f"{close - 0.5:.2f}"),
               high=Price.from_str(f"{hi:.2f}"), low=Price.from_str(f"{close - 1.0:.2f}"),
               close=Price.from_str(f"{close:.2f}"), volume=Quantity.from_int(1_000),
               ts_event=_T0 + i * _DAY, ts_init=_T0 + i * _DAY)


def _hist_row(i: int, close: float, high: float | None = None):
    """The intraday adapter keeps `_hist` rows with `.date`/`.close`, not Bars."""
    return SimpleNamespace(date=pd.Timestamp(_T0 + i * _DAY, unit="ns"), close=close,
                           high=close + 1.0 if high is None else high, low=close - 1.0,
                           open=close - 0.5)


def _position(sym: str, entry: float, opened_i: int):
    return SimpleNamespace(instrument_id=SimpleNamespace(symbol=SimpleNamespace(value=sym)),
                           avg_px_open=entry, ts_opened=_T0 + opened_i * _DAY, signed_qty=10.0)


def _bar_row(sym: str, i: int, close: float, high: float | None = None) -> dict:
    """qc345 keeps `_bars` as dict rows (its `_ingest` shape), not Bars."""
    return {"ticker": sym, "date": pd.Timestamp(_T0 + i * _DAY, unit="ns").normalize(),
            "open": close - 0.5, "high": close + 1.0 if high is None else high,
            "low": close - 1.0, "close": close, "volume": 1_000.0}


def _host(cls, *, entry: float, closes: list[float], take_profit_atr: float | None = None,
          exits: ExitConfig | None = None, symbols: tuple[str, ...] = ("X",),
          history_for: tuple[str, ...] | None = None, last_high: float | None = None):
    """The adapter's REAL `_trail_exits` (and `_panel` where it has one) bound to a host holding one
    position per symbol opened at bar 0 and the bars since. `history_for` limits which symbols carry
    bars (per-symbol independence); `exits` overrides the ExitConfig outright."""
    exit_cfg = exits if exits is not None else ExitConfig(take_profit_atr=take_profit_atr)

    class Host:
        pass

    Host._trail_exits = cls._trail_exits
    if "_panel" in vars(cls):
        Host._panel = cls._panel
    h = Host()
    h.id = "TEST-000"
    if cls.__name__ == "QC345RotationStrategy":
        h._cfg = QC345RotationConfig(exits=exit_cfg)
    else:
        h._cfg = MomentumRotationConfig(exits=exit_cfg)
    h._held = set(symbols)
    h.cache = SimpleNamespace(
        positions_open=lambda **kw: [_position(sym, entry, 0) for sym in symbols])
    h.log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                            error=lambda *a, **k: None)
    with_bars = symbols if history_for is None else history_for
    last = len(closes) - 1

    def hi(i):
        return last_high if (i == last and last_high is not None) else None
    if cls.__name__ == "IntradayMomentumRotation":
        h._hist = {sym: [_hist_row(i, c, hi(i)) for i, c in enumerate(closes)] for sym in with_bars}
    elif cls.__name__ == "QC345RotationStrategy":
        h._bars = {sym: [_bar_row(sym, i, c, hi(i)) for i, c in enumerate(closes)] for sym in with_bars}
    else:
        h._bars = {sym: [_bar(sym, i, c, hi(i)) for i, c in enumerate(closes)] for sym in with_bars}
    return h


def _drive(cls, h) -> set[str]:
    """`_trail_exits()` for the no-arg adapters; `_trail_exits(session)` for qc345, with the session
    AFTER the last bar so every bar is `prior`."""
    if cls.__name__ == "QC345RotationStrategy":
        return cls._trail_exits(h, pd.Timestamp(_T0 + len(_CLOSES) * _DAY, unit="ns").normalize())
    return cls._trail_exits(h)


#: 20 bars, entry 100, a steady climb to 112: 12 points of gain against an ATR of ~2.0 = ~6 ATR,
#: comfortably past a 4.5-ATR target and comfortably short of a 6.5 one.
_CLOSES = [100.0 + 0.6 * i for i in range(21)]


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda c: c.__name__)
def test_with_take_profit_set_the_adapter_path_does_NOT_raise_and_the_target_EXITS(cls):
    """The defect and the fix in one drive: on 1f5fe2e momentum/intraday raise ValueError at
    exits.py:190 the moment `take_profit_atr` is set; after, the position 6 ATR in profit exits."""
    h = _host(cls, entry=100.0, closes=_CLOSES, take_profit_atr=4.5)
    try:
        exits = _drive(cls, h)
    except ValueError as exc:
        pytest.fail(f"{cls.__name__}._trail_exits raised on an ATR-scaled rule it was configured "
                    f"with — the adapter path never passes `atr` (#222): {exc}")
    assert "X" in exits, f"{cls.__name__}: 6 ATR in profit against a 4.5-ATR target did not exit"


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda c: c.__name__)
def test_a_target_ABOVE_the_gain_does_not_exit_so_the_atr_is_REAL_not_a_placeholder(cls):
    """A fix that passed `atr={sym: 1e-9}` (or any tiny constant) would make every target trivially
    reached. 6 ATR of gain must NOT clear a 9-ATR target."""
    h = _host(cls, entry=100.0, closes=_CLOSES, take_profit_atr=9.0)
    assert "X" not in _drive(cls, h), f"{cls.__name__}: the ATR passed is not the bars' ATR"


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda c: c.__name__)
def test_with_NO_atr_rule_the_path_is_byte_identical_to_before(cls):
    """`atr=None` when nothing needs it — the pre-#222 call, so the deployed lanes' give-back path is
    untouched by this change."""
    h = _host(cls, entry=100.0, closes=_CLOSES, take_profit_atr=None)
    assert _drive(cls, h) == set()


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda c: c.__name__)
def test_a_symbol_WITHOUT_history_does_not_suppress_the_OTHER_symbols_exit(cls):
    """Per-symbol independence (review mutant (a)): Y holds a position but carries no bars, so it
    has no ATR and cannot be judged; X, 6 ATR in profit, must still exit at 4.5."""
    h = _host(cls, entry=100.0, closes=_CLOSES, take_profit_atr=4.5, symbols=("X", "Y"),
              history_for=("X",))
    exits = _drive(cls, h)
    assert "X" in exits and "Y" not in exits, exits


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda c: c.__name__)
def test_a_HIGHS_only_rule_with_no_atr_rule_is_fed_its_highs(cls):
    """Review mutant (c): the frame/highs must be built under `needs_highs` alone, not only under
    `needs_atr`. `peak_fade_off_pct` needs highs and no ATR; with highs missing `evaluate_exits`
    cannot judge the fade — so a config with ONLY that rule must not raise and must still see the
    last high (a fade on the final bar → exit)."""
    cfg = ExitConfig(peak_fade_off_pct=0.05, peak_fade_confirm=1)
    # Closes climb to 112, the last close is 107.5 — 4.0% off the close peak — but the last bar's
    # HIGH is 115: 6.5% off the true peak. With highs the 5% fade fires; without them (`hi = px`)
    # the rule measures 4.0% and holds. That is precisely the silent under-firing `needs_highs`'s
    # docstring records, and the highs-only gate must feed it.
    closes = _CLOSES[:-1] + [107.5]
    h = _host(cls, entry=100.0, closes=closes, exits=cfg, last_high=115.0)
    try:
        exits = _drive(cls, h)
    except ValueError as exc:
        pytest.fail(f"{cls.__name__}: a highs-only rule raised — highs are not built without an ATR rule: {exc}")
    assert "X" in exits, f"{cls.__name__}: the fade measured against closes, not highs — highs were not fed"


def test_every_adapter_path_passes_atr_AND_highs_by_keyword_and_asks_needs_atr_first():
    """Structural, AST-bound (a substring "atr=" would be satisfied by a comment): every
    `_trail_exits` calls `evaluate_exits(..., atr=..., highs=...)` and consults `needs_atr` — so the
    no-rule path stays byte-identical (the "atr always computed" mutant is otherwise invisible to
    the driven tests: same output, extra work). QC345 (session-taking, frame-based) is covered here
    too. Kept beside the driven tests, never instead of them."""
    import ast
    import textwrap

    for cls in ADAPTERS:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls._trail_exits)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", "")) == "evaluate_exits"]
        assert calls, f"{cls.__name__}._trail_exits does not call evaluate_exits"
        kws = {k.arg for k in calls[0].keywords}
        assert {"atr", "highs"} <= kws, f"{cls.__name__}: evaluate_exits called without {sorted({'atr', 'highs'} - kws)}"
        asks = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "id", getattr(n.func, "attr", "")) == "needs_atr"]
        assert asks, f"{cls.__name__}: computes (or skips) atr without asking needs_atr"
