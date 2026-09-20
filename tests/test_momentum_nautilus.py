

def test_a_watched_symbol_with_no_bars_is_named():
    """FTNR resolved to an instrument, was counted in "watching 71 instruments", then received zero
    bars for a whole session — Alpaca has no such asset. Nothing logged it, so a pool symbol that
    can never be traded looked identical to one that simply did not rank."""
    from types import SimpleNamespace

    from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy

    warned: list[str] = []
    # `log` is a read-only Cython attribute on Actor, so the method is called unbound against a
    # duck-typed self rather than stubbed onto an instance.
    strat = SimpleNamespace(
        # Log lines name the real StrategyId now rather than a module constant. BCTROT reported
        # itself as "MOMENTUM-002 watching 95 instruments" on startup because of that constant,
        # which cost a double-take during a live check and would cost more during an incident.
        id="MOMENTUM-002",
        _iids=[SimpleNamespace(symbol=SimpleNamespace(value=s)) for s in ("AFL", "FTNR", "MET")],
        _bars={"AFL": [object()] * 120, "MET": [object()] * 3},
        _need=101,
        log=SimpleNamespace(warning=lambda m: warned.append(m), info=lambda m: None),
    )

    MomentumRotationStrategy._report_dataless(strat)

    assert len(warned) == 1
    assert "FTNR" in warned[0], "the symbol with no bars at all must be named"
    assert "MET" not in warned[0], "a merely-thin symbol is not the same failure"


def _bar(sym, day_ns, close):
    from types import SimpleNamespace
    return SimpleNamespace(
        bar_type=SimpleNamespace(instrument_id=SimpleNamespace(symbol=SimpleNamespace(value=sym))),
        ts_event=day_ns, close=close)


def test_republished_daily_bar_replaces_rather_than_appends():
    """Alpaca republishes the CURRENT session's daily bar as it updates.

    Appending each arrival put ~103 copies of one date in the panel after thirteen hours of uptime.
    `day = panel[date == date.max()]` then selected all of them — 7289 rows across 71 tickers — and
    the trailing 20-bar window for the newest rows was nothing but repeats of a single price: zero
    variance, NaN score. The 2026-08-05 session found 0/7289 rankable and refused to decide.

    It passed on day one only because that engine had restarted eight minutes earlier, so this bug
    is invisible to any test that does not let bars accumulate within one session.
    """
    import collections

    import pandas as pd

    from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy

    strat = type("S", (), {})()
    strat._bars = collections.defaultdict(list)

    d1 = pd.Timestamp("2026-08-04 20:00", tz="UTC").value
    d2 = pd.Timestamp("2026-08-05 13:40", tz="UTC").value
    d2_later = pd.Timestamp("2026-08-05 19:55", tz="UTC").value  # same session, updated

    for b in (_bar("AFL", d1, 100.0), _bar("AFL", d2, 101.0), _bar("AFL", d2_later, 102.5)):
        MomentumRotationStrategy._ingest(strat, b)

    bars = strat._bars["AFL"]
    assert len(bars) == 2, "one bar per session — the republished 08-05 bar must not add a row"
    assert bars[-1].close == 102.5, "the LATEST version of the session's bar must win"
    assert bars[0].close == 100.0, "the prior session must be untouched"
