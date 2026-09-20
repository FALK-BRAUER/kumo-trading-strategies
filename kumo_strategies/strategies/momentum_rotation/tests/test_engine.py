"""Deterministic tests for the momentum-rotation decision engine.

These pin the behaviours whose absence produced a false result during research (issue #13, #15).
Each is a bug that actually happened, not a hypothetical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kumo_strategies.strategies.momentum_rotation import (
    MomentumRotationConfig, apply_gates, decide, score_panel,
)
from kumo_strategies.strategies.momentum_rotation.config import GateConfig, PortfolioConfig
from kumo_strategies.strategies.momentum_rotation.candidates import StaticList, build, registered
from kumo_strategies.strategies.momentum_rotation.score import vol_normalised_momentum


def series(prices, vols=None, ticker="AAA", start="2026-01-01", band=0.02):
    """A price series with a realistic intraday range.

    `band` matters: with high == low == close every bar has zero true range, so ATR is 0 and any
    ATR-based gate blocks the entire fixture. That is why min_atr_pct went unenforced for so long —
    it could not have been tested against these bars. Default band puts ATR ~2% of price, above the
    1% gate, so a test only trips the gate it is actually about. Pass band=0 to test the gate.
    """
    n = len(prices)
    px = np.asarray(prices, dtype=float)
    return pd.DataFrame({
        "ticker": ticker,
        "date": pd.bdate_range(start, periods=n),
        "open": px, "high": px * (1 + band), "low": px * (1 - band), "close": px,
        "volume": vols if vols is not None else [1_000_000] * n,
    })


# --- gates ---------------------------------------------------------------------------------

def test_split_is_blocked_not_traded():
    """A 5:1 split reads as -80% in raw unadjusted data. NOW did exactly this and it dominated a
    backtest until gated."""
    px = [100.0] * 30 + [20.0] * 30          # the split
    d = apply_gates(series(px), MomentumRotationConfig())
    assert d.loc[d.date == d.date.iloc[30], "blocked"].all()
    # and the window extends around it, because a hold spanning the action is equally corrupted
    assert d["blocked"].sum() > 1


def test_reverse_split_is_blocked_too():
    """RKLZ went +685% on an 8:1 reverse split and contributed more profit than the whole strategy."""
    px = [3.0] * 30 + [24.0] * 30
    d = apply_gates(series(px), MomentumRotationConfig())
    assert d["blocked"].any()


def test_leveraged_products_excluded_by_instrument_type():
    """HIMZ and GDXD are leveraged ETNs that reverse-split constantly; they produced ~$193k of a
    $194k 'profit' before exclusion."""
    df = pd.concat([series([10.0] * 40, ticker="GOOD"), series([10.0] * 40, ticker="GDXD")])
    d = apply_gates(df, MomentumRotationConfig(), {"GDXD": "etf_or_product", "GOOD": "profiled_equity"})
    assert d.loc[d.ticker == "GDXD", "blocked"].all()
    assert not d.loc[d.ticker == "GOOD", "blocked"].any()


def test_flat_names_blocked_by_atr_gate():
    """min_atr_pct was declared in GateConfig and never applied in apply_gates — the gate existed
    everywhere except in the code path. A near-zero-ATR name (JPST et al) gets an enormous share
    count under risk-based sizing, so one tick becomes multiple R."""
    flat = series([50.0] * 40, band=0.0)                     # no intraday range at all
    blocked = apply_gates(flat, MomentumRotationConfig())["blocked"]
    # ATR needs 10 bars before it exists; those rows carry no estimate so the gate cannot speak.
    # They are unscoreable anyway (min_history=25), so the claim is about every bar that HAS an ATR.
    assert blocked.iloc[10:].all()

    moves = series([50.0] * 40, band=0.02)                   # 2% range, above the 1% gate
    assert not apply_gates(moves, MomentumRotationConfig())["blocked"].any()


def test_atr_gate_can_be_disabled():
    flat = series([50.0] * 40, band=0.0)
    cfg = MomentumRotationConfig(gates=GateConfig(min_atr_pct=0.0))
    assert not apply_gates(flat, cfg)["blocked"].any()


def test_illiquid_names_blocked():
    df = series([1.0] * 40, vols=[100] * 40)
    assert apply_gates(df, MomentumRotationConfig())["blocked"].all()


# --- scoring -------------------------------------------------------------------------------

def test_momentum_is_volatility_normalised():
    """A quiet name up 10% must outrank a wild name up 10% — that is the whole point of dividing by
    the name's own SD. A fixed percentage threshold cannot express this."""
    rng = np.random.default_rng(0)
    quiet = 100 * np.cumprod(1 + rng.normal(0.001, 0.002, 90))
    wild = 100 * np.cumprod(1 + rng.normal(0.001, 0.030, 90))
    q = vol_normalised_momentum(pd.Series(quiet), 20, 60, 25).iloc[-1]
    w = vol_normalised_momentum(pd.Series(wild), 20, 60, 25).iloc[-1]
    assert np.isfinite(q) and np.isfinite(w)
    assert abs(q) > abs(w)      # same drift, far less noise -> stronger signal


def test_score_has_no_ties():
    """A rolling percentile saturates at 1.0 and ties break by frame order — that once made 77% of
    picks A/B/C tickers against a 23% universe share. A ratio-based score must not tie."""
    rng = np.random.default_rng(1)
    vals = [vol_normalised_momentum(
        pd.Series(100 * np.cumprod(1 + rng.normal(0.002, 0.02, 90))), 20, 60, 25).iloc[-1]
        for _ in range(50)]
    assert len(set(vals)) == len(vals)


def test_blocked_bars_get_no_score():
    px = [100.0] * 30 + [20.0] * 40
    d = score_panel(apply_gates(series(px), MomentumRotationConfig()), MomentumRotationConfig())
    assert d.loc[d["blocked"], "score"].isna().all()


# --- decision ------------------------------------------------------------------------------

def _panel(scores: dict, d="2026-03-02"):
    return pd.DataFrame({"ticker": list(scores), "date": pd.Timestamp(d),
                         "score": list(scores.values()), "blocked": False})


def test_holds_the_top_n():
    cfg = MomentumRotationConfig()
    dec = decide(_panel({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0, "F": 0.5}),
                 StaticList(list("ABCDEF")), cfg, held=set())
    assert len(dec.hold) == cfg.portfolio.n_hold
    assert set(dec.hold) == set("ABCDE")


def test_buffer_prevents_thrashing():
    """A held name just outside the top N is NOT sold — only once it leaves top (n_hold + buffer).
    Without this the book churns on rank noise and turnover doubles for nothing."""
    cfg = MomentumRotationConfig()
    scores = {t: float(20 - i) for i, t in enumerate("ABCDEFGHIJ")}
    dec = decide(_panel(scores), StaticList(list(scores)), cfg, held={"F"})   # F is rank 6 of 10
    assert "F" not in dec.exit
    dec2 = decide(_panel(scores), StaticList(list(scores)), cfg, held={"J"})  # J is rank 10
    assert "J" in dec2.exit


def test_only_candidates_are_considered():
    """The pool IS the edge — a name outside the source must never be bought however strong."""
    dec = decide(_panel({"IN": 1.0, "OUT": 99.0}), StaticList(["IN"]), MomentumRotationConfig(),
                 held=set())
    assert "OUT" not in dec.hold and dec.hold == ("IN",)


def test_unscored_names_are_not_bought():
    p = _panel({"A": 5.0, "B": np.nan})
    dec = decide(p, StaticList(["A", "B"]), MomentumRotationConfig(), held=set())
    assert "B" not in dec.hold


# --- registry ------------------------------------------------------------------------------

def test_candidate_sources_are_pluggable():
    assert {"bct_positions", "static_list", "industry_members", "union"} <= set(registered())
    s = build("static_list", symbols=["X", "Y"])
    assert s.eligible(pd.Timestamp("2026-01-01")) == {"X", "Y"}


def test_unknown_source_fails_loudly():
    with pytest.raises(ValueError, match="unknown candidate source"):
        build("does_not_exist")


# --- staleness ------------------------------------------------------------------------------

def test_unsold_positions_expire_out_of_the_pool(tmp_path):
    """Sells go unrecorded, so an unsold row would sit in the book forever. That inflated the pool
    to 121 names against a real book of ~40, and the stale tail is biased toward winners because
    those are the ones that quietly persist."""
    from kumo_strategies.strategies.momentum_rotation.candidates import BctPositions

    csv = tmp_path / "ledger.csv"
    csv.write_text("symbol,buy_date,sell_date\n"
                   "FRESH,2026-03-01,\n"
                   "STALE,2025-01-01,\n"
                   "SOLD,2026-01-01,2026-02-01\n")
    src = BctPositions(str(csv), lag_days=1, max_open_days=58)
    elig = src.eligible(pd.Timestamp("2026-03-20"))
    assert "FRESH" in elig          # bought 19 days ago, plausibly still held
    assert "STALE" not in elig      # unsold for over a year — a missed sell, not a hold
    assert "SOLD" not in elig       # explicitly closed


def test_staleness_cutoff_can_be_disabled(tmp_path):
    from kumo_strategies.strategies.momentum_rotation.candidates import BctPositions

    csv = tmp_path / "l.csv"
    csv.write_text("symbol,buy_date,sell_date\nSTALE,2025-01-01,\n")
    assert "STALE" in BctPositions(str(csv), max_open_days=None).eligible(pd.Timestamp("2026-03-20"))


# --- ledger-tool-owned book --------------------------------------------------------------------

def test_ledger_tool_book_parses_closes_and_replays_deltas(tmp_path):
    """ledger-tool owns the ledger. Its file marks closes with strikethrough and carries dated
    ADDED/CLOSED deltas between weekend re-seeds."""
    from kumo_strategies.strategies.momentum_rotation.candidates import LedgerToolHoldings

    md = tmp_path / "holdings.md"
    md.write_text(
        "| AAA | Jul10 | held |\n"
        "| ~~BBB~~ | CLOSED Jul20 +5% | gone |\n"
        "  2026-07-25 (Fri): ADDED CCC; CLOSED AAA 3.1%\n"
    )
    s = LedgerToolHoldings(str(md))
    assert s.closed["BBB"] == pd.Timestamp("2026-07-20")
    assert "AAA" in s.eligible(pd.Timestamp("2026-07-15"))     # before its close
    assert "BBB" not in s.eligible(pd.Timestamp("2026-07-25"))  # struck through
    assert "CCC" in s.eligible(pd.Timestamp("2026-07-26"))      # added by delta
    assert "AAA" not in s.eligible(pd.Timestamp("2026-07-26"))  # closed by delta


# --- ledger-tool's machine-readable book ---------------------------------------------------------

def _book(tmp_path, rows: str):
    p = tmp_path / "ledger-book.csv"
    p.write_text("symbol,open_date,close_date,close_pct,still_open,close_source\n" + rows)
    return str(p)


def test_ledger_book_respects_open_and_close_dates(tmp_path):
    from kumo_strategies.strategies.momentum_rotation.candidates import LedgerBook

    csv = _book(tmp_path, "AAA,2026-03-01,2026-03-20,5.1,False,post\n"
                          "BBB,2026-03-05,,,True,\n")
    src = LedgerBook(csv, lag_days=1, max_open_days=None)
    assert src.eligible(pd.Timestamp("2026-03-10")) == {"AAA", "BBB"}
    assert src.eligible(pd.Timestamp("2026-03-25")) == {"BBB"}      # AAA closed


def test_ledger_book_refuses_dates_it_cannot_support(tmp_path):
    """ledger-tool measured that Oct 2025 - Feb 2026 has 20 posts in five months, verified as genuine
    absence. Returning a book there would invent membership, so it returns nothing instead."""
    from kumo_strategies.strategies.momentum_rotation.candidates import LedgerBook

    csv = _book(tmp_path, "AAA,2025-11-01,,,True,\n")
    src = LedgerBook(csv, coverage_start="2026-03-01")
    assert src.eligible(pd.Timestamp("2025-12-01")) == set()
    assert src.eligible(pd.Timestamp("2026-03-15")) == {"AAA"}


def test_expiry_is_calibrated_against_a_snapshot_not_the_hold_distribution(tmp_path):
    """The hold distribution is drawn from the same biased data the expiry corrects — ledger-tool's p95
    is 118d against mine of 58d for exactly that reason. Calibrate against an observed video count."""
    import json

    from kumo_strategies.strategies.momentum_rotation.candidates import LedgerBook

    csv = _book(tmp_path, "".join(
        f"S{i},2026-0{1 if i < 5 else 6}-01,,,True,\n" for i in range(10)))
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({"snapshots": {"2026-07-10": {"count": 5, "held": []}}}))
    days = LedgerBook.calibrate_expiry(csv, str(snap), anchors="2026-07-10")
    got = len(LedgerBook(csv, max_open_days=days, coverage_start=None).eligible(pd.Timestamp("2026-07-10")))
    assert got == 5


def test_expiry_fits_across_multiple_anchors(tmp_path):
    """Fit only against COMPLETE reviews. Counts are lower bounds — a thin review means he annotated
    fewer names that week, not that the book shrank. ledger-tool's 2026-05-10 shows 6 and 2026-07-25
    shows 8; fitting to those would drive the expiry far too short."""
    import json

    from kumo_strategies.strategies.momentum_rotation.candidates import LedgerBook

    csv = _book(tmp_path, "".join(f"S{i},2026-03-0{i%9+1},,,True,\n" for i in range(12)))
    snap = tmp_path / "s.json"
    snap.write_text(json.dumps({"snapshots": {
        "2026-03-15": {"count": 8, "held": []},
        "2026-07-10": {"count": 8, "held": []},
    }}))
    days = LedgerBook.calibrate_expiry(csv, str(snap), anchors=["2026-03-15", "2026-07-10"])
    assert isinstance(days, int) and days > 0


def test_pre_coverage_opens_resolved_by_observed_absence(tmp_path):
    """ledger-tool's third option, better than expiring or excluding: check the next video review. If
    the name is absent he was out by then, so close it AT the snapshot date — an observation rather
    than a guess. Absence is weaker evidence than presence, so this only fires when a snapshot
    exists after the open."""
    import json

    from kumo_strategies.strategies.momentum_rotation.candidates import LedgerBook

    csv = _book(tmp_path, "GONE,2026-01-05,,,True,\n"
                          "KEPT,2026-01-05,,,True,\n"
                          "LATER,2026-06-01,,,True,\n")
    snap = tmp_path / "s.json"
    snap.write_text(json.dumps({"snapshots": {"2026-02-01": {"count": 1, "held": ["KEPT"]}}}))

    src = LedgerBook(csv, max_open_days=None, coverage_start=None)
    assert src.validate_pre_coverage(str(snap)) == 1        # only GONE resolves
    assert "GONE" not in src.eligible(pd.Timestamp("2026-02-10"))
    assert "KEPT" in src.eligible(pd.Timestamp("2026-02-10"))
    # LATER opens after the only snapshot, so there is nothing to validate against — left alone
    assert "LATER" in src.eligible(pd.Timestamp("2026-06-10"))


def test_coverage_start_is_december_not_march(tmp_path):
    """ledger-tool corrected themselves: Oct 2025 - Feb 2026 was a shallow fetch, not a silent author.
    He posts every trading day, and 1-2 posts a month is a puller that stopped early. A deeper pull
    took coverage from 55 to 201 date dirs and Dec/Jan/Feb are now ~20 posts each. Oct/Nov stay
    excluded — they recovered only partially against a YouTube continuation cap of 244 posts."""
    from kumo_strategies.strategies.momentum_rotation.candidates import LedgerBook

    assert LedgerBook.COVERAGE_START == pd.Timestamp("2025-12-01")
    csv = _book(tmp_path, "AAA,2025-11-01,,,True,\n")
    src = LedgerBook(csv)
    assert src.eligible(pd.Timestamp("2025-11-15")) == set()      # below the cutoff
    assert src.eligible(pd.Timestamp("2025-12-15")) == {"AAA"}


def test_a_held_name_that_leaves_the_pool_IS_sold():
    """Following the trader's exits is the single most valuable rule here — worth 29 points of
    return (+62.5% vs +33.4% over 167 sessions). The pool drops a name when he closes it, so a
    pool departure is his sell signal.

    Protection against a bad feed is UPSTREAM (stale gate, shrink guard), not here — an earlier
    attempt to protect the book inside decide() by never selling an unranked holding also stopped
    following his genuine exits, and cost those 29 points.
    """
    df = pd.concat([series([10.0 * (1 + 0.004 * (i + 1) * d) for d in range(60)], ticker=t)
                    for i, t in enumerate(["AAA", "BBB", "CCC", "DDD"])])
    panel = score_panel(apply_gates(df, MomentumRotationConfig()), MomentumRotationConfig())
    day = panel[panel.date == panel.date.max()]
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=2, buffer=0))
    d = decide(day, StaticList(["AAA", "BBB", "CCC"]), cfg, held={"AAA", "GONE"})
    assert "GONE" in d.exit, "a name the source dropped is his exit signal — follow it"


def test_a_held_name_still_in_the_pool_is_exited_when_it_falls_out_of_rank():
    df = pd.concat([series([10.0 * (1 + 0.004 * (i + 1) * d) for d in range(60)], ticker=t)
                    for i, t in enumerate(["AAA", "BBB", "CCC", "DDD"])])
    panel = score_panel(apply_gates(df, MomentumRotationConfig()), MomentumRotationConfig())
    day = panel[panel.date == panel.date.max()]
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=1, buffer=0))
    ranked = day.dropna(subset=["score"]).nlargest(4, "score").ticker.tolist()
    assert len(ranked) >= 2, "fixture must produce a real ranking or the test proves nothing"
    worst = ranked[-1]
    d = decide(day, StaticList(ranked), cfg, held={worst})
    assert worst in d.exit


# --- boundaries ----------------------------------------------------------------------------
# The 26 tests above pin behaviours that produced a false result during research. None of them
# feeds the engine a degenerate frame, and the money path has to survive one: a feed gap, a pool
# that shrank to nothing, a universe where every name scores the same.

def test_gates_survive_an_empty_panel():
    """A source that returns nothing, or a date filter that matches nothing."""
    empty = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    out = apply_gates(empty, MomentumRotationConfig(), {})
    assert len(out) == 0


def test_scoring_survives_an_empty_panel():
    empty = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    out = score_panel(apply_gates(empty, MomentumRotationConfig(), {}), MomentumRotationConfig())
    assert len(out) == 0


def test_a_single_bar_scores_nan_rather_than_raising():
    """One bar cannot support a 20-day lookback. It must produce no score, not an exception and
    not a fabricated one."""
    out = score_panel(apply_gates(series([100.0]), MomentumRotationConfig(), {}),
                      MomentumRotationConfig())
    assert len(out) == 1
    assert pd.isna(out["score"].iloc[0])


def test_a_universe_of_one_does_not_crash_decide():
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=8, buffer=5))
    panel = score_panel(apply_gates(series(list(range(100, 160))), cfg, {}), cfg)
    day = panel[panel.date == panel.date.max()]
    d = decide(day, StaticList(["AAA"]), cfg, set())
    assert set(d.enter) <= {"AAA"}


def test_identical_scores_do_not_break_ranking():
    """Every name moving identically is degenerate but not impossible — an index-tracking day, or
    a synthetic feed. Ranking must still return n_hold names and must not raise."""
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=3, buffer=0))
    prices = [100 * 1.01 ** i for i in range(60)]
    panel = pd.concat([series(prices, ticker=t) for t in ("AAA", "BBB", "CCC", "DDD", "EEE")],
                      ignore_index=True)
    scored = score_panel(apply_gates(panel, cfg, {}), cfg)
    day = scored[scored.date == scored.date.max()]
    d = decide(day, StaticList(["AAA", "BBB", "CCC", "DDD", "EEE"]), cfg, set())
    assert len(set(d.enter) | set(d.hold)) <= cfg.portfolio.n_hold + cfg.portfolio.buffer
    assert d.enter, "ranked nothing — the source must name the panel's actual tickers"


def test_all_nan_scores_enter_nothing():
    """A gate blanking the score column, or a panel too short to score. `decide` must not treat an
    unrankable universe as a decision to hold nothing — it must simply enter nothing."""
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=3, buffer=0))
    panel = pd.concat([series([100.0, 101.0], ticker=t) for t in ("AAA", "BBB")],
                      ignore_index=True)
    scored = score_panel(apply_gates(panel, cfg, {}), cfg)
    day = scored[scored.date == scored.date.max()]
    assert day["score"].isna().all(), "premise: two bars cannot score a 20-day lookback"
    d = decide(day, StaticList(["AAA", "BBB"]), cfg, set())
    assert d.enter == (), f"entered on an all-NaN ranking: {d.enter}"


def test_one_symbols_history_does_not_leak_into_another():
    """The panel is long-format and every rolling calc must be grouped by ticker. A leak here is
    invisible in the output — the numbers look plausible, they are just another name's."""
    cfg = MomentumRotationConfig()
    rising = series([100 + i for i in range(60)], ticker="RISE")
    falling = series([160 - i for i in range(60)], ticker="FALL")
    together = score_panel(apply_gates(pd.concat([rising, falling], ignore_index=True), cfg, {}), cfg)
    alone = score_panel(apply_gates(rising, cfg, {}), cfg)
    a = together[(together.ticker == "RISE") & (together.date == together.date.max())]["score"].iloc[0]
    b = alone[alone.date == alone.date.max()]["score"].iloc[0]
    assert np.isclose(a, b, equal_nan=True), f"RISE scored {a} beside FALL, {b} alone"


def test_symbol_order_in_the_frame_does_not_change_the_decision():
    """Row order is an accident of the source. If it changes what we buy, ties are being broken by
    position — which is how 77% of picks once came from A/B/C tickers."""
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=2, buffer=0))
    frames = [series([100 + i * k for i in range(60)], ticker=t)
              for k, t in ((3, "AAA"), (1, "BBB"), (2, "CCC"))]
    fwd = score_panel(apply_gates(pd.concat(frames, ignore_index=True), cfg, {}), cfg)
    rev = score_panel(apply_gates(pd.concat(frames[::-1], ignore_index=True), cfg, {}), cfg)
    src = StaticList(["AAA", "BBB", "CCC"])
    a = decide(fwd[fwd.date == fwd.date.max()], src, cfg, set())
    b = decide(rev[rev.date == rev.date.max()], src, cfg, set())
    assert set(a.enter) == set(b.enter), f"row order changed the buy list: {a.enter} vs {b.enter}"


def test_equal_scores_break_ties_deterministically():
    """Four names scoring identically entered ('AAA','BBB') read forwards and ('DDD','CCC') read
    backwards. Row order is an accident of the source; it must not choose what we buy.

    The earlier order test only used DISTINCT scores, so it never exercised a tie — and the
    identical-score test passed a source naming tickers the panel did not contain, so it ranked
    nothing at all. Both were green while the bug was live."""
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=2, buffer=0))
    day = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "DDD"], "score": [1.0] * 4,
                        "date": [pd.Timestamp("2026-08-04")] * 4, "close": [10.0] * 4})
    src = StaticList(["AAA", "BBB", "CCC", "DDD"])
    fwd = decide(day, src, cfg, set()).enter
    rev = decide(day.iloc[::-1].reset_index(drop=True), src, cfg, set()).enter
    assert fwd == rev, f"tie broken by row order: {fwd} forwards, {rev} backwards"


def test_a_duplicated_ticker_row_does_not_take_two_slots():
    """A feed replaying a bar, or two sources merged without dedupe. With n_hold=2 and rows
    AAA/AAA/BBB the ranking returned enter=('AAA','AAA'): BBB never got its slot, the recorded
    score was the loser's, and the runtime would submit two BUYs for one symbol in one session."""
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=2, buffer=0))
    dup = pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"], "score": [10.0, 9.0, 8.0],
                        "date": [pd.Timestamp("2026-08-04")] * 3, "close": [10.0] * 3})
    d = decide(dup, StaticList(["AAA", "BBB"]), cfg, set())
    assert len(set(d.enter)) == len(d.enter), f"duplicate entry: {d.enter}"
    assert "BBB" in d.enter, f"the duplicate stole the second slot: {d.enter}"


def test_an_unrankable_day_holds_the_book_instead_of_selling_it():
    """If nothing scores — every candidate NaN after gates, a panel too short, a feed gap — then
    `keep` is empty and every held name looks like it fell out. The Pg runner guards this, but
    backtest and the local Nautilus path call decide() directly and had no such guard: a
    data-collapsed day liquidated the book and read as an ordinary rotation."""
    cfg = MomentumRotationConfig(portfolio=PortfolioConfig(n_hold=3, buffer=0))
    day = pd.DataFrame({"ticker": ["AAA", "BBB"], "score": [np.nan, np.nan],
                        "date": [pd.Timestamp("2026-08-04")] * 2, "close": [10.0, 10.0]})
    d = decide(day, StaticList(["AAA", "BBB"]), cfg, {"AAA"})
    assert d.exit == (), f"sold the book on an unrankable day: {d.exit}"
    assert "AAA" in d.hold


def test_the_liquidity_gate_cannot_see_future_volume():
    """The median was taken over every row supplied. In a backtest that is the whole history, so a
    name that became liquid later cleared the gate on dates when its volume was below the floor.
    Live never had it — the runner passes trailing bars only — so this is what makes the two agree
    rather than a new rule."""
    cfg = MomentumRotationConfig()
    n = 50
    px = pd.DataFrame({"ticker": "AAA", "date": pd.bdate_range("2026-01-01", periods=2 * n),
                       "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0,
                       "volume": [1e5] * n + [1e6] * n})     # $1M/day then $10M/day, floor $5M
    alone = apply_gates(px.head(n).copy(), cfg, {})
    with_future = apply_gates(px.copy(), cfg, {})
    assert bool(alone["blocked"].iloc[0]), "premise: $1M/day is below the $5M floor"
    assert bool(with_future["blocked"].iloc[0]), \
        "appending future high-volume bars unblocked a past date"
