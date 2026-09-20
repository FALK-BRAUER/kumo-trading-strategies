"""The decision engine — pure. Given prices, a candidate source and a config, emit target holdings.

No Nautilus, no broker, no I/O. That boundary is the point: the same function decides in a backtest
and in production, so there is nothing to reconcile between them. The runtime adapter turns these
decisions into orders; it does not make any.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .candidates import CandidateSource
from .config import MomentumRotationConfig
from .score import vol_normalised_momentum


@dataclass(frozen=True)
class Decision:
    """What to hold at the close of `date`, to be filled at the next open."""

    date: pd.Timestamp
    hold: tuple[str, ...]
    enter: tuple[str, ...]
    exit: tuple[str, ...]
    scores: dict
    weights: dict = field(default_factory=dict)
    """Fraction of deployed capital per held name. Equal weight unless inverse_vol_sizing."""


def gap_declines_entry(open_today: float | None, prior_close: float | None,
                       cfg: MomentumRotationConfig) -> bool:
    """Does the overnight gap dead band decline this entry?

    THE GAP IS A PROPERTY OF THE DAY, NOT OF THE SLOT. It is `today's open / yesterday's close`,
    fixed the moment the market opens and identical at 09:35, 12:00 and 15:40. A lane that decides
    three times a session must reach the same verdict at all three, because the input has not moved.

    An earlier version took the FILL price instead of the open and gated the rule to a list of
    slots. That was a fix for a defect it had itself introduced: at 09:35 the fill is close enough to
    the open that the two agree, so it looked right, while at 12:00 `fill / prior close` silently
    becomes "the day's move so far" — a different quantity that nothing has measured. The slot list
    then existed to hide the disagreement rather than to express anything real. 2026-09-04:
    *"the gap stays the same for the day. why would the slots trade differently."*

    This also matches what the research measured. `runner_verified` fills at `next_open` and compares
    against the signal session's close, so `+0.75 sharpe` is an overnight-gap result and nothing else.

    FAILS OPEN. A rule left at None, or a missing or non-positive price, ADMITS the entry — declining
    on absent data would let a feed hiccup quietly shrink the book.
    """
    if cfg.execution.min_abs_gap_pct is None:
        return False
    if open_today is None or prior_close is None:
        return False
    try:
        o, pc = float(open_today), float(prior_close)
    except (TypeError, ValueError):
        return False
    # NaN survives `> 0` as False, so this rejects it rather than trusting an `or` default: a NaN
    # gap must admit the entry, not silently decline every one of them.
    if not (o > 0) or not (pc > 0):
        return False
    return abs(o / pc - 1.0) < cfg.execution.min_abs_gap_pct


def sizing_denominator(book: int, planned: int, cfg: MomentumRotationConfig) -> int:
    """How many ways to split the deployed gross: the book that exists, or the one that was planned.

    ONE implementation for the backtest, the cadence harness and the live runner, for the same reason
    `evaluate_exits` and `panel_stats` are one. Three copies of a sizing rule is how the number that
    was measured stops being the number that trades.

    `book` is the post-decision holding count — survivors plus entries, exits already removed. NEVER
    the pre-decision book, which still contains names being sold this session and would size every
    entry too small.

    Falls back to `planned` whenever `book` is not a usable count, so a decision that somehow arrives
    empty sizes exactly as it does today rather than dividing by zero or deploying everything into
    one name.
    """
    if not cfg.portfolio.size_to_book:
        return max(planned, 1)
    if not isinstance(book, int) or book <= 0:
        return max(planned, 1)
    return book


def filter_entries_by_gap(enters: "tuple[str, ...]", opens: dict, prior_closes: dict,
                          cfg: MomentumRotationConfig
                          ) -> "tuple[tuple[str, ...], dict[str, float]]":
    """Entries surviving the gap dead band, plus the ones declined and by how much.

    RETURNS the survivors rather than mutating in place, so the live runner's use of it is a single
    assignment to `enters`. That shape is deliberate and it is what
    `test_the_live_runner_ASSIGNS_enters_from_the_gap_filter` binds: a conditional call can be
    neutered to `if False and ...` and still satisfy an "is it called" assertion, which is how a
    structural test comes to certify a rule that never fires — the #197 B12 defect one level up, in
    the test instead of the code.

    `opens` is TODAY'S OPEN per symbol, not the fill price. See `gap_declines_entry`.
    """
    if cfg.execution.min_abs_gap_pct is None:
        return enters, {}
    declined: dict[str, float] = {}
    for sym in enters:
        o, prior = opens.get(sym), prior_closes.get(sym)
        if gap_declines_entry(o, prior, cfg):
            declined[sym] = round(float(o) / float(prior) - 1.0, 5)
    if not declined:
        return enters, {}
    return tuple(x for x in enters if x not in declined), declined


def apply_gates(px: pd.DataFrame, cfg: MomentumRotationConfig,
                instrument_type: dict[str, str] | None = None) -> pd.DataFrame:
    """Mark bars that must not be traded. Runs BEFORE any scoring, because a corporate action in raw
    data is a fake move and every scorer will believe it."""
    g = px.groupby("ticker")
    out = px.copy()
    out["ret1"] = out["close"] / g["close"].shift(1) - 1.0

    # TRAILING, not centered. A centered window also blocked the days BEFORE a split — which live
    # cannot know is coming, and which are not corrupted anyway: a bar before the split has a
    # trailing lookback that does not contain the fake move. So centering bought nothing and cost
    # point-in-time correctness, letting the backtest sidestep losses live would have taken.
    #
    # The days that ARE corrupted are the split day and the lookback window after it, because their
    # trailing momentum includes the fake return. That is exactly what a trailing window blocks.
    #
    # Third instance of this class in one review pass (bar timestamps, liquidity median, this), and
    # the liquidity one alone was worth 20 points of backtest return.
    flag = (out["ret1"].abs() > cfg.gates.corporate_action_move).astype(int)
    out["blocked"] = flag.groupby(out["ticker"]).transform(
        lambda s: s.rolling(cfg.gates.corporate_action_window, min_periods=1).max()
    ).astype(bool)

    if instrument_type:
        bad = {s for s, t in instrument_type.items() if t in cfg.gates.exclude_instrument_types}
        out.loc[out.ticker.isin(bad), "blocked"] = True

    # POINT IN TIME. `transform("median")` takes the median over every row supplied, so in a
    # backtest -- where the full history is scored in one pass -- a name that became liquid later
    # cleared the gate on dates when its actual volume was below the floor. Verified: 50 bars at
    # $1M then 50 at $10M against a $5M floor unblocks the early bars the moment the later ones are
    # appended. Live never had it, because the runner only ever passes trailing bars; this is what
    # makes backtest and live agree rather than a new rule.
    #
    # min_periods=1 so early bars are judged on what exists rather than being blocked for youth --
    # the corporate-action and history gates already handle "too new to rank".
    dv = out["close"] * out["volume"]
    mdv = (dv.groupby(out["ticker"])
             .rolling(cfg.gates.dollar_volume_window, min_periods=1).median()
             .reset_index(level=0, drop=True))
    out.loc[mdv < cfg.gates.min_dollar_volume, "blocked"] = True

    # min_atr_pct was declared in GateConfig and never enforced here — the gate existed in config,
    # in the docs and in every summary of this strategy, while doing nothing. It is the JPST case:
    # a near-zero-ATR name gets an enormous share count under risk sizing and one tick becomes
    # multiple R. Enforced now, from true range over the same window the config names.
    if cfg.gates.min_atr_pct:
        pc = g["close"].shift(1)
        tr = pd.concat([out["high"] - out["low"], (out["high"] - pc).abs(),
                        (out["low"] - pc).abs()], axis=1).max(axis=1)
        atr = tr.groupby(out["ticker"]).transform(
            lambda x: x.rolling(14, min_periods=10).mean())
        out.loc[(atr / out["close"]) < cfg.gates.min_atr_pct, "blocked"] = True
    return out


def score_panel(px: pd.DataFrame, cfg: MomentumRotationConfig) -> pd.DataFrame:
    s = cfg.score
    out = px.copy()
    out["score"] = out.groupby("ticker")["close"].transform(
        lambda c: vol_normalised_momentum(c, s.lookback, s.vol_window, s.min_history))
    out.loc[out["blocked"], "score"] = np.nan
    return out


def score_intraday(bars: pd.DataFrame, cfg: MomentumRotationConfig) -> pd.Series:
    """Latest intra-session momentum per ticker, from 5-minute bars.

    Same vol-normalised form as the daily score — the horizon changes, the reasoning does not.
    `bars` needs columns ticker/ts/close and should carry at least vol_window_bars of history.
    """
    s = cfg.intraday
    b = bars.sort_values("ts")
    sc = b.groupby("ticker")["close"].transform(
        lambda c: vol_normalised_momentum(c, s.lookback_bars, s.vol_window_bars, s.min_history))
    return b.assign(score=sc).groupby("ticker")["score"].last().dropna()


def needs_panel_stats(cfg: MomentumRotationConfig) -> bool:
    """Does this config ask for the trailing correlation/volatility estimates?

    Callers must consult this rather than deciding for themselves. `decide()` accepts `corr` and
    `vol` as optional and degrades SILENTLY without them — `_diversified` returns the plain ranking,
    `_weights` returns equal weight — so a driver that forgets them turns `max_correlation` and
    `inverse_vol_sizing` into no-ops that still typecheck, still run, and still report numbers.
    Both flags were swept that way, and one of them was inert in production. (#26)
    """
    p = cfg.portfolio
    return (p.max_correlation is not None or p.inverse_vol_sizing
            or p.cluster_exit_score is not None)


def trailing_returns(px: pd.DataFrame) -> pd.DataFrame:
    """date x ticker daily returns. Built once by callers that evaluate many sessions."""
    return (px.pivot_table(index="date", columns="ticker", values="close")
              .sort_index().pct_change())


def panel_stats(rets: pd.DataFrame, window: int, want: list[str],
                as_of: pd.Timestamp | None = None
                ) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Trailing correlation matrix and volatilities over the last `window` sessions ending `as_of`.

    ONE implementation for every driver, for the same reason `evaluate_exits` is one (#197 P2): the
    backtest and the live runner had no shared home for this, so the backtest grew one and the live
    runner grew none. A second copy is how the two disagree without anyone noticing.

    Restricted to `want` — what is held plus what could be ranked in. A 500x500 matrix per session
    over the full universe would dominate the runtime and none of it would be read.

    Returns `(None, None)` when there is not enough history to estimate anything, which the caller
    passes straight through: `decide()` then falls back to the plain ranking and equal weight, which
    is the right degradation for a genuinely unknowable input. That is NOT the same as never asking
    — see `needs_panel_stats`.
    """
    if rets is None or rets.empty:
        return None, None
    win = rets.loc[:as_of] if as_of is not None else rets
    win = win.tail(window)
    if len(win) < max(10, window // 4):
        return None, None
    sub = win[[c for c in want if c in win.columns]].dropna(axis=1, how="all")
    if sub.shape[1] < 2:
        return None, None
    return sub.corr(), sub.std()


def trailing_atr(px: pd.DataFrame, window: int = 14) -> dict[str, float]:
    """Latest average true range per ticker, in PRICE units.

    ONE implementation, for the same reason `panel_stats` and `evaluate_exits` are one: `apply_gates`
    already computes an ATR for `min_atr_pct` and throws it away, so a second copy in each driver is
    how they drift. Callers that need ATR for `give_back_min_peak_atr` take it from here.

    True range rather than high-low, so an overnight gap counts as range — a name that gaps 4% and
    then trades quietly has moved 4%, and a give-back guard measuring "one average day" must agree.
    """
    g = px.sort_values("date").groupby("ticker")
    pc = g["close"].shift(1)
    tr = pd.concat([px["high"] - px["low"], (px["high"] - pc).abs(),
                    (px["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.groupby(px["ticker"]).transform(lambda x: x.rolling(window, min_periods=max(2, window // 2)).mean())
    out = px.assign(_atr=atr).sort_values("date").groupby("ticker")["_atr"].last()
    return {k: float(v) for k, v in out.items() if pd.notna(v)}


def cluster_exits(held: set[str], ranked: pd.Series, corr: pd.DataFrame | None,
                  cfg: MomentumRotationConfig) -> set[str]:
    """Held names whose correlation cluster has, as a group, stopped working.

    A name is exited when the MEAN score of its cluster — itself plus every held name correlated
    above `cluster_corr` — falls below `cluster_exit_score`. A singleton cluster is just the name,
    so an uncorrelated holding is judged on its own score and this rule adds nothing for it.

    WEAKENING, NOT SIZE. Firing on how many names share a cluster would cap concentration, and #25
    measured concentration as the edge (Q4 volatility quartile: +90.4% of P&L). The intended failure
    this catches is the 2026-08-13 shape: four miners that had collectively lost leadership while
    each was individually still inside the buffer, exited one at a time as each dropped out.

    Returns an empty set when the config does not ask for it or when no correlation matrix was
    supplied — the latter is the #26 shape, so callers must consult `needs_panel_stats`, which
    already reports True whenever `cluster_exit_score` is set.
    """
    floor = cfg.portfolio.cluster_exit_score
    if floor is None or corr is None or not held:
        return set()
    out = set()
    for sym in held:
        if sym not in ranked.index:
            continue          # unranked names are the ranking's business, not this rule's
        peers = [p for p in held
                 if p != sym and p in corr.index and sym in corr.columns and p in ranked.index
                 and abs(float(corr.loc[p, sym])) > cfg.portfolio.cluster_corr]
        members = [sym, *peers]
        if float(ranked.reindex(members).mean()) < floor:
            out.add(sym)
    return out


def _diversified(order: list[str], held: set[str], room: int, corr: pd.DataFrame | None,
                 cap: float | None) -> list[str]:
    """Walk the ranked candidates, skipping any that duplicate a bet already in the book.

    Greedy and order-preserving: the strongest name always gets in, and a weaker one is only
    displaced if it is too close to something already held. A name with no correlation estimate is
    admitted rather than blocked — missing data must not silently shrink the book.
    """
    if corr is None or cap is None:
        return [t for t in order if t not in held][:max(0, room)]
    book, out = list(held), []
    for t in order:
        if t in held or len(out) >= room:
            continue
        if t in corr.index:
            peers = [b for b in book if b in corr.columns]
            if peers and float(corr.loc[t, peers].abs().max()) > cap:
                continue
        out.append(t); book.append(t)
    return out


def _apply_max_weight(w: dict, cap: float) -> dict:
    """Cap any single weight at `cap`, REDISTRIBUTING the excess across the uncapped names.

    The previous version was `min(x, cap)` with no redistribution. That does not cap the book, it
    SHRINKS it: the weights stopped summing to 1, so setting `max_weight` quietly moved capital to
    cash instead of moving it to the other names. A position limit that silently reduces gross
    exposure is a different instrument from the one the config describes, and the difference only
    shows up as a return shortfall nobody can attribute.

    Iterative, because redistributing pushes the remaining names UP and can carry a second one over
    the cap. One pass would leave that second breach in place — the exact bug in miniature. Converges
    in at most one round per name.

    A cap below 1/n cannot be satisfied by any allocation summing to 1: every name would have to
    exceed it. In that case the weights are equal and the caller's gross target is what gives, which
    is the honest degradation — the alternative is silently holding cash again.
    """
    if cap >= 1.0 or not w:
        return w
    if cap <= 1.0 / len(w):
        return {n: 1.0 / len(w) for n in w}
    out, capped = dict(w), set()
    while True:
        over = [n for n, x in out.items() if x > cap + 1e-12 and n not in capped]
        if not over:
            return out
        capped.update(over)
        excess = sum(out[n] - cap for n in over)
        for n in over:
            out[n] = cap
        free = [n for n in out if n not in capped]
        if not free:                    # everything is at the cap; nothing left to absorb the excess
            return out
        total_free = sum(out[n] for n in free)
        for n in free:
            # Pro rata by current weight, so redistribution preserves the ranking among the names
            # that still have room rather than flattening them.
            out[n] += excess * (out[n] / total_free) if total_free > 0 else excess / len(free)


def _weights(names: tuple[str, ...], vol: pd.Series | None,
             cfg: MomentumRotationConfig) -> dict:
    """Equal weight, or inverse-volatility normalised to the same gross exposure.

    `max_weight` is applied on both branches now. That is a simplification, NOT a bug fix: equal
    weights are exactly 1/n, so a cap above 1/n cannot bind and one below it cannot be satisfied by
    any allocation summing to 1. The audit initially reported the old early return as a second
    defect; a mutation bite reverting it left every test green, and the claim was retracted.
    """
    if not names:
        return {}
    cap = cfg.portfolio.max_weight
    if not cfg.portfolio.inverse_vol_sizing or vol is None:
        return _apply_max_weight({n: 1.0 / len(names) for n in names}, cap)
    inv = {n: 1.0 / v for n in names
           if (v := float(vol.get(n, float("nan")))) and np.isfinite(v) and v > 0}
    if len(inv) < len(names):                      # fall back rather than concentrate on partial data
        return _apply_max_weight({n: 1.0 / len(names) for n in names}, cap)
    tot = sum(inv.values())
    return _apply_max_weight({n: x / tot for n, x in inv.items()}, cap)


def decide(panel: pd.DataFrame, source: CandidateSource, cfg: MomentumRotationConfig,
           held: set[str], overlay: pd.Series | None = None,
           corr: pd.DataFrame | None = None, vol: pd.Series | None = None) -> Decision:
    """One session's decision. `panel` is a single date's rows; `held` is the current book.

    A held name is only exited once it falls out of the top (n_hold + buffer). Without that buffer
    the book churns on rank noise at the boundary and turnover doubles for nothing.

    A held name that has left the candidate pool IS exited — that is the followed trader's own sell
    signal. Protection against a bad feed lives upstream (stale gate, shrink guard), not here.
    """
    d = pd.Timestamp(panel["date"].iloc[0])
    elig = source.eligible(d)
    cand = panel[panel.ticker.isin(elig)].dropna(subset=["score"])
    # ONE ROW PER TICKER. A duplicated (ticker, date) -- a feed replaying a bar, two sources merged
    # without dedupe -- survived into the ranking as two entries: with n_hold=2 and rows
    # AAA/AAA/BBB, `enter` came back ('AAA', 'AAA'), BBB never got its slot, the recorded score was
    # the loser's, and the runtime would try to submit two BUYs for one symbol in one session.
    # Last row wins, matching "the latest bar for that date".
    cand = cand.drop_duplicates(subset=["ticker"], keep="last")
    # Sort by score, then by TICKER, so the order rows happen to arrive in cannot change what we
    # buy. Without the second key, four names scoring identically entered ('AAA','BBB') read
    # forwards and ('DDD','CCC') read backwards.
    #
    # An alphabetical tiebreak is only safe because the score is continuous -- exact ties mean a
    # degenerate day, not a routine one. It would NOT be safe on a saturating percentile, where
    # ties are common and this bias once sent 77% of picks to A/B/C tickers.
    ranked = (cand.sort_values(["score", "ticker"], ascending=[False, True])
                  .set_index("ticker")["score"])

    # Blend the intra-session rank into the daily one. Ranks, not raw scores: the two horizons have
    # different scales and adding them directly would let whichever is noisier dominate. A name with
    # no intraday history scores neutral (0.5) rather than being dropped.
    w = cfg.intraday.weight
    blended = ranked
    if overlay is not None and w and len(ranked):
        base = ranked.rank(pct=True)
        ov = overlay.reindex(ranked.index).rank(pct=True).fillna(0.5)
        blended = (base + w * ov).sort_values(ascending=False)
    if not cfg.intraday.entry_only:
        ranked = blended

    p = cfg.portfolio
    # Eligibility and exits stay on `ranked`. Under entry_only that is the DAILY score, so the
    # overlay can never force a sale — it only reorders which eligible name is bought first.
    keep = set(ranked.head(p.n_hold + p.buffer).index)

    # SELLING WHEN A NAME LEAVES THE POOL IS DELIBERATE AND VALUABLE — it is worth 29 points of
    # return (+62.5% vs +33.4% over 167 sessions). The pool drops a name precisely when the trader
    # being followed EXITS it, so a pool departure is his sell signal, not noise.
    #
    # The safety concern — "a feed glitch must not liquidate the book" — is handled UPSTREAM, and
    # has to be, because it cannot be judged from here:
    #   - a stale or failed source blocks the whole session (runner: stale gate)
    #   - a refresh that loses an implausible share of its set is rejected and keeps the old set
    #     (pool: shrink guard) — this catches the dangerous case of a source that SUCCEEDS while
    #     returning a truncated set, which would otherwise pass the stale gate
    # An earlier attempt to protect the book here instead, by never selling an unranked holding,
    # cost those 29 points: it also stopped following the trader's genuine exits.
    # An EMPTY ranking is not a decision to sell the book. If nothing scored -- every candidate NaN
    # after gates, a panel too short, a feed gap -- then `keep` is empty and every held name looks
    # like it fell out. The Pg runner guards this upstream, but backtest and the local Nautilus path
    # call decide() directly and had no such guard, so a data-collapsed day liquidated everything
    # and read in the trade log as an ordinary rotation.
    #
    # Leaving the book untouched is the conservative answer: we cannot rank, so we cannot claim a
    # name has stopped being worth holding. Exits driven by evidence -- give-back, operator
    # blacklist -- live outside this function and still fire.
    if not len(ranked):
        return Decision(date=d, hold=tuple(sorted(held)), enter=(), exit=(), scores={})
    exits = tuple(sorted(set(t for t in held if t not in keep)
                         | cluster_exits(held, ranked, corr, cfg)))

    room = p.n_hold - (len(held) - len(exits))
    order = [t for t in blended.index if t in keep] if cfg.intraday.entry_only else list(blended.index)
    order = [t for t in order if t not in exits]
    enters = tuple(_diversified(order, held - set(exits), room, corr, p.max_correlation))
    hold = tuple(sorted((held - set(exits)) | set(enters)))
    return Decision(date=d, hold=hold, enter=enters, exit=exits,
                    scores=ranked.head(p.n_hold + p.buffer).to_dict(),
                    weights=_weights(hold, vol, cfg))
