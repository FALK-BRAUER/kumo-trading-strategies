"""CRSISHORT's PURE decision layer. No Nautilus, no database, no broker, no clock.

The same function serves the backtest and the live path, because two implementations of one decision
disagree — and #123 is the ticket that proves the cost: four engines were written in the lab and each
disagreed with the others by 20-40 points. Every disagreement was a real defect (RAW-price splits, a
variant ranking on the session it executed in, positions dropped from the universe without booking
the trade, orders placed from two-day-stale signals), and NONE was found by inspection.

WHAT MUST NEVER APPEAR HERE: order submission, position lookups, wall-clock reads, a broker, or a
price fetch. A decision that touches those cannot be replayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kumo_strategies.strategies.crsi_short.config import CrsiShortConfig
from kumo_strategies.strategies.crsi_short.indicators import annualised_log_vol, connors_rsi

#: The only price adjustment this strategy may be computed on.
ADJUSTED = "all"


@dataclass(frozen=True)
class ShortDecision:
    """WHAT the strategy wants, never HOW it is executed.

    `hold`/`enter`/`exit` are the vocabulary every gateway and both slot detectors read. `limits`
    carries the SHORT-SELL limit price per entering name, because the limit is part of the rule and
    not of the plumbing: #123 measures a different strategy without it (+99.4% on 384 trades, worse
    tail, and it loses to the limit above 100bps of cost).

    `refused` is symbol -> reason for a signal that was NOT taken. Recorded rather than dropped —
    #123's acceptance requires refused entries to be logged, and a borrow gate that silently removes
    names is indistinguishable from a signal that never fired.
    """

    hold: tuple[str, ...] = ()
    enter: tuple[str, ...] = ()
    exit: tuple[str, ...] = ()
    limits: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)


def build_feature_panel(bars: pd.DataFrame, cfg: CrsiShortConfig, *, adjustment: str) -> pd.DataFrame:
    """Features as of the PRIOR close, for every (ticker, session).

    `adjustment` HAS NO DEFAULT and is keyword-only, so no call site can inherit one. #123: the
    lab's RAW store booked reverse splits as -1000% short losses on 7.4% of trades and INVERTED the
    2025-26 verdict. This repo's `backtesting/data.py` answers that risk by gating corporate actions
    in the strategy layer instead of adjusting prices — which is sound for a long lane that gates at
    ENTRY, and insufficient here: the -1000% is booked on a position that was ALREADY OPEN when the
    split happened, and no entry gate reaches it. So this strategy requires adjusted prices, refuses
    anything else, and says so at the boundary rather than in a comment.

    EVERY feature is `.shift(1)`. The decision for session `d` is computable from data through `d-1`
    only: the signal is the prior close, the limit rests during `d`. A look-ahead here does not fail,
    it inflates the result and ships.
    """
    if adjustment != ADJUSTED:
        raise ValueError(
            f"CRSISHORT requires split-adjusted prices (adjustment={ADJUSTED!r}), got "
            f"{adjustment!r}. On RAW prices a reverse split reads as a -1000% short loss on an "
            "already-open position, which is not reachable by an entry-time corporate-action gate "
            "(#123). Pull with Alpaca `adjustment=all` or an equivalent.")
    missing = {"ticker", "date", "high", "close", "volume"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    panel = bars.sort_values(["ticker", "date"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    g = panel.groupby("ticker", sort=False)

    panel["asof_close"] = g["close"].shift(1)
    panel["asof_high"] = g["high"].shift(1)
    panel["asof_dollar_volume"] = g["close"].shift(1) * g["volume"].shift(1)
    panel["crsi"] = g["close"].transform(
        lambda s: connors_rsi(s, cfg.crsi_rsi_period, cfg.crsi_streak_period,
                              cfg.crsi_rank_period).shift(1))
    panel["ann_vol"] = g["close"].transform(
        lambda s: annualised_log_vol(s, cfg.vol_window).shift(1))
    # WARMUP IS A UNIVERSE PROPERTY. Signalling on three warm names out of five thousand is not an
    # early answer, it is a different and much narrower strategy.
    panel["sessions_seen"] = g.cumcount()
    panel["adjustment"] = adjustment

    # THE LIQUIDITY AND PRICE FLOORS GATE ENTRIES ONLY -- #123 holds through liquidity dips, and
    # `decide` never exits a held name for leaving the universe. That was one of the four lab
    # defects: positions dropped from the universe without booking the trade, so their P&L stayed
    # in equity and vanished from the statistics.
    return apply_gates(panel, cfg)


#: Everything a cached panel must agree with the config about. The indicator PERIODS cannot be
#: re-applied to a panel that was computed with different ones, and the two floors decide which rows
#: were kept at all, so a cache built under other values is a different panel wearing this one's
#: column names.
PANEL_SIGNATURE = ("crsi_rsi_period", "crsi_streak_period", "crsi_rank_period", "vol_window",
                   "price_floor", "manage_min_dollar_volume")


def panel_signature(cfg: CrsiShortConfig) -> str:
    return "|".join(f"{k}={getattr(cfg, k):g}" for k in PANEL_SIGNATURE)


def apply_gates(panel: pd.DataFrame, cfg: CrsiShortConfig) -> pd.DataFrame:
    """Re-derive every THRESHOLD from the indicator columns, for this config.

    Split from `build_feature_panel` so a sweep does not recompute ConnorsRSI once per arm — and,
    more importantly, so a cached panel is never fed back through the indicator layer. Recomputing
    on a panel that has already been FILTERED silently changes the answer: the rows a liquidity
    floor removed leave GAPS, and a 100-session window over a gappy series is a different
    statistic. That bug produced 146 trades where the lab has 228, and nothing raised.

    What can be re-applied here: the ConnorsRSI and volatility thresholds, both dollar-volume
    floors, the price floor, and the entry limit. What cannot: the indicator PERIODS, which are
    baked into the columns. A panel carrying a different signature is refused rather than re-gated.
    """
    got = panel["panel_signature"].iat[0] if "panel_signature" in panel.columns else None
    want = panel_signature(cfg)
    if got is not None and got != want:
        raise ValueError(
            f"this panel was built as {got!r} and the config asks for {want!r}. The indicator "
            "periods and the floors that decided which rows were kept cannot be re-applied to an "
            "existing panel — rebuild it.")

    panel = panel.copy()
    warm = panel["sessions_seen"] >= cfg.warmup_sessions
    panel["in_universe"] = (panel["asof_close"].gt(cfg.price_floor)
                            & panel["asof_dollar_volume"].gt(cfg.min_dollar_volume))
    panel["eligible"] = warm & panel["in_universe"] & panel["crsi"].notna() & panel["ann_vol"].notna()

    # MANAGEABLE is a lower bar than eligible and a different question: can this name still be
    # priced and traded at all? Below it the lab's signal table has no row, and a held position is
    # closed at its last mark — 4% of #123's trades. Held names are NOT dropped for falling below
    # the ENTRY floor (that is `hold_through`); they are dropped for falling out of the tape.
    panel["manageable"] = (panel["asof_close"].gt(cfg.price_floor)
                           & panel["asof_dollar_volume"].gt(cfg.manage_min_dollar_volume)
                           & panel["crsi"].notna() & panel["ann_vol"].notna())

    # TWO UNIVERSE CHECKS, ONE SESSION APART, because the order rests overnight. The lab screens on
    # the SIGNAL session and screens again on the session the limit rests in — `for tkr in [t for t
    # in pend if t not in entry_pool.index]: del pend[tkr]` cancels a resting order whose name has
    # dropped below the floor. Both are knowable in advance (dollar volume is always the PRIOR
    # session's tape), so neither is look-ahead; dropping either changes the trade set.
    vol_ok = (panel["ann_vol"].gt(cfg.min_annual_vol) if cfg.min_annual_vol is not None
              else panel["ann_vol"].notna())
    panel["signalled_yesterday"] = (
        panel.groupby("ticker", sort=False)["in_universe"].shift(1).fillna(False).astype(bool)
        & warm & vol_ok & panel["crsi"].gt(cfg.crsi_entry))
    #: What `decide` acts on: signalled at the prior close AND still in the universe today.
    panel["signal"] = panel["signalled_yesterday"] & panel["eligible"]
    # ROUNDED TO THE PENNY, because that is the order that gets sent: every name here is above the
    # $5 floor, so the tick is $0.01, and an unrounded limit is a price no venue accepts. It also
    # changes fills — a limit at 12.3449 rests at 12.34, not 12.35.
    panel["limit_px"] = (panel["asof_close"] * (1.0 + cfg.entry_limit_pct)).round(2)
    panel["panel_signature"] = want
    return panel


def decide(day: pd.DataFrame, cfg: CrsiShortConfig, held: set[str],
           borrow: dict[str, float | None] | None = None,
           pending: set[str] | None = None) -> ShortDecision:
    """One session's entry decision. `held` is what THIS STRATEGY is short — never the account's book.

    Passing the account's positions here is how TECHIVOL-005 proposed exiting eight names belonging
    to two other strategies on its first live session.

    NO EXITS COME OUT OF HERE. CRSISHORT's exits are structural and per-position (`exits.py`); there
    is no ranking-driven exit, and a held name that leaves the universe stays short. A `decide` that
    also exited would give the book two exit authorities that must agree and cannot be tested
    together.

    `borrow` is symbol -> annual locate fee, `None` for a name with NO locate. Required whenever
    `cfg.max_borrow_fee_annual` is set: a fee ceiling configured without the data would look armed
    and be inert, which is the #26 failure mode this repo has already paid for twice.

    `pending` is the names with a resting limit order from an earlier session. THEY CONSUME SLOTS:
    the lab computes `room = slots - len(pos) - len(pend)`, and counting only filled positions would
    let the book commit to more names than it can hold and then reject the fills that arrive last.

    RANKING, when more names signal than there are slots: most overbought first, which is the lab's
    `cands.connors.nlargest(room)`. Ties break on ticker so the order is deterministic.
    """
    if cfg.max_borrow_fee_annual is not None and borrow is None:
        raise ValueError(
            "cfg.max_borrow_fee_annual is set but no `borrow` map was passed. Proceeding would "
            "leave the fee ceiling armed and inert. Pass the locate data, or set the ceiling to "
            "None to state deliberately that this run has no borrow gate.")

    pending = pending or set()
    hold = tuple(sorted(held))
    room = cfg.n_slots - len(held) - len(pending)
    signals = day.loc[day["signal"].fillna(False)] if "signal" in day.columns else day.iloc[0:0]
    signals = signals.loc[~signals["ticker"].isin(held | pending)]
    if signals.empty or room <= 0:
        return ShortDecision(hold=hold, scores=_scores(day))

    ranked = signals.sort_values(["crsi", "ticker"], ascending=[False, True])
    enter: list[str] = []
    limits: dict[str, float] = {}
    refused: dict[str, str] = {}
    for row in ranked.itertuples(index=False):
        sym = row.ticker
        if len(enter) >= room:
            # Recorded, not dropped: a signal we had no slot for is a capacity fact, and #123's
            # trade count is the number the paper gate is declared against.
            refused[sym] = f"no slot ({cfg.n_slots} held or resting)"
            continue
        why = _borrow_refusal(sym, cfg, borrow)
        if why:
            refused[sym] = why
            continue
        px = float(row.limit_px)
        if not np.isfinite(px) or px <= 0:
            refused[sym] = "no usable limit price"
            continue
        enter.append(sym)
        limits[sym] = px
    return ShortDecision(hold=hold, enter=tuple(enter), limits=limits,
                         scores=_scores(day), refused=refused)


class _Locatable:
    """"Borrowable, fee unknown" — a third state the fee map could not otherwise express.

    NOT A NUMBER, deliberately. Interactive Brokers publishes a shortability code and shares
    available (ticks 46 and 89) and NO fee-rate tick, so on the venue this lane reaches first every
    locate is exactly this fact. The alternatives both lie:

      0.0                a fabricated fee that clears any ceiling and reads as FREE BORROW forever,
                         in the decision, in the journal, and in anything reconciled against it.
                         Nothing downstream can tell it from a venue that genuinely quoted zero.
      ceiling = None     turns the whole gate off, availability included, so an unborrowable name
                         consumes a slot with an order that cannot fill — TECHIVOL-005 did that
                         eight times in one session — and moves the refusals out of the lane's
                         journal, which #123's acceptance requires to be logged.

    Recognised by IDENTITY, so it cannot be produced by arithmetic or by a parsed number.
    """

    __slots__ = ()

    def __repr__(self) -> str:                                          # shows up in journal rows
        return "LOCATABLE(fee unknown)"


#: The single instance. `borrow[sym] is LOCATABLE` is the whole protocol.
LOCATABLE = _Locatable()


def _borrow_refusal(sym: str, cfg: CrsiShortConfig, borrow: dict[str, float | None] | None) -> str | None:
    """Why this name cannot be shorted, or None.

    Absence from the map and a `None` fee are the SAME fact — no locate — and both refuse. #123:
    shortability is the constraint, not the fee (IBKR has locate on 88% of the traded names, median
    1.06%/yr, break-even 216%/yr). Treating an unknown name as borrowable would consume a slot with
    an order that cannot fill, which is what TECHIVOL-005 did eight times in one session.
    """
    if cfg.max_borrow_fee_annual is None:
        return None
    fee = (borrow or {}).get(sym)
    if fee is None:
        return "no locate"
    # AVAILABLE, FEE UNKNOWN. Checked before any numeric test, because every one of them below
    # would raise or lie on a non-number. The ceiling is a statement about fees; with no fee there
    # is nothing to compare, and the comparison is SKIPPED rather than resolved in either direction.
    if fee is LOCATABLE:
        return None
    if not np.isfinite(fee):
        return "no locate (non-finite fee)"
    if fee > cfg.max_borrow_fee_annual:
        return f"borrow {100*fee:.0f}%/yr over the {100*cfg.max_borrow_fee_annual:.0f}% ceiling"
    return None


def _scores(day: pd.DataFrame) -> dict[str, float]:
    if "crsi" not in day.columns:
        return {}
    d = day.loc[day["crsi"].notna()]
    return {str(t): float(c) for t, c in zip(d["ticker"], d["crsi"])}
