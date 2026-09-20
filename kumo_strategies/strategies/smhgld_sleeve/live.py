"""The promoted configuration, and the caveats an operator must read before arming this.

These travel in CODE rather than a document because the thing that ships is what gets read.
"""

from __future__ import annotations

from kumo_strategies.strategies.smhgld_sleeve.config import SmhGldSleeveConfig, weight_split


def live_config() -> SmhGldSleeveConfig:
    """The researched configuration. Every departure from defaults is justified below.

    There are none. Every value in `SmhGldSleeveConfig` was set to the researched value at
    definition, because this lane has exactly one form and a second spelling of it would be the
    drift the other lanes' configs warn about.
    """
    return SmhGldSleeveConfig()


def live_notes(cfg: SmhGldSleeveConfig | None = None) -> dict[str, str]:
    """Operator-facing caveats. Read these before arming, not after.

    `cfg` defaults to `live_config()`; the split the notes state is READ off it (`weight_split`),
    never typed here — this text said "32/68" while the runtime label said "48/52" (#186, #219).

    Deliberately harsher than the backtest, because the backtest is mine and the caveats are the
    part I am least likely to be wrong about.
    """
    split = weight_split(cfg or live_config())
    return {
        "THE_FILL_IS_MARKET_ON_CLOSE_AND_THE_SIZING_IS_TWENTY_MINUTES_EARLIER":
            "Every number on this lane was measured with the sizing and the fill on the SAME "
            "close. Live, they are twenty minutes apart, and that is the residual deviation after "
            "the best available fix.\n\n"
            "The fill itself is the close BY CONSTRUCTION: the lane sends a MARKET-ON-CLOSE order "
            "(`OrderRequest(limit_px=None, time_in_force='AT_THE_CLOSE')`), driven end to end on "
            "both venues — IB's adapter emits orderType='MOC', Alpaca's emits "
            "{'type':'market','time_in_force':'cls'}, and an unmapped time-in-force RAISES at every "
            "step rather than downgrading to a day order.\n\n"
            "WHAT IS NOT CLOSED: `target_qty = weight * equity / price` is computed at the decision "
            "slot, 15:40, and filled at 16:00. Twenty minutes of sizing error where the backtest "
            "had none. Strictly better than the six and a half hours an open+5m decision would "
            "have carried, and not zero. Do not expect the live curve to reproduce the PR's CAGR "
            "or drawdown to the decimal.\n\n"
            "WHY 15:40 AND NOT LATER: a MOC has a cutoff — IB ~15:50 ET, Alpaca 15:45 ET, both "
            "DOCUMENTED rather than measured. `close-20m` clears them by +10m and +5m. `close-10m` "
            "is 15:50: exactly IB's cutoff and five minutes past Alpaca's, with no session left to "
            "retry in. `close-30m` was rejected because ten more minutes of margin costs twenty "
            "more minutes of drift, and on a 0.25pp band that drift is the thing being traded "
            "away.\n\n"
            "THE HALF-DAY CASE IS ACCEPTED AND NOISY, NOT SOLVED. On a 13:00 close `close-20m` "
            "moves to 12:40; whether either venue shifts its MOC cutoff by the same three hours is "
            "UNKNOWN. Roughly nine sessions a year, all low-volume, and the next session catches "
            "the drift. A refused closing order is reported as its own named condition — \"THE "
            "REBALANCE DID NOT HAPPEN\" — so a missed half-day close cannot read as a lane that "
            "decided nothing (platform issue 965).",
        "what_this_lane_is":
            f"A fixed {split} sleeve of SMH and GLD, rebalanced whenever the risk leg drifts a "
            "quarter of a point from target — about 93 times a year. It takes no view of its own, "
            "forecasts nothing, and its only decision is when the weights have moved far enough to "
            "be worth correcting. (Its market view, INDEX_VS_MA / EXIT_ONLY, is the platform's to act "
            "on and changes no order this lane places — #212.)",
        "the_weight_is_the_strategy":
            f"At {split} (the operator's decision, 2026-09-13, #219): over 2024-2026 the sleeve returned "
            "46.4% CAGR at a 14.8% maximum drawdown against SMH's 57.0% and 35.7%; over 2021-2026, "
            "25.8% at 27.6% against 34.1% and 45.3%. Nearly all of the difference from SMH alone is "
            "the weight. At 48% the lane tracks semis and gold about equally by dollars — size it "
            "knowing that.",
        "the_weight_was_32_and_the_measurement_behind_that_still_holds":
            "32% was chosen against the ulcer index on 2021-2026, where it bottoms flat across "
            "28-32%; 48% was what the bear-free 2024-2026 window preferred. The DECISION (the operator, "
            "2026-09-13, #219) took 48% for return and accepted the cost the lab measured (2 bps, "
            "fixed-weight walk): 2018-2026 "
            "CAGR +21.8% -> +25.2%, maxDD -23.0% -> -27.6%, 2022 -11.0% -> -16.3%; 2005-2026 CAGR "
            "+14.7% -> +16.1%, maxDD -32.4% -> -37.3%; Sharpe flat on both. If someone proposes "
            "moving the weight again, ask which window they measured it on, and quote both.",
        "eleven_timing_mechanisms_failed_on_this_pair":
            "Binary trend switches at five lookbacks, continuous tilts with and against the trend, "
            "volatility targeting at three windows, and rotation into GLD, UUP, BTAL and BIL. "
            "Under a null that re-runs the lookback sweep inside every permutation, none separated "
            "from chance (p = 0.25 to 1.00). Tilting in EITHER direction deepened the drawdown, "
            "monotonically with tilt size. If someone proposes adding a signal to this lane, that "
            "is the evidence to weigh it against.",
        "gold_is_not_a_hedge_here":
            "GLD's own maximum drawdown over 2024-2026 was 26.4%, deeper than the sleeve's. It "
            "earns its place through weak correlation (+0.21), not stability. Measured properly it "
            "returns the same annualised rate inside SMH drawdowns as outside them (15.1% against "
            "13.9%), and in genuine deep stress over 2021-2026 it LOST 9.2% annualised. Do not "
            "expect it to rescue the book; expect it not to participate.",
        "2022_IS_THE_STRESS_CASE_AND_THE_SLEEVE_FAILS_IT":
            "Read this before sizing. Over 2021-2026, which includes the 2022 bear, NO SMH/GLD "
            f"weight holds a 15% drawdown. At the decided {split} the sleeve lost -16.3% in 2022 "
            "and drew 27.6% over 2021-2026, against 14.8% over 2024-2026 alone — because gold "
            "returned -0.8% in 2022 while semis fell 33.5%, so the defensive leg did not rescue the "
            "book, it merely declined to participate. 2024-2026 contains no bear market. Size for "
            "roughly 28%, not 15%.",
        "THE_SAMPLE_IS_EFFECTIVELY_ONE_WINDOW":
            "Quote this with every figure in these notes. 2021-01-04 to 2026-09-10 is 1,428 "
            "sessions but it is ONE gold bull market and ONE equity bear. Every drawdown figure "
            "here is computed on a single 2022. That is one observation about one regime, not "
            "1,428 observations, and the -27.6% drawdown should never be quoted without it.",
        "gold_had_a_bull_run_inside_the_test_window":
            "GLD returned +26.7% in 2024 and +63.7% in 2025, then 0.0% in 2026 so far. A material "
            "part of the sleeve's CAGR is that run rather than diversification. The drawdown "
            "benefit is the robust half; the return contribution is not. In 2026 alone the sleeve "
            "returned 24.2% against SMH's 50.1%.",
        "the_pair_is_not_the_best_available":
            "Held to the same 15% drawdown budget, a memory basket (MU/STX/WDC) against the dollar "
            "scored higher on BOTH windows — 38.5% CAGR at Sharpe 2.15 recently and 21.8% at 1.59 "
            "through 2022, where SMH/GLD cannot reach the budget at all. SMH/GLD was chosen for "
            "liquidity and simplicity, not because it won.",
        "a_wider_band_is_not_a_safer_band":
            "The band was swept from 0.25 to 10 points over both windows at the earlier 32% weight; "
            "there the tightest band tested was best on drawdown (23.1%), ulcer (6.23%), 2022 "
            "(-10.9%) AND return at once, and the lab's 48% walk uses the same 0.25pp band. Anyone "
            "tempted to widen it to save trading costs should note that "
            "a tight band makes each trade SMALL — 0.75% of the sleeve on average — so at 50bp per "
            "unit traded, five times what is assumed here, it costs 0.4 points of CAGR.",
        "this_lane_trades_about_twice_a_week":
            "93 rebalances a year, average size 0.75% of the sleeve. That is by design and it was "
            "measured, not assumed: 5-point and 3-point bands were tried first and traded 3-9 "
            "times a year. Both SMH and GLD trade billions of dollars a day, so the trade sizes "
            "here are immaterial to liquidity at any sleeve size this book will reach.",
        "THIS_LANE_TRADES_BY_TARGET_AND_DELTA_NOT_BY_ENTRY_AND_EXIT":
            "Read this before wiring it. Every other lane trades as an ENTRY sized once at arrival "
            "and an EXIT that sells everything, and nothing revisits a position's size. This lane "
            "never enters and never exits after its first session: it holds two legs forever and "
            "trades only deltas toward a target, about 93 times a year. `order_plan()` is the whole "
            "execution contract — target_qty = weight * equity / price, delta = target_qty - "
            "held_qty, sells before buys. A runner that consumes only `enter` and `exit` will "
            "execute NOTHING here while the journal shows a lane deciding every session.",
        "the_four_partial_sell_invariants":
            "momentum_rotation's `rebalance_band` is backtest-only because a partial sell disturbs "
            "four live invariants. This lane cannot defer them, so each is handled and pinned in "
            "tests/strategies/smhgld_sleeve/test_partial_sell_invariants.py: TRAIL STATE — no "
            "give-back, stop or peak tracking exists here, so there is none to corrupt; ORDER "
            "IDENTITY — a target/delta plan emits at most one order per symbol per session, so a "
            "trim and an exit can never collide on a client_order_id keyed (session, symbol, side, "
            "attempt); ROUND TRIPS — neither leg is ever taken to zero, so the lane opens and "
            "closes no round trips and FIFO round-trip KPIs simply do not describe it, position-"
            "level accounting does; CLAIMS LEDGER — the claim is SET ABSOLUTELY to the "
            "post-trade book quantity via `sync_claim_to`, never adjusted by a delta. Every order "
            "carries `post_trade_qty` (held + rounded delta) for exactly that call. Do not set it "
            "to `target_qty`: the delta is rounded to a whole lot and the target is not.",
        "housekeeping_this_lane_needs":
            "It holds both legs continuously, so there is no flat state to reconcile against and "
            "'no position' is always an error rather than a resting state. Three things follow. "
            "Protective orders sized to a whole position are wrong here and must be resized or "
            "absent, since the lane's own sells would trip them. Reconciliation must compare "
            "HELD_QTY against target rather than presence or absence of a position. And a missing "
            "leg is an incident, not a decision: `decide()` returns regime 'rebalance' with that "
            "leg in `enter`, which is the only path where this lane opens anything after day one.",
        "ENTRIES_BLOCKED_TELLS_YOU_ALMOST_NOTHING_ABOUT_THIS_LANE":
            "The lane declares signal=NONE, so `entries_blocked` answers NO naming that — eleven "
            "timing mechanisms were tested on this pair and none separated from chance, so there is "
            "no window to block on. That is a measurement, not a gap. But understand what the hook "
            "would mean here even if a view WERE declared: EXIT_ONLY keeps every HELD name at its "
            "own target weight, and this sleeve holds both legs at all times, so EXIT_ONLY is a "
            "NO-OP for it. Rebalancing sells AND buys both continue. Reading 'entries blocked' "
            "against a two-leg sleeve is close to reading nothing.",
        "AN_EMERGENCY_EXIT_OF_THIS_SLEEVE_IS_UNDONE_BY_ITS_NEXT_DECISION":
            "No emergency trigger is wired for any lane, so `emergency_exit` answers NO naming "
            "that. If an operator flattens this lane by hand, know what happens next: the sleeve is "
            "empty, `decide()` returns regime 'opening', and it buys BOTH legs back on its next "
            "session. An emergency stop on this lane is not a stop, it is a round trip and a "
            "taxable event. To actually stop it, deregister or halt the lane — do not flatten it.",
        "it_cannot_assess_itself_and_that_is_the_correct_answer":
            "`self_assessment` answers UNKNOWN naming that no envelope is registered, with "
            "evidence_sufficient=False, so the lane cannot quarantine itself. Do NOT assess it "
            "against target drift: drift is the INPUT this strategy consumes, about 93 times a "
            "year, so drifting is the mechanism working rather than the lane misbehaving. A real "
            "envelope needs pre-registered distributions over independent live windows, and by "
            "this lane's own sample caveat there is effectively ONE.",
        "A_SPLIT_IN_EITHER_LEG_SILENTLY_HALVES_WHAT_THIS_LANE_MAY_TRADE":
            "issue 179. A split doubles the SHARE COUNT of a live position while the "
            "claims ledger still holds the old number, so the lane is refused access to half of "
            "its own position — and it is SILENT, because `over_claimed` reports only over-"
            "claiming and a post-split ledger UNDER-claims. For this lane that means a rebalance "
            "executes at half size and the drift it was correcting simply persists, session after "
            "session, with the board showing a lane that decided and traded. This is not "
            "hypothetical here: SMH split 2:1 on 2023-05-05, inside this lane's own measured "
            "window. The lane itself survives a split correctly — target_qty scales with held_qty "
            "so the delta is unchanged and no spurious rebalance fires — but the LEDGER does not, "
            "and until #179 lands a split day needs a manual claim reconciliation.",
        "costs_are_modelled_not_paid":
            "Research charged 10bps per unit of weight traded, on closes. No slippage and no real "
            "fills. The measured half-spreads used elsewhere in this repo do not cover ETFs "
            "(half_spreads.json is equities-only; SMH and GLD both fall back to a 922-stock "
            "median). Measured separately: GLD 0.373bps, SMH not yet measured. At roughly 1.2 "
            "units of turnover per year this lane is far less cost-sensitive than the switching "
            "version it replaces, which turned over 14 times a year.",
    }
