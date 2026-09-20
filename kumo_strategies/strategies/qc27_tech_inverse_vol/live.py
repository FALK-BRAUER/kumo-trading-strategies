"""The ONE config QC27 trades with, and why each field departs from the default (#33, #63).

WHY THIS EXISTS. `ACTIVATION.md` asked for a live config builder so cockpit does not hardcode
runtime values that drift from the verified backtest. Three of QC27's shipped settings are NOT the
dataclass defaults, and each was measured this week. If cockpit spells them itself, the first person
to change one has no way to know what it cost -- and a config that differs from the researched one
is a different strategy wearing its name.

Cockpit's settings domain may still override any field for an operator. The point is that the
DEFAULT it starts from is this, and this is versioned next to the research that produced it.
"""

from __future__ import annotations

from kumo_strategies.strategies.market_view import (
    MarketAction, MarketSignal, MarketViewConfig)
from kumo_strategies.strategies.qc27_tech_inverse_vol.config import QC27TechInverseVolConfig

#: Trader-visible identity. `005` because 001-004 are MANUAL, MOMENTUM, QC345 and BCTROT; a shared
#: `order_id_tag` does not degrade, it raises at `Trader.add_strategy` and the node does not boot.
STRATEGY_NAME = "TECHIVOL"
ORDER_ID_TAG = "005"
WIRE_ID = f"{STRATEGY_NAME}-{ORDER_ID_TAG}"

#: Minutes after the open at which the rebalance fires. 150 = 12:00.
OPEN_OFFSET_MINUTES = 150


def live_config() -> QC27TechInverseVolConfig:
    """The researched configuration, with every departure from default justified below."""
    return QC27TechInverseVolConfig(
        # A Nautilus `Bar` carries OHLCV and nothing else, so the live feed can NEVER supply
        # `close_adj` -- the package default. The adapter refuses to construct with it rather than
        # failing at the first rebalance weeks later, which is the shape of QC345's `KeyError:
        # 'eligible'`. Cost of the switch, measured over the full 20 months: monthly -3.0pp, weekly
        # +11.5pp, daily -6.6pp, Sharpe moving at most 0.05. The sign is inconsistent, so this is
        # noise rather than a systematic penalty. The real residual risk is that QC27 has no
        # `corporate_action_window`, so `close` carries raw split and dividend exposure.
        momentum_price_field="close",

        # DAILY. Operator decision (2026-08-21), taken on the drawdown rather than the
        # return. Over 2025-01-02 -> 2026-08-10 daily is the best of the three on risk -- maxDD
        # -21.9% against weekly's -26.6% and monthly's -33.5%, Calmar 5.10 against 4.30 and 2.90 --
        # while giving up ~7pp of total return to weekly.
        #
        # The mechanism is measured, not assumed. Monthly gets one to three decision points across a
        # ten-week decline: protective when the dip recovers before it can act (V-shaped Oct-Nov 25:
        # monthly -18.2%, daily -20.2%) and the whole problem when it does not (sustained Jun-Aug 26:
        # monthly -21.8%, daily -0.6%). Daily is the cadence that can leave.
        #
        # WHAT THIS CHOICE COSTS, stated because it is the reason weekly was the analytic pick:
        #   * Daily is WORST of the three in the first half of the window (+61.6%, Sharpe 1.73) and
        #     best in the second (+103.5%, 1.99). A split-half flips the ordering, so its 20-month
        #     win rests on one regime. Weekly is the only cadence never worst in either half.
        #   * Daily sends ~7.95 legs per session on a TEN-name book while rotating about one name.
        #     The other ~7 are the inverse-vol weights drifting a fraction of a percent and the
        #     runner trading that drift -- $23.7k of cost over the window buying nothing.
        # A no-trade band would remove most of that churn and is UNTESTED. Until it is measured,
        # this config pays the churn deliberately.
        rebalance_period="D",

        # THE LANE'S OWN MARKET VIEW. The operator approved it 2026-09-11 at a stated price — 31% of the
        # gains for halving the hole — and THOSE FIGURES ARE THE LIQUIDATE ARM, which this lane no
        # longer declares. They are kept because they are what he decided on and because the
        # reproduction of them in this repo is what validated the harness; the action in force is
        # EXIT_ONLY, for the semantic reason recorded at the field itself. Measured in-runner over 8 independent
        # 60-session periods at this lane's own fill hour, on a 20k sleeve:
        #
        #     no view    +317% compounded    $83,392    worst drawdown -40.1%   needs +67% to recover
        #     50d view   +220%               $64,013    worst drawdown -25.5%   needs +34%
        #
        # So $19,379 of $63,392 in gains, for halving the hole. The +317% is not collectable: a lane
        # 41% underwater gets switched off by its operator, correctly, and that is the argument --
        # survivability, not return. Every arm here COSTS return; none of them is free.
        #
        # WINDOW 50 IS MEASURED, NOT ROUND. Summed return against no view: 63d -78.5pp, 50d -42.8pp,
        # 20d -71.7pp. Faster is WORSE -- fast views whipsaw, and the 20-day arm sat in cash 30% of
        # sessions on average, missing recoveries as well as falls. Horizon was never the defect.
        #
        # A LEVEL, NOT A COUNT, and that is the whole finding. This lane already ships a valve --
        # residual cash to GLD when fewer than `portfolio_size` names carry positive momentum -- and
        # in production it is INVERTED: mean cash weight 0.01 while the book is >30% below its high,
        # against 0.44 while within 5% of it. Near the top it sits in cash; at the bottom it is
        # fully invested. An arm at the SAME 63-day horizon, asking whether the index sits below its
        # own average, cut the worst period's drawdown from -40.1% to -25.5%. A count of names still
        # carrying positive momentum stays high through a decline, because after a run-up ten still
        # qualify. A count made of the falling things cannot tell you they are falling.
        #
        # DECLARING ONLY. Nothing acts on this yet: platform issue 873's poller is what turns a declaration
        # into an exit, and until it exists this lane trades without the view while research
        # measures it with one. `test_backtest_enforced_config_reaches_the_live_runner` records that
        # gap by name rather than letting it be silent -- the same lane's `stop_loss_portfolio_frac`
        # is already on that list, so this is the second rule research verifies and production does
        # not run. #873 requires a SHADOW month before anything is armed.
        market_view=MarketViewConfig(
            signal=MarketSignal.INDEX_VS_MA, window=50, dwell=1,
            # EXIT_ONLY, AND THE REASON IS SEMANTIC RATHER THAN NUMERICAL.
            #
            # 2026-09-11, naming the three conditions:
            #     emergency_exit   "market explodes, get out"   LIQUIDATE   act NOW
            #     entries_blocked  "today is not the day"       EXIT_ONLY   bear, stop adding
            #     self_assessment  "I'm a loser, help me"       STAND_DOWN  the lane is not working
            #
            # A 50-day moving average is a BEAR DETECTOR. Wiring it to LIQUIDATE is a bear detector
            # driving an emergency action — the wrong pairing whatever a backtest says, which is
            # why this change needs no number and is not waiting on one.
            #
            # IT WAS LIQUIDATE "because that is the arm that was MEASURED", and that reason was
            # true and is now obsolete. EXIT_ONLY was not merely unmeasured, it was UNMEASURABLE:
            # `runner_qc27_verified` read `.liquidates` and never `.blocks_entries`, so both
            # EXIT_ONLY arms reproduced the control byte for byte, 29,285 identical fills (#170,
            # fixed). Every figure published for this view is a LIQUIDATE arm because LIQUIDATE
            # was the only action the runner implemented.
            #
            # BOTH WINDOWS, so nobody later reads this as having been justified on the good one:
            #
            #     8 x 60 sessions, 2024-08-21..2026-09-10 (worst drawdown -40.1%)
            #         no view +317.0%   EXIT_ONLY +391.8% (+74.8pp)   LIQUIDATE +220.1% (-96.9pp)
            #     2025-12-01..2026-09-10, 195 sessions (worst drawdown -29.4%)
            #         no view +125.4%   EXIT_ONLY +116.3% ( -9.1pp)   LIQUIDATE  +32.2% (-93.2pp)
            #
            # EXIT_ONLY beats LIQUIDATE on BOTH. Whether the view is worth arming AT ALL is a
            # different question and is the operator's: over 25/26 it costs return on every lane that can
            # run it (MOMENTUM -1.84pp, BCTROT -4.71pp, TECHIVOL -9.05pp) for almost no drawdown.
            # A bear-market mechanism pays in a window containing a bear market, and 25/26 does
            # not contain much of one. See research/market-view/FINDINGS-all-lanes.md.
            #
            # Still DECLARING ONLY — nothing acts on this until #873's poller exists.
            action=MarketAction.EXIT_ONLY),
    )


def live_notes() -> dict[str, str]:
    """Operator-facing caveats. Deliberately code rather than prose in a document, because the
    thing that ships is what gets read."""
    return {
        "cash_proxy_never_opens":
            "GLD sits at 0% at the end of BOTH selloffs in the test window, in all three cadences. "
            "It only fills the shortfall when fewer than 10 names carry positive 63-day momentum, "
            "and after a run-up ten still qualify mid-decline. Do NOT size this position believing "
            "the cash proxy is a defensive valve -- in the measured window it never opened.",
        "universe_not_point_in_time":
            "Tech membership comes from a current sectors.parquet snapshot, not point-in-time "
            "membership. Measured survivorship effect 1.91%.",
        "cost_coverage":
            "The measured spread model covers ~19.4% of the tech universe but 65.3% of the names "
            "actually traded; uncovered names are mostly microcaps the top-100 liquidity filter "
            "never ranks.",
        "market_view_declared_not_armed":
            "The 50-day market view on this config is DECLARED, not armed. No poller reads it, so "
            "the lane trades today with no view at all while every research number is measured "
            "with one -- the second rule on this lane in that state, after "
            "stop_loss_portfolio_frac. It costs 31% of the gains and buys the worst drawdown down "
            "from -40.1% to -25.5%; kumo-trading-platform issue 873 requires a SHADOW month before it is armed. "
            "Do NOT read a backtest of this lane as describing what it currently trades.",
        "one_window_no_holdout":
            "All of the above is one 20-month window with no holdout, and the cadence mechanism "
            "rests on one V-shaped selloff and one sustained one.",
    }
