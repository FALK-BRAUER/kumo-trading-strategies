"""BCTROT — the BCT rotation deciding three times a session: open, midday and close.

    issue 32. Runs ALONGSIDE MOMENTUM-002 and replaces it over time, with no cutoff
    (2026-08-16). Two strategies, two books, two order id tags — not a reconfiguration of the
    live one.

WHY IT IS A SUBCLASS AND NOT A COPY
-----------------------------------
Everything that makes a decision is identical: same ledger-provider pool, same engine, same exit evaluator,
same reconciliation. Only the identity and the schedule differ. A forked adapter would double the
surface on which live and backtest can drift apart, and that drift is the defect class this repo
keeps paying for — the give-back rule, the ATR rules and PEAK's highs each diverged between drivers
before anyone noticed.

So the schedule is a PARAMETER of the existing strategy, not a new implementation of it. The
`decision_slots` this passes go through `momentum_rotation.slots`, the same resolver the backtest
uses, so "midday and close" means the same instants on both sides by construction.

WHAT THE SCHEDULE IS AND WHAT IT IS NOT
---------------------------------------
`("open+5m", "open+150m", "close-20m")` — 09:35, 12:00 and 15:40 on a normal session, clipped
correctly on a half day.

THE 09:35 SLOT WAS ADDED BACK ON 2026-09-04 (the operator), AND IT IS THE SLOT THIS RESEARCH REJECTED.
Recorded plainly because the rest of this docstring argues against it, and a reader who finds the
schedule disagreeing with the reasoning above it should find the disagreement acknowledged rather
than quietly edited out.

What the original finding actually says: midday+close beat 09:35 on COST, not on signal — 09:35 pays
roughly 2.8x the midday half-spread ($11.56/fill against $4.10) and buys the opening auction print.
That cost is unchanged and is now paid on the morning slot.

What is new is a rule that only exists at an open slot. `min_abs_gap_pct` declines an entry whose
overnight gap sits inside a dead band, and the flat middle it removes is 45% of candidates and the
place the return disappears (`research/residual-gate/FINDINGS-gap-entry.md`: +0.75 sharpe at the
live configuration, holding across every other config tested). It is implementable only where the
runner already knows the gap when it submits, which is the 09:35 decision and nowhere else.

So the morning slot is not a reversal of the earlier result — it is that result plus a filter the
earlier result did not have. Whether the filter pays for the spread is the open question, and the
first week of live data answers it. Watch morning fills against midday fills, per slot.

It is the only arm that survived this research cycle. It beat live 09:35 in ALL FOUR measurements
(two signal lags x two period schemes) with no sign flips, where every exit-rule change either
flipped sign or was retracted. It was chosen over midday-alone on LAG STABILITY rather than summed
return: the two are a coin on return (27.71 vs 27.86), but midday+close moves 3.21 points across
the signal-lag switch against midday's 7.09, and lag sensitivity is precisely what invalidated
everything else.

It is NOT a signal claim, and that is probably why it survived. Trading at 09:35 pays roughly 2.8x
the midday half-spread ($11.56/fill against $4.10) and buys the opening auction print. Moving later
removes a cost and a gap rather than predicting a return, so it does not depend on the entry
signal's age — the exact dependence that destroyed the rest.

Exits are also evaluated twice a session rather than once, so a position is not left unmanaged from
one midday to the next.

EXIT RULES ARE MOMENTUM-002'S, DELIBERATELY UNCHANGED
-----------------------------------------------------
`give_back_frac=0.5` and nothing else. Every alternative measured this cycle failed: PEAK in both
regimes, the 4/6/8 threshold band, `take_profit_atr`, `cluster_exit_score`, and `gb=0.15` which was
retracted as a daily-runner artefact. The hard stop passed its pre-registered criterion but only
with the give-back removed, and removing it failed sign-stability on its own — one proven component
and one unproven one, which is not a package to ship.

Changing the schedule and the exit rules together would also make the live comparison against
MOMENTUM-002 useless: two changes, one number. The schedule is the only thing that varies.
"""

from __future__ import annotations

from kumo_strategies.strategies.momentum_rotation.nautilus import MomentumRotationStrategy

EXTERNAL_ID = "BCTROT"
STRATEGY_NAME = "BCTROT"
STRATEGY_LABEL = "BCT rotation (open + midday + close)"

#: THERE IS NO DEFAULT TAG, and that is the point. An earlier version defaulted to "003" because
#: cockpit had reserved it for BCTROT at the time — but cockpit then allocated 003 to QC345 and gave
#: BCTROT 004, and this default became a claim on another strategy's tag. Two strategies sharing an
#: `order_id_tag` does not degrade: Nautilus raises at `Trader.add_strategy` and the node does not
#: boot. Only cockpit passing the tag explicitly kept it bootable.
#:
#: The tag is cockpit's — it is the only place that sees every strategy in ONE trader. A default here
#: is this repo guessing at an allocation it does not own, and a wrong guess is a node that will not
#: start. So the caller must pass it. Safe to require because BCTROT has never filled; MOMENTUM keeps
#: its default only because it holds positions keyed to `{instrument}-MOMENTUM-002`.

#: 12:00 and 15:40 on a normal session. Resolved by `momentum_rotation.slots`, the same code the
#: backtest uses — a slot that resolved differently on the two sides would make every cadence result
#: unverifiable rather than merely wrong.
DECISION_SLOTS: tuple[str, ...] = ("open+5m", "open+150m", "close-20m")

#: THE OPEN SLOT WAS ADDED THIRD (2026-09-04), and the order matters to nothing except a
#: reader: slots fire on their own anchors and the journal's unique index is
#: `(strategy_id, session, slot)`, so three decisions a day file as three rows rather than
#: colliding — the failure `_j` was built to prevent when the slot was being dropped on 31 call
#: sites.
#:
#: It also puts BCTROT on the same anchor as MOMENTUM-002, deliberately. Both lanes rank the same
#: pool, so on most mornings they will want the same names. That is not a new collision: instrument
#: ownership is already exclusive (`build_bctrot_strategy` — any instrument given to both stops the
#: node booting) and BCTROT claims no external orders. What it does mean is that the two lanes now
#: compete for the same fills at the same minute, which is a REAL change to the live comparison and
#: is the thing to watch in the first week.
#:
#: `min_abs_gap_pct` applies at ALL THREE slots, not just this one: the overnight gap is a property
#: of the day, so every decision reaches the same verdict. Note what has and has not been measured —
#: `runner_verified` models ONE decision per session and `runner_cadence` never applies the gap
#: rule, so "+0.75 sharpe" is a single-open-slot number and declining a 12:00 or 15:40 entry on the
#: morning's gap is untested anywhere.


class BCTRotationStrategy(MomentumRotationStrategy):
    """MOMENTUM's engine on a midday+close clock, under its own identity."""

    EXTERNAL_ID = EXTERNAL_ID
    LABEL = STRATEGY_LABEL

    def __init__(self, *args, order_id_tag: str, strategy_name: str = STRATEGY_NAME,
                 decision_slots: tuple[str, ...] = DECISION_SLOTS,
                 read_slots: object | None = None,
                 symbols: list[str] | None = None,
                 history_days: int | None = None,
                 shadow_only: bool = False, **kwargs) -> None:
        """Named explicitly rather than left to `**kwargs`.

        The three parameters that make this a different strategy from MOMENTUM are exactly the three
        a caller most needs to see in the signature, and a bare `*args, **kwargs` hid them: cockpit
        could not tell from the signature that `order_id_tag` was honoured at all.

        `read_slots` IS THE FOURTH, AND IT RECURRED AS THE SAME DEFECT ONE PARAGRAPH BELOW ITS OWN
        LESSON. db25413 added the live slot re-reader (kumo-trading-platform issue 514) to the momentum base, and
        `**kwargs` did forward it — but cockpit passes it only to adapters whose SIGNATURE accepts it
        (`slot_reader._live_reread_kwargs`), because this repo is pinned by revision and an unknown
        kwarg raises `TypeError` at build, taking every other lane down with it (#377). Forwarding
        without naming is therefore invisible to the only caller that matters, and BCTROT-004 kept
        its build-time schedule with a green suite. Measured on the running engine 2026-08-25:
        `MomentumRotationStrategy accepts read_slots: True`, `BCTRotationStrategy: False`.

        BCTROT is the lane with TWO slots, so it is the one where a schedule edit matters most.

        `shadow_only` IS THE FIFTH, NAMED FOR THE SAME REASON AND NOT A DIFFERENT ONE. It reaches the
        base's `rebalance_band` guard (#184), and a lane that cannot be built in shadow is a lane
        whose guard cannot be tested or bypassed by the caller who owns the decision. Forwarded
        through `**kwargs` it would work in Python and be invisible to cockpit, which passes a kwarg
        only to adapters whose SIGNATURE accepts it — the exact defect the paragraph above records.

        `strategy_name` is the one that was missing entirely. It reaches `StrategyConfig.strategy_id`,
        which under NETTING keys every position as `{instrument}-{strategy_id}` — so it is the
        cycle-attribution key and it is permanent from the first fill.

        `history_days` IS THE SIXTH, AND IT IS THE SAME DEFECT AGAIN — recurring one paragraph below
        its own lesson, for the second time. The base's `on_start` calls `request_bars(bt, start)`
        with `start` derived from `self._history_days`, and `**kwargs` forwards the value perfectly
        in Python. But cockpit passes a kwarg only to adapters whose SIGNATURE accepts it, so
        BCTROT-004 never received one, `self._history_days` stayed None, the request was guarded off
        and THE LANE REQUESTED NO HISTORY AT ALL. Measured on ibkr-paper's 16:19Z boot of 2026-09-11:
        RequestBars by lane — 28 MANUAL, 0 BCTROT. It has been trading on whatever bars accumulated
        live since each boot.

        `order_id_tag` is REQUIRED. See the module note: defaulting it made this repo claim an
        allocation cockpit owns, and the value it claimed had since been given to QC345.
        """
        # `symbols` AFTER `*args`, which is where it belongs. `f(symbols=x, *args)` is legal Python
        # and means `f(*args, symbols=x)` anyway — but it reads as though the keyword binds first,
        # and it raises `got multiple values for 'symbols'` the moment a caller passes four
        # positional arguments, since `symbols` is the fourth parameter on the base.
        super().__init__(*args, symbols=symbols, strategy_name=strategy_name,
                         order_id_tag=order_id_tag, decision_slots=decision_slots,
                         read_slots=read_slots, history_days=history_days,
                         shadow_only=shadow_only, **kwargs)
