"""One QC27 session, with the controls. Spec shape follows `pgrunner.PgSessionRunner` (#33).

THE ORDER OF OPERATIONS IS THE SAFETY MODEL, so it is fixed here rather than left to callers:

  1. bail if this session already decided. The journal is the idempotency key.
  2. read the lifecycle. SHADOW computes and publishes; only TRADING may submit entries.
  3. decide with the PURE functions the backtest uses. No second implementation.
  4. journal the decision WITH its basis, BEFORE any order is sent.
  5. re-read the lifecycle immediately before submitting — the state captured at step 2 can be
     seconds old, and an operator who HALTs in that window must not watch it submit anyway.
  6. submit, journaling intent before the broker sees it.

WHY THIS LIVES IN kumo-trading-strategies AND NOT COCKPIT. QC345's equivalent gateway lives cockpit-side
(`backend/strategies/qc345.py`) because `PgSessionRunner` is momentum-specific and could not be
reused. But every control that matters — `Lifecycle`, `PgJournal`, `DuplicateDecision`,
`RiskLimits` — is defined HERE. Putting the runner here makes those controls testable in the repo
that owns them, and leaves cockpit only the wiring it genuinely owns: settings resolution, the
budget gate, and DB session construction. Cockpit passes those in.

WHAT COCKPIT MUST STILL SUPPLY, and the runner refuses to guess:
  * `lifecycle` — the operator's current state, read from cockpit's store
  * `recheck_state` — an async re-read for step 5
  * `budget_gate` — cockpit's `may_submit`, because two derivations of one limit disagree and the
    disagreement looks like a strategy quietly holding more than it was granted
  * `limits.allocated_equity` — this strategy's allocation, NOT the account's. Sizing off the whole
    account is how a 20k strategy on a 100k account builds two oversized positions where the
    research measured ten.

    "REFUSES TO GUESS" IS TRUE OF 0.0 AND FALSE OF `None` (clarified 2026-08-24). A 0.0 now means
    zero and the lane declines to enter. A `None` still guesses, and it guesses the ACCOUNT — so a
    caller that manages this lane but failed to resolve its allocation gets the exact sizing this
    field exists to prevent, silently. Nothing here can tell that apart from "no allocation
    configured", which is what every running config passes.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field

import pandas as pd

from kumo_strategies.runtime.executor.broker import Broker, DryRunBroker, OrderRequest
from kumo_strategies.runtime.executor import daily_loss
from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
from kumo_strategies.runtime.executor.pgjournal import (
    DECISION, ERROR, ORDER, RISK, DuplicateDecision, PgJournal)
from kumo_strategies.runtime.executor.runner import RiskLimits, SessionResult
from kumo_strategies.strategies.qc27_tech_inverse_vol import (
    OPEN_OFFSET_MINUTES, QC27TechInverseVolConfig, build_feature_panel, decide, rebalance_dates,
)

DECISION_SLOT = f"open+{OPEN_OFFSET_MINUTES}m"
"""One decision per rebalance session, so one slot; the cadence itself is `cfg.rebalance_period`.

DERIVED from `OPEN_OFFSET_MINUTES` rather than written out, because it was the literal `"open+5m"`
while the shipped offset is 150 minutes. The slot name is not decoration: it is the IDEMPOTENCY KEY
(`2025-06-02/open+150m` survives a restart where a wall-clock timestamp cannot be told apart from a
genuinely new decision), and cockpit PARSES it to schedule the alert — `qc345.py`'s
`_open_offset_from_settings` reads exactly this `open+Nm` shape. A slot that names 09:35 while the
strategy fills at 12:00 schedules the decision at the wrong time and labels the journal with an hour
it never traded at."""


@dataclass
class QC27SessionRunner:
    """One session. See the module docstring: the ORDER of operations is the safety model."""

    journal: PgJournal
    lifecycle: Lifecycle
    cfg: QC27TechInverseVolConfig
    broker: Broker = field(default_factory=DryRunBroker)
    limits: RiskLimits = field(default_factory=RiskLimits)
    strategy_id: str = "TECHIVOL-005"
    read_state: object = None
    """async () -> State. THE AUTHORITATIVE PER-SESSION READ, and the one cockpit must supply.

    `lifecycle` is a value object captured when the runner is CONSTRUCTED — which happens once, at
    node boot. Live, that would freeze the operator's state for the lifetime of the process: a HALT
    written on Tuesday would not be seen on Wednesday, and the strategy would keep trading against
    an instruction given days earlier. `QC345SessionGateway._lifecycle` reads a fresh row every
    session for exactly this reason.

    Left None for tests and backtests, where `lifecycle` is already the whole truth."""

    recheck_state: object = None
    """async () -> State. Re-read of the operator's lifecycle, called immediately before submitting.
    Without it a HALT only takes effect next session.

    DEFAULTS TO `read_state` when that is supplied. A caller who wires the per-session read and
    forgets this one would otherwise get a runner that reads state freshly at the top and then
    submits without ever looking again — the last-look silently absent on the only path that has
    real money behind it."""
    daily_loss_armed: bool = False
    """Does THIS lane's daily-loss stop actually fire? Separate from the threshold, deliberately.

    `RiskLimits.daily_loss_frac` DEFAULTS TO 0.05 and cockpit constructs this runner with a bare
    `RiskLimits()`. So wiring the stop without this flag would have armed TECHIVOL-005 at 5% the
    moment it deployed — silently making the decision that `RiskLimits.daily_loss_frac`'s own
    docstring reserves for an operator: it arms a stop on a lane that has never had one, and on a
    shared account another lane's drawdown trips it (kumo-trading-platform issue 517).

    A live default that quietly turns on a trading halt is a wrong answer with good manners. The
    number lives in `limits` where every other threshold lives; the DECISION to arm this lane lives
    here, and it is one flag.

    Unarmed is NOT silent: `daily_loss.enforce` still runs every session and journals that the lane
    cannot halt itself. That is the difference between this and the opt-in-and-dead shape of
    `max_price_age_seconds`, which nothing ever set and nothing ever mentioned.
    """

    on_halt: object = None
    """async (reason) -> None. Persist a HALT where the OPERATOR's state actually lives.

    `lifecycle` is a value object captured at construction, and `_read_state` ignores it entirely
    when cockpit supplies a reader. So `lifecycle.halt()` alone mutates an object nothing reads next
    session: the durable stop degrades to a one-session decline, and the lane resumes tomorrow into
    the same drawdown. None is correct for tests and backtests, where `lifecycle` IS the whole truth.

    IT NEEDS AN ADAPTER — do NOT pass cockpit's `save_if_unchanged` directly. The signatures do not
    meet, and the mismatch fails on every halt, forever:

        enforce                  await on_halt(reason)                        1 positional arg
        save_if_unchanged(sm, strategy_id, life, expected)                    4 required args

    Passed raw it raises TypeError inside enforce, which journals "HALT could not be PERSISTED" and
    continues — loud, and permanently one-session. Wrap it.

    THE ADAPTER MUST ALSO CAPTURE `expected` BEFORE `enforce` RUNS. By the time this hook is called
    `lifecycle.halt()` has already happened, so reading the state here returns HALTED and the
    compare-and-set has nothing to compare against. `momentum.py:228` is the working pattern.

    Return None or True on success. Anything else — cockpit's OPERATOR_WON or CONTRADICTION, which
    are RETURNED rather than raised — is treated as a failure to persist and journalled as such,
    because a returned failure that nothing inspects is how an unpersisted halt reads as persisted.
    """

    budget_gate: object = None
    """async (symbol, notional) -> (bool, reason). Cockpit's own `may_submit`, passed in rather than
    reimplemented — a second derivation of one limit disagrees with the first, and the disagreement
    looks like a strategy quietly holding more than it was granted."""

    #: The slot the CURRENT session is filing under. A class-level default because terminal rows
    #: arrive on order events, which can land before any session has run in this process — a restart
    #: mid-flight is exactly when that happens. `run()` overwrites it per session.
    _slot: str = DECISION_SLOT

    async def run(self, panel: pd.DataFrame, session: str, jobs: object | None = None,
                  slot: str = DECISION_SLOT, opens: dict | None = None) -> SessionResult:
        """`jobs`, `slot` and `opens` are accepted by EVERY runner even where unused — the
        SessionRunner protocol. `opens` joins them (2026-09-05): the gap dead band needs TODAY's open, which the
        panel cannot carry because it is trimmed to sessions strictly before the one being decided.
        Unused here, accepted because the protocol is "every runner takes every shape".

         On 2026-08-17 an adapter began passing `slot=` and its paired gateway did not accept
        it: every session died on the call with `TypeError: run() got an unexpected keyword argument`,
        nothing journalled it, and two trading days were lost behind 41 and 75 rows of ordinary
        chatter. Accepting the known superset means an adapter can pass either without knowing which
        runner it is wired to.

        NOT `**kwargs`. Tolerating anything would swallow a typo — `slott=` would be silently dropped
        and the behaviour it asked for would never happen. The superset is explicit so an UNKNOWN
        kwarg still fails loudly, and `test_session_runner_protocol` catches a new one at the seam.

        `jobs` is momentum-specific (pool refresh) and unused here.
        """
        st = await self._read_state()

        # The slot every journal write in this session is filed under, set BEFORE anything can
        # write. Same shape as `pgrunner._j`: the helpers below journal too, and they have no `slot`
        # in scope, so a parameter read only where it is convenient is a parameter half-read.
        self._slot = slot
        # 1. IDEMPOTENCY FIRST. A restart mid-session must not decide the same slot twice.
        if await self.journal.decided_this_session(session, slot=self._slot):
            return SessionResult(session, st.value, False,
                                 blocked=f"{session}/{self._slot} already decided")

        # 2. LIFECYCLE. SHADOW still decides and publishes; it simply does not act.
        if not st.decides:
            await self.journal.write(RISK, f"state {st.value} does not decide", session=session,
                                     slot=self._slot)
            return SessionResult(session, st.value, False, blocked=f"state {st.value}")

        # 2b. DAILY-LOSS STOP. Before the decision, and before the cadence check: a HALT is a
        #     durable state change and must be recorded on a non-rebalance day too.
        #
        #     THIS LANE COULD NOT HALT ITSELF ON RISK UNDER ANY CONDITION (kumo-trading-platform issue 548).
        #     `pgrunner` enforced `daily_loss_frac` and this runner read one of RiskLimits' ten
        #     fields, which was not this one. The worse half was that it left no trace: "had no
        #     reason to halt" and "has no ability to halt" produced the same journal — nothing — so
        #     `enforce` writes a row when the limit is unset instead of returning quietly.
        #
        #     TECHIVOL-005's THRESHOLD IS NOT SET HERE. Arming a stop on a lane that has never had
        #     one is a strategy decision (see `RiskLimits.daily_loss_frac`), and on a shared account
        #     another lane's drawdown can trip it (kumo-trading-platform issue 517). The mechanism runs every
        #     session and says so; the number is the operator's.
        if st.may_submit_entries:
            verdict = await daily_loss.enforce(
                journal=self.journal, lifecycle=self.lifecycle,
                write=lambda k, m, detail=None: self.journal.write(
                    k, m, session=session, detail=detail, slot=self._slot),
                equity=self.broker.equity, session=session, on_halt=self.on_halt,
                frac=self.limits.daily_loss_frac if self.daily_loss_armed else None)
            if verdict.action is daily_loss.Action.BLOCK:
                return SessionResult(session, st.value, False, blocked=verdict.reason)
            if verdict.action is daily_loss.Action.HALT:
                return SessionResult(session, State.HALTED.value, False,
                                     blocked="daily loss limit breached — strategy halted")

        # 3. NOT A REBALANCE DAY is not a failure. Delegated to the PURE function the backtest uses,
        #    so cadence cannot drift between research and production.
        if panel.empty:
            return SessionResult(session, st.value, False, blocked="empty panel")
        day_ts = pd.Timestamp(session)
        if day_ts not in set(rebalance_dates(panel["date"], self.cfg.rebalance_period)):
            return SessionResult(session, st.value, False, blocked="not a rebalance session")

        # 4. THE PURE DECISION. Same functions, same config, as the backtest.
        try:
            scored, diag = build_feature_panel(panel, self.cfg)
            day = scored.loc[scored["date"] == day_ts]
            if day.empty:
                return SessionResult(session, st.value, False, blocked="no feature rows")
            held = set(await self._held_symbols())
            d = decide(day, self.cfg, held)
        except Exception as exc:                                       # noqa: BLE001
            await self.journal.write(ERROR, f"decision failed: {type(exc).__name__}: {exc}",
                                     session=session, slot=self._slot)
            return SessionResult(session, st.value, False, blocked=f"decision failed: {exc}")

        # 5. JOURNAL THE DECISION BEFORE ANY ORDER, with its basis, so "why did it buy X" is
        #    answerable months later without re-running anything. A DuplicateDecision here means a
        #    concurrent run already claimed this slot — stop rather than submit a second book.
        basis = {"hold": list(d.hold), "enter": list(d.enter), "exit": list(d.exit),
                 "weights": {k: round(v, 6) for k, v in d.weights.items()},
                 "cash_proxy_weight": round(d.cash_proxy_weight, 6),
                 "scores": {k: round(v, 6) for k, v in d.scores.items()},
                 "eligible_name_days": diag.eligible_name_days,
                 # ANCHORS THE DAILY-LOSS CHECK for the next session. Without this key
                 # `daily_loss.baseline` finds nothing on every session and the halt above can never
                 # fire — a mechanism wired to a producer that feeds it nothing, which is the
                 # built-never-executed shape it was added to end.
                 #
                 # Through `anchor()`, which OMITS the key when the equity is not a number, so the
                 # invariant holds in both directions: present if and only if usable (#111).
                 **daily_loss.anchor(self.broker.equity)}
        try:
            await self.journal.write(
                DECISION,
                f"{st.value}: hold {len(d.hold)} · enter {len(d.enter)} · exit {len(d.exit)}",
                session=session, detail=basis, slot=self._slot)
        except DuplicateDecision as exc:
            return SessionResult(session, st.value, False, blocked=str(exc))

        # 6. LAST LOOK BEFORE MONEY MOVES.
        moved = await self._state_changed(session, st)
        if moved is not None:
            return SessionResult(session, st.value, True, blocked=f"lifecycle moved to {moved}",
                                 detail=basis)

        exited, entered, submitted, refusals = await self._submit(d, session, st)
        return SessionResult(session, st.value, True, entered=tuple(entered), exited=tuple(exited),
                             held=tuple(sorted(held)), submitted=submitted,
                             blocked="; ".join(refusals) or None, detail=basis)

    # -- internals ---------------------------------------------------------------------------
    async def _read_state(self) -> State:
        """Fresh if cockpit supplied a reader, otherwise the constructed value object."""
        if self.read_state is None:
            return self.lifecycle.state
        return await self.read_state()

    async def _held_symbols(self) -> list[str]:
        """WHAT THIS STRATEGY OWNS, never the account's book.

        This called `broker.positions()` — every open position the node knows about — so TECHIVOL-005's
        first live decision (2026-08-21) proposed exiting AEM, AMGN, BDX, BETA, CGAU, WHD, WPM and XLV:
        the entire BCTROT and MOMENTUM book, eight names it does not own. Nothing reached the broker
        that session, so it cost nothing; had it submitted, it would have flattened two other
        strategies' positions.

        `strategy_positions()` is the same read narrowed by Nautilus's own NETTING attribution, and is
        what QC345's gateway uses in all four of its ownership reads.
        """
        try:
            return sorted(self.broker.strategy_positions().keys())
        except Exception:                                              # noqa: BLE001
            return []

    async def _state_changed(self, session: str, st: State) -> str | None:
        # Falls back to `read_state`, so wiring the per-session read is enough to get BOTH looks.
        recheck = self.recheck_state or self.read_state
        if recheck is None:
            return None
        try:
            now = await recheck()
        except Exception as exc:                                       # noqa: BLE001
            # Cannot confirm the operator's intent, so do not act on a stale copy of it.
            await self.journal.write(RISK, f"could not re-read lifecycle: {exc} — not submitting",
                                     session=session, slot=self._slot)
            return st.value
        if now is st:
            return None
        await self.journal.write(
            RISK, f"lifecycle moved {st.value} -> {now.value} while the session ran — the decision "
                  f"stands but nothing is submitted under a state the operator has left",
            session=session, detail={"was": st.value, "now": now.value}, slot=self._slot)
        return now.value

    async def _submit(self, d, session: str, st: State):
        """EXITS FIRST. Selling before buying frees both cash and the venue's share reservation
        before anything asks for them, so a rotation cannot be half-applied into a shortfall —
        left holding what it meant to sell and unable to buy what it meant to hold."""
        exited, entered, refusals, submitted = [], [], [], 0

        for sym in sorted(d.exit):
            if not st.may_submit_exits:
                refusals.append(f"exit {sym}: state {st.value} may not exit")
                continue
            ok, detail = await self._send(sym, "SELL", session)
            (exited.append(sym) if ok else refusals.append(f"exit {sym}: {detail}"))
            submitted += int(ok)

        if not st.may_submit_entries:
            # LIQUIDATING exits and never enters. Reached here rather than skipped earlier so the
            # exits above still run — a wind-down that could not sell is the worst of both.
            return exited, entered, submitted, refusals

        # ZERO IS NOT ABSENT. This was `or`, so `allocated_equity=0.0` — a lane granted nothing —
        # fell through to the ACCOUNT's book and sized every entry off the full account: 99,400 of a
        # 100,000 account in the test that now holds this. Found 2026-08-24 from kumo-trading-platform's
        # report that on kumo-staging BCTROT-004 holds 100000 and every other allocation is 0, so
        # every unfunded lane there was one decision away from trading BCTROT's capital. It had not
        # bitten only because nothing on that stack has ever decided.
        #
        # `None` still falls through, unchanged — that is what every running config passes, and it
        # means "cockpit configured no allocation", not "cockpit allocated zero".
        alloc = self.limits.allocated_equity
        equity = self._account_equity() if alloc is None else float(alloc)
        for sym in d.enter:
            weight = d.weights.get(sym, 0.0)
            notional = equity * weight
            if notional <= 0:
                continue
            if self.budget_gate is not None:
                allowed, why = await self.budget_gate(sym, notional)
                if not allowed:
                    refusals.append(f"enter {sym}: {why}")
                    continue
            ok, detail = await self._send(sym, "BUY", session, notional=notional)
            (entered.append(sym) if ok else refusals.append(f"enter {sym}: {detail}"))
            submitted += int(ok)
        return exited, entered, submitted, refusals

    def _account_equity(self) -> float:
        """0.0 when the account cannot be read as a finite number, so sizing degrades to inaction.

        NON-FINITE IS NOT A NUMBER, and `x or 0.0` does not catch one: **NaN is truthy**, so the `or`
        never fires. A nan returned here makes `notional` nan, survives the `notional <= 0` skip
        (every comparison with a nan is False), and raises `ValueError: cannot convert float NaN to
        integer` inside `_qty` PART WAY THROUGH the entry loop — after earlier symbols have already
        been submitted. A half-executed rotation is worse than none.
        """
        try:
            value = float(self.broker.equity() or 0.0)
        except Exception:                                              # noqa: BLE001
            return 0.0
        return value if math.isfinite(value) else 0.0

    async def _send(self, symbol: str, side: str, session: str, notional: float | None = None):
        """FAIL CLOSED. The intent is journalled before the broker sees it: if that write fails we
        have not traded, and resume — which decides what was attempted from those records — would
        otherwise send it again. No record, no order."""
        qty = self._qty(symbol, side, notional)
        if qty <= 0:
            return False, "quantity resolved to zero"
        wrote = await self.journal.write(
            ORDER, f"{side} {qty} {symbol}: submitting", session=session, symbol=symbol,
            detail={"phase": "intent"}, slot=self._slot)
        if wrote is None:
            await self.journal.write(ERROR, f"{side} {symbol}: intent not journalled — not sending",
                                     session=session, symbol=symbol, slot=self._slot)
            return False, "intent not journalled"
        # AN EXIT GOES THROUGH `exit()`, NEVER `submit()`. `exit()` calls cockpit's
        # `release_for_exit`: it suppresses protection, cancels the SPECIFIC resting stop, and waits
        # for the venue to confirm the shares are actually free. `submit()` does none of that, and a
        # plain SELL against a position a protective stop already reserves is refused with
        # `403 insufficient qty available (available: 0)` -- "available" means UNRESERVED is 0, not
        # that the position is 0. FSM and VCTR both hit exactly this (issue 48,
        # kumo-trading-platform issue 358), and `NautilusBroker.exit`'s own docstring has said so since.
        #
        # This runner was written afterwards and routed both sides through `submit()`. It is not a
        # conditional failure: every held symbol at Alpaca currently reports qty_available = 0, 100%
        # reserved by resting stops, so every exit these lanes attempted was structurally refused.
        #
        # ENTRIES stay on `submit()` -- routing one through `exit()` would suppress protection on a
        # position being opened.
        req = OrderRequest(symbol=symbol, side=side, qty=qty, session=session,
                           strategy_id=self.strategy_id, slot=self._slot)
        res = await self.broker.exit(req) if side == "SELL" else self.broker.submit(req)
        ok = bool(getattr(res, "ok", False))
        # `.detail`, not `.reason` — OrderResult is (ok, order_id, detail, request). Reading a field
        # that does not exist made every refusal read "refused", discarding the one thing that says
        # WHY: "not a subscribed instrument", "no instrument definition cached".
        detail = getattr(res, "detail", "refused")
        await self.journal.write(ORDER if ok else ERROR, f"{side} {qty} {symbol}: {detail}",
                                 session=session, symbol=symbol,
                                 detail={"phase": "result", "ok": ok}, slot=self._slot)
        if ok and side != "SELL":
            await self._write_claim(symbol, side, qty, session)
        return ok, detail


    async def sync_claim(self, symbol: str, qty, px=None) -> bool:
        """The claim follows the venue's answer, not the submit (#133, kumo-trading-platform issue 829).

        Called from an adapter's order-event handlers, including
        `RegistrationMixin.protective_close` — a protective stop that fills is this lane's position
        closing, and the claim has to follow it down.

        THIS METHOD'S ABSENCE WAS THE DEFECT. The adapters reach it through
        `getattr(runner, "sync_claim", None)`, which cannot tell "missing" from "working", so on
        every runner in this package the re-sync was silently inert while cockpit's gateways had it.
        `test_sync_claim_exists_on_the_real_runners` pins it on the shipped class for that reason.
        """
        from kumo_strategies.runtime.executor.store import sync_claim_to

        return await sync_claim_to(self.journal, self.strategy_id, symbol, qty, px)

    async def _write_claim(self, symbol: str, side: str, qty: int, session: str) -> None:
        """Record or release this lane's ownership claim (kumo-trading-platform issue 540).

        THIS LANE NEVER WROTE ONE. `exec_position_state` had rows only for the two lanes that run a
        trailing stop, so TECHIVOL-005 contributed nothing to `pgrunner._foreign_claims` — and
        `own_ceiling(acct, mine, other)` narrows sizing by exactly that term. Measured on alpaca-paper
        2026-08-25: DELL was QC345 7 + TECHIVOL 2 with neither claiming, so any lane that DID claim
        DELL computed `own_ceiling(acct, mine, 0)` = `mine`. The attribution term vanished precisely
        when there were two other holders. The guard was not weakened; it was silently skipped.

        ON ACCEPT, NOT ON FILL, matching `pgrunner`'s entry path (`:1261-1264`, `if r.ok`). `ok` means
        accepted for submission, so this can claim shares that never fill — and that is the SAFE
        direction: an over-claim only narrows other lanes' ceilings, while an under-claim lets them
        size into a position we hold. Consistency between lanes matters more here than precision.

        `quality="adopted"`, NOT `live`, and that differs from pgrunner deliberately. pgrunner writes
        LIVE because it observed the entry AND keeps updating the peak every session. This lane has
        no trailing stop and never revisits the row, so a LIVE quality would assert a trustworthy
        peak frozen at the entry price — worse than admitting the peak is unknown.

        NEVER RAISES INTO THE ORDER PATH. The order is already accepted at the venue by the time this
        runs; failing here must not turn a live position into an exception. A missing claim is the
        defect this fixes, so it is journalled loudly rather than swallowed.
        """
        from kumo_strategies.runtime.executor.store import record_claim

        try:
            # Entries claim on accepted submit. Exits do not release on accepted submit: Nautilus
            # acceptance is not a venue fill, and a later rejection would otherwise orphan the live
            # position. The adapter's terminal event sync owns the release/quantity adjustment.
            if side == "SELL":
                return
            # `last_price`, NOT `price` — the latter exists on no broker, and reading it here
            # would raise into the `except` below and lose the claim silently on every entry.
            px = float(self.broker.last_price(symbol))
            await record_claim(self.journal, self.strategy_id, symbol, qty, px)
        except Exception as exc:                                       # noqa: BLE001
            # THE FAILURE WRITE IS ITSELF GUARDED. Without this the contract above is a lie: a claim
            # failure whose journal write ALSO raises escapes into the order path, and the order is
            # already accepted at the venue by then — turning a live position into an exception for a
            # bookkeeping reason (found by 2026-08-26). The log is the last resort, and it
            # cannot fail the same way because it touches no database.
            try:
                await self.journal.write(
                    ERROR, f"{side} {symbol}: claim not recorded — {type(exc).__name__}: {exc}",
                    session=session, symbol=symbol,
                    detail={"phase": "claim", "ok": False}, slot=self._slot)
            except Exception as inner:                                 # noqa: BLE001
                # THE LAST RESORT IS ITSELF GUARDED, and the first version was not. `print` raises
                # `BrokenPipeError`, `OSError` or `ValueError` when stdout is closed, replaced or
                # broken, and `flush=True` adds a second failure point — so the fallback protecting
                # the "never raises into the order path" contract could violate it (codex,
                # 2026-08-26). Nothing follows this: if even the write to stdout fails there is
                # nowhere left to report, and raising here would abandon a rotation with orders
                # already at the venue.
                try:
                    print(f"CLAIM LOST {self.strategy_id} {side} {symbol}: {exc!r}; "
                          f"journal also failed: {inner!r}", flush=True)
                except Exception:                                      # noqa: BLE001
                    pass

    def _qty(self, symbol: str, side: str, notional: float | None) -> int:
        if side == "SELL":
            # OUR quantity, not the account's. Reading `positions()` here would submit an exit for the
            # FULL account holding of a symbol another strategy also owns -- selling their shares to
            # flatten ours, which is precisely the failure `strategy_positions` was written to end.
            try:
                return int(abs(self.broker.strategy_positions().get(symbol, 0)))
            except Exception:                                          # noqa: BLE001
                return 0
        try:
            # `last_price`, NOT `price` -- the latter exists on no broker. The bare
            # `except Exception: return 0` below turned that AttributeError into "0 shares", so
            # TECHIVOL-005 journalled `sizing yielded 0 shares at 492.73` against a price the caller
            # had already fetched, and it read like a sizing bug for a day.
            px = float(self.broker.last_price(symbol))
        except Exception:                                              # noqa: BLE001
            return 0
        return int(notional / px) if px > 0 and notional else 0

    async def record_terminal(self, session: str, symbol: str, ok: bool, detail: str,
                              *, client_order_id: str | None = None,
                              filled_qty: int = 0, status: str | None = None,
                              side: str | None = None) -> None:
        """The venue's actual answer, called from the adapter's order-event handlers.

        The `phase="result"` row above is submit-time only — Nautilus accepting the order, not the
        venue filling it. This is the row that arrives later, when it is actually known. Present from
        the start here because its ABSENCE is what left MOMENTUM with 22 fills and zero terminal
        rows on 2026-08-20: the caller was wired for weeks while no gateway defined the method, so
        every call no-opped through a getattr (kumo-trading-platform issue 383)."""
        await self.journal.write(
            ORDER if ok else ERROR,
            f"terminal: {symbol} {'filled' if ok else 'rejected'} — {detail}",
            session=session, symbol=symbol,
            # PER-ORDER OUTCOME (#164). See pgrunner.record_terminal for the full reasoning: a
            # terminal row carrying only {ok, phase} is not attributable to an order, which forced
            # `attempts_for` to key by symbol and let a late reject from an older order on the same
            # symbol re-arm the id of a filled one.
            correlation=client_order_id,
            detail={"phase": "terminal", "ok": ok,
                    "client_order_id": client_order_id,
                    "filled_qty": int(filled_qty or 0),
                    "status": status,
                    "side": side},
            slot=self._slot)
