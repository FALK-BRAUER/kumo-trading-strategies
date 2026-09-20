"""Shared types for the session runner.

The SQLite-backed SessionRunner that used to live here is GONE. It was a second, reachable way to
run the strategy with different behaviour — no shrink guard, an in-memory give-back trail, no
Postgres idempotency — and `--live` drove it, so the one path that could touch a real broker was
the one missing every protection. There is now exactly one runner: `pgrunner.PgSessionRunner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskLimits:
    allocated_equity: float | None = None
    """The capital THIS strategy is allowed to deploy. None means "use the broker's account equity",
    which is today's behaviour and what every running config does.

    It exists because sizing had no idea what a strategy was allocated. `slot` was
    `broker.equity() * max_deployed_frac / max_positions`, so a strategy with a 20k target on a 100k
    account sized every entry as though it owned all 100k — roughly 10k a name. The budget gate then
    refused the order at submission. The result is not "buys smaller" or "buys fewer", it is "buys
    the same and gets rejected", and once it does fit under target it rebuilds as TWO oversized
    positions where the strategy is designed for eight.

    That matters most for a strategy being deployed off a backtest: the research measured an
    8-name equally-weighted book of the strategy's OWN capital, and a live book of 2 names sized off
    the whole account is a different portfolio with ~4x the single-name concentration — not the one
    that was tested.

    Setting it makes the sizing agree with the allocation, so the gate becomes a backstop against
    mistakes rather than the mechanism that shapes the book.
    """

    book_size: int = 8
    """The researched number of positions, and the SIZING basis:
    `slot = equity * max_deployed_frac / book_size`.

    Split from `max_positions` on 2026-08-21 because one number was doing two jobs and the two wanted
    different values. MOMENTUM-002's researched book is n_hold=8 and its limits are a bare
    `RiskLimits()`, so the cap was ALSO 8 — target book size equal to the hard cap, leaving zero
    rotation headroom. It sold N and had every one of the N replacements refused by the names it had
    just sold, because the cap counts what is held right now and deliberately does not deduct sells
    submitted moments ago. 19 entries were lost that way across seven sessions from 2026-08-06, each
    journalled as a tidy "position cap" while the book shrank and the strategy looked healthy.

    Raising the cap was not available as a fix precisely BECAUSE of the conflation: the same number
    divides the equity slot, so a bigger cap silently halves every position. Splitting them is what
    makes the cap adjustable at all, and it leaves sizing exactly where the research put it."""

    max_positions: int = 12
    """HARD CEILING on open positions. Must EXCEED `book_size`, or the strategy cannot rotate.

    Bigger than the book on purpose: a rotation sells N and buys N, and the sold names have not
    vacated yet when the buys are sized. The headroom is what the replacements occupy while the sells
    settle. It is still a cap — an unbounded book is what it exists to prevent."""

    max_position_notional: float = 20_000.0
    max_deployed_frac: float = 0.80
    daily_loss_frac: float = 0.05
    """Halts the strategy when the account is down this much on the session — enforced in
    pgrunner, before any decision.

    THE BASIS IS THE ACCOUNT, NOT THE LANE, and that is the fact the open decision turns on
    (recorded 2026-08-25, kumo-trading-platform issue 517). Both sides of the comparison are `broker.equity()` —
    account net liquidation — so a lane halts on a drawdown in the SHARED account even when its own
    book is flat or up. On alpaca-paper five lanes share ~103k, so a 5% move driven entirely by
    MANUAL-001's positions halts the systematic lanes with it.

    IT HALTS, IT DOES NOT MERELY DECLINE. `lifecycle.halt()` is a durable state change: the lane
    stops and stays stopped until an operator clears it. This is not entry suppression for one
    session.

    BASELINES DIFFER PER LANE. `_session_start_equity` reads each lane's OWN journal for the first
    DECISION row of the session, so a lane that decides at open+5m and one that decides at open+150m
    anchor on different account values, and on a falling day they cross the threshold at different
    times — or one crosses and the other does not.

    WHO ENFORCES IT: every runner with a journal, a lifecycle and limits, through
    `daily_loss.enforce` — one derivation, asserted by `test_every_runner_can_halt_on_daily_loss.py`.
    It used to be `pgrunner` alone, so MOMENTUM-002 and BCTROT-004 had it and TECHIVOL-005 had no
    automatic loss stop at all (kumo-trading-platform issue 548). `momentum_rotation.broker_equity` calls this
    limit "the only automatic stop this strategy has".

    ARMING IT IS STILL A STRATEGY DECISION. Wiring the mechanism is a defect fix; choosing a
    threshold for a lane that has never had one is not, and TECHIVOL's is deliberately unset. An
    unarmed lane now JOURNALS that it cannot halt, once per session, because "had no reason to halt"
    and "has no ability to halt" otherwise leave the same record — nothing.

    A NON-FINITE EQUITY ON EITHER SIDE REFUSES THE SESSION rather than passing the check. #111: the
    recorded baseline was unguarded while the live read was not, so a `Decimal("NaN")` anchor made
    `now < NaN * frac` False and a measured 99% fall did not halt."""
    max_source_age_hours: float = 36.0
    max_price_age_seconds: float | None = None
    """Reject a live price older than this when sizing or evaluating exits, instead of accepting
    any non-null value regardless of age. `None` (default) is today's behaviour exactly.

    Was flipped to 1800.0 for one night (2026-08-19) and reverted the same night: cockpit found the
    bar-cache fallback in `last_price` untested against a live REPRODUCTION of what this strategy
    actually subscribes to -- with a bound set, the bare `cache.price()` path is skipped and every
    lookup falls through to the bar cache, and whether that path actually returns a fresh price
    for every symbol was never verified against a running node, only against a fake. Setting a real
    default here risks turning "skipped, no live price to size against" into every entry of every
    session, which is a worse failure than the dead mechanism it was meant to fix. `_bar_type` DOES
    exist on `MomentumRotationStrategy` (confirmed directly in the running container) -- an earlier
    claim otherwise was wrong -- but that alone does not establish the bar path returns fresh data
    for every subscribed symbol, and this needs a live-node reproduction before it defaults on
    again, not another fake."""
    min_bar_coverage: float = 0.80
    """Refuse to decide when bars are missing for too much of the pool. Ranking only what happens
    to have data narrows the universe silently and looks like a normal run."""
    min_rankable_frac: float = 0.50
    """Refuse to decide when too few candidates carry a SCORE, even though they have bars.

    Having rows is not the same as being rankable. Gates blank the score column, and thin history
    yields NaN -- so 40 candidates can produce 1 ranked name, and `decide()` will then exit every
    held position not in that one-name ranking. Requiring at least zero scores was not enough: one
    survivor passes it. This is the fraction of candidates that must actually rank."""


    def __post_init__(self) -> None:
        # LOUD AT CONSTRUCTION, not as a strategy that quietly stops rotating. The original defect
        # was silent for two weeks: nothing anywhere asserted that the cap must exceed the book.
        if self.max_positions <= self.book_size:
            raise ValueError(
                f"max_positions={self.max_positions} must EXCEED book_size={self.book_size}, or a "
                f"full book can sell but never re-enter — it can only shrink, and cannot rotate. "
                f"`max_positions` is the hard ceiling; `book_size` is what the strategy is sized for.")


@dataclass
class SessionResult:
    session: str
    state: str
    decided: bool
    entered: tuple[str, ...] = ()
    exited: tuple[str, ...] = ()
    held: tuple[str, ...] = ()
    submitted: int = 0
    blocked: str | None = None
    detail: dict = field(default_factory=dict)



class _FixedSource:
    """Adapts an already-resolved pool to the CandidateSource protocol the pure engine expects."""

    def __init__(self, symbols: set[str]) -> None:
        self._s = set(symbols)

    def eligible(self, d) -> set[str]:                       # noqa: ARG002 — pool is point-in-time
        return self._s
