"""Sizing divides ACCOUNT EQUITY by an INSTRUMENT PRICE, and nothing asserts they are one unit.

MIRROR OF A kumo-trading-platform ASSERTION, HELD DELIBERATELY AT BOTH ENDS (agreed 2026-08-24). Cockpit
carries its own `test_sizing_basis_hazard.py`. This is not a copy of it: a seam asserted from one side
only lets a change on the other side close or reopen the hazard with nothing in the reader's repo
giving them a reason to look. Two derivations of one fact are a detector — identical when they should
differ means a dead mechanism, differing when they should match means a live defect. Keep both.

THE FACT, verified by cockpit from the running node's own Nautilus object rather than inferred:

    AccountState(account_id=INTERACTIVE_BROKERS-DUPTEST01, account_type=MARGIN,
                 base_currency=None, is_reported=True,
                 balances=[AccountBalance(total=1_000_000.00 SGD, locked=9_067.46 SGD, ...)])

`base_currency=None` means Nautilus treats the account as MULTI-CURRENCY and converts nothing, so
there is no layer beneath this runner quietly making the units agree. Cockpit's publish seam
(`engine_node.py:5435`) then emits bare floats — `{"equity": 999215.3, ...}` — and `AccountDTO`
declares no currency field, so `AccountBalance(... SGD)` goes in and an unlabelled number comes out.
By the time it reaches `broker_equity()` (qc27_rotation.py:513, `float(v)`) the unit is gone.

There is no currency handling anywhere in this executor: grep -ri 'currency|SGD|USD|base_ccy|fx' over
`runtime/executor/` returns nothing. It is only sound while every account's base currency happens to
equal its instruments'.

THE OPERATOR RULED ON THE SGD ACCOUNT, 2026-08-24, AND THIS FILE IS NOT AN OPEN DEFECT: "it is paper. all
good. ibkr is in sgd that is unrelated to techvol." DUPTEST01 being SGD is an INTENDED property of the
paper account, not a misconfiguration, and nothing here should be read as saying otherwise. The
composition kumo-trading-platform and I originally worried about — the uncapped qc27 lane meeting the SGD
account — CANNOT OCCUR: TECHIVOL-005 is `QC27_ENABLED: false` on ibkr-paper-retired. The lanes that DO use
the account-equity fallback there are MOMENTUM-002 and BCTROT-004, both on `pgrunner`, both clipped by
`max_position_notional` — the masking the regime test below names and scopes.

WHAT THE RULING DOES AND DOES NOT SETTLE, because they are different claims and only one was ruled on:
  settled      the DENOMINATION. An SGD paper account is intended and is not a bug to chase.
  not settled  that `equity / px` mixes units with nothing asserting they match. That is structural,
               it is still true, and it is what these tests measure.
"CURRENTLY UNREACHABLE" WAS WRONG WHEN THIS FILE FIRST SAID IT (corrected 2026-08-24, same day).
kumo-trading-platform measured the deployed pin one lane wider and found MOMENTUM-002 on ibkr-paper-retired was
allocated 0, registered, armed, and TRADING — not disabled. Its RUNBOOK said an absent lifecycle row
means DISABLED; `momentum.py._lifecycle` says the opposite, "AN ABSENT ROW IS TRADING (2026-08-19)". Both of us had been calling that instance unexercised because `exec_action_log` was
empty, and an empty journal means nothing has decided YET, not that nothing can.

What it actually did on that session, both lanes measured together:

    BCTROT-004     hold 8 · enter 8    ~9,900 a name   off its 100,000 ALLOCATION
    MOMENTUM-002   hold 8 · enter 8   ~19,900 a name   off the 999,215 SGD ACCOUNT  (~159,000 total)

Exactly double, and the second row is the `be244d9` fallback firing on live config: 999,215 * 0.80 / 8
= 99,921 a name, clipped to 20,000 by `max_position_notional`. So the masking this file describes was
not a thought experiment — it was the only thing standing between a zero-allocation lane and a
six-figure book, on the instance we had both written off. Note it is pgrunner, so the cap DID bind; on
`qc27_runner` nothing would have absorbed it (see the uncapped-lane test below).

Closed by cockpit with an explicit `DISABLED` row rather than a deploy — an explicit row beats the
default — verified `submitted: 0, state: DISABLED`, BCTROT unchanged.

SO THE HONEST READING, replacing the one this paragraph used to give: the denomination is settled and
is not a bug to chase. The unit mixing is structural and still true. And the account-equity fallback
that carries it to an order was REACHABLE IN PRODUCTION, was reached, and was closed by configuration
rather than by code. What makes it unreachable today is a lifecycle row, not an impossibility. This
file is the detector for the day that row changes. It still carries no urgency for the DENOMINATION —
but "unreachable" is a claim about configuration, and configuration is exactly the thing that moved.

BLAST RADIUS IS NARROWER THAN IT LOOKS, and the narrowing is the useful half:

  sizing          AT RISK. The ONLY consumer that divides equity by a PRICE, so it is the only place
                  a unit mismatch survives into a quantity.
                  SCOPE: the regime tests below measure `PgSessionRunner` (MOMENTUM-002, BCTROT-004).
                  `QC27SessionRunner` (TECHIVOL-005) sizes on a DIFFERENT path with no notional clip,
                  so the masking described below does NOT protect it — see
                  `test_qc27_has_no_notional_cap_so_NOTHING_absorbs_the_error_there`.
  daily-loss halt NOT at risk. `pgrunner:520-534` takes `start` and `now` from the SAME equity source
                  and compares `now < start * (1 - frac)`. A ratio of two figures in one currency is
                  unit-free whatever that currency is.
  allocated path  NOT at risk. An `allocated_equity` resolved from cockpit's settings is a figure a
                  human chose in the instrument currency. `broker.equity()` is the only path that
                  injects the ACCOUNT's base currency.

So this is not a separate workstream from the `None` fallback documented in `be244d9`/`0b5a591`: it is
that same hole with a second failure mode stacked on it — the fallback is wrong in SIZE (the whole
account rather than the sleeve) AND wrong in UNIT. A required-allocation state closes both at once.
"""

from __future__ import annotations

import asyncio as aio

import pytest

from kumo_strategies.runtime.executor.lifecycle import Lifecycle, State
from kumo_strategies.strategies.momentum_rotation.runner import PgSessionRunner
from kumo_strategies.runtime.executor.runner import RiskLimits
from kumo_strategies.strategies.momentum_rotation.config import MomentumRotationConfig

#: A rate, not a quote. The defect is a MULTIPLICATIVE error of exactly the FX rate, so the assertions
#: below are exact in this constant and do not depend on what SGDUSD actually printed on any day.
SGD_PER_USD = 1.30

#: The instrument price. US equities quote in USD; the account above reports in SGD.
PX_USD = 100.0


def _fixture(*, equity: float, allocated: float | None):
    sent = []

    class Broker:
        def submit(self, req):
            from kumo_strategies.runtime.executor.broker import OrderResult
            sent.append(req)
            return OrderResult(True, "id", "detail", req)

        async def exit(self, req):
            return self.submit(req)

        def positions(self):
            return {}

        def equity(self):
            """A BARE FLOAT. The unit was dropped two seams upstream and cannot be recovered here."""
            return equity

        def last_price(self, sym, **_):
            return PX_USD

    class Jrn:
        strategy_id = "MOMENTUM-002"

        async def write(self, *a, **k):
            return 1

        async def tail(self, n=400, kind=None, symbol=None):
            return []

    class Runner(PgSessionRunner):
        async def _save_state(self, sym, st, qty=None):
            return None

        async def _drop_state(self, sym):
            return None

    r = Runner(strategy_id="MOMENTUM-002", pool=None, journal=Jrn(),
               lifecycle=Lifecycle(State.TRADING, "armed"), cfg=MomentumRotationConfig(),
               broker=Broker(), limits=RiskLimits(allocated_equity=allocated, book_size=8))
    return r, sent


def _qty(*, equity: float, allocated: float | None = None) -> int:
    r, sent = _fixture(equity=equity, allocated=allocated)
    aio.run(r._submit("2026-08-04", ("AAA",), (), {}, {"AAA": PX_USD}, State.TRADING))
    assert sent, "fixture produced no entry — every assertion below would be vacuous"
    return sent[0].qty


# -- the defect, measured -------------------------------------------------------------------------
#: Equity small enough that `max_position_notional` does NOT bind. See the regime test below for why
#: this matters: at kumo-staging's actual size the cap clips both readings to the same number and the
#: FX error is invisible.
UNCAPPED_SGD = 100_000.0

#: kumo-staging's real reported figure. slot = 1_000_000.00 * 0.80 / 8 = 99_921, far above the 20_000
#: cap, so this regime is cap-bound rather than equity-bound.
STAGING_SGD = 1_000_000.00


def test_an_SGD_account_oversizes_by_EXACTLY_the_fx_rate_WHEN_THE_CAP_DOES_NOT_BIND():
    """Two derivations of one position size, differing when they must match.

    Same account, same price, same weights — the only difference is whether the equity figure is the
    SGD number the venue reported or the USD value it represents. A correct sizing path is INDIFFERENT
    to that choice, because it would convert. This one is not: the quantities differ by exactly the
    rate, and the SGD reading is the larger. Over-sizing is the dangerous direction — it builds a
    bigger book than the research measured, and every intermediate number looks plausible.
    """
    sgd_reading = _qty(equity=UNCAPPED_SGD)                    # what the venue reports, taken as-is
    usd_truth = _qty(equity=UNCAPPED_SGD / SGD_PER_USD)        # what that is actually worth

    assert sgd_reading > usd_truth, (
        "the two readings agreed — either sizing gained a conversion (update this file) or the "
        "fixture stopped varying the basis, which would make every assertion here vacuous")
    assert sgd_reading == pytest.approx(usd_truth * SGD_PER_USD, rel=2e-2), (
        f"over-size is not the plain FX factor: {sgd_reading} vs {usd_truth} × {SGD_PER_USD}. The "
        f"defect has changed shape and this file no longer describes it")


def test_at_STAGING_SIZE_pgrunners_notional_cap_HIDES_the_fx_error_and_the_cap_is_unit_free():
    """WHY NOBODY WOULD HAVE SEEN THIS ON STAGING, and why that is not reassuring.

    Found while writing the test above: at 1_000_000.00 the equity-derived slot is ~99_921, so
    `max_position_notional` (20_000.0) binds and BOTH readings clip to the identical quantity. The FX
    error does not reach the order — the cap absorbs it. Two derivations agreeing where the mechanism
    under them is dead: the basis is not being handled correctly here, it is being OVERRIDDEN by a
    constant that happens to sit below both answers.

    That makes the cap load-bearing for a job it was never given, and the cap is a bare float with no
    currency either (`RiskLimits.max_position_notional`). If it means USD 20k against an SGD account,
    the real ceiling is ~USD 15.4k — tighter, so safe today, but tighter by an accident nobody chose
    and nobody can see. Raise the cap, or shrink the account, and the FX error stops being absorbed.

    This is the reason the hazard survives observation: on the one stack where the basis is wrong, the
    number that would reveal it is being clipped before anyone can compare it to anything.
    """
    sgd_reading = _qty(equity=STAGING_SGD)
    usd_truth = _qty(equity=STAGING_SGD / SGD_PER_USD)

    assert sgd_reading == usd_truth, (
        "the cap no longer binds at staging size — this regime test has stopped describing staging, "
        "and the FX error is now reaching orders there")
    capped = int(RiskLimits().max_position_notional / PX_USD)
    assert sgd_reading == capped, (
        f"sized to {sgd_reading}, not to the {capped} the notional cap implies — the masking "
        f"mechanism described here is not the one actually running")


def test_qc27_has_no_notional_cap_so_NOTHING_absorbs_the_error_there():
    """THE MASKING IS A PROPERTY OF `pgrunner`, NOT OF THE PLATFORM. Caught by kumo-trading-platform reading
    the two sizing paths against each other after I claimed the cap absorbed the FX error (2026-08-24).

    `max_position_notional` has exactly two mentions in `src/`: its definition on `RiskLimits`, and
    `pgrunner.py:1227`'s `min(slot * scale, self.limits.max_position_notional)`. `qc27_runner` never
    reads it — its entry sizing is `notional = equity * weight` with no clip at all.

        MOMENTUM-002, BCTROT-004  -> pgrunner     -> capped -> both bases clip to the same quantity
        TECHIVOL-005              -> qc27_runner  -> uncapped -> nothing absorbs anything

    So on the uncapped lane the FX error arrives at full size AND so does the whole-account fallback:
    999_215 × 0.80 / 8 ≈ 99_921 a name, with only cockpit's `budget_gate` in the way — and that gate
    is absent from the IBKR order path entirely.

    What actually stood between TECHIVOL and that, for its whole life, was that it never HAD a None
    allocation — because a hardcoded `ALLOCATED_EQUITY = 20_000.0` in cockpit was filling the hole.
    The dead knob was doing the uncapped lane's safety. Resolving from settings without stopping
    absent on cockpit's side would have opened exactly this.

    Asserted on the AST rather than by running qc27's sizing: a behavioural test needs a full decision
    fixture and would go quiet the moment the fixture stopped producing entries, which is the failure
    mode `test_an_SGD_account_oversizes...`'s own guard clause exists to catch. The structural claim —
    THIS RUNNER NEVER CONSULTS THE CAP — is the one that must not silently become false.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[3] / "strategies" / "qc27_tech_inverse_vol" / "runner.py"
    tree = ast.parse(src.read_text())
    read = {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Attribute) and n.value.attr == "limits"}

    assert "max_position_notional" not in read, (
        "qc27_runner now reads max_position_notional — the uncapped-lane finding above is stale and "
        "this file must be re-derived")
    assert "allocated_equity" in read, (
        f"the AST walk found no allocation read at all, so its 'not in' result proves nothing: {read}")


def test_the_allocated_path_is_NOT_exposed_and_that_is_what_makes_this_narrow():
    """The claim that limits the blast radius, asserted rather than argued.

    `allocated_equity` is a figure chosen in the instrument currency, so sizing off it cannot inherit
    the account's base currency NO MATTER WHAT the account reports. If this ever goes red, the account
    figure has started leaking into the allocated path and the hazard is no longer confined to the
    fallback — which would also invalidate "a required-allocation state closes both at once".

    Sized below the cap on purpose, so the assertion rests on the allocation rather than on the
    clipping described above.
    """
    on_sgd_account = _qty(equity=STAGING_SGD, allocated=50_000.0)
    on_usd_account = _qty(equity=STAGING_SGD / SGD_PER_USD, allocated=50_000.0)

    assert on_sgd_account == on_usd_account, (
        f"the account's basis reached a lane sized off its own allocation: {on_sgd_account} vs "
        f"{on_usd_account}")


# -- the guard that does not exist ----------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason=(
    "No currency information reaches this runner at all: cockpit's AccountDTO declares no currency "
    "field and broker_equity() returns float(v). There is nothing here to compare, so a mismatch "
    "CANNOT be detected — this is the hazard stated as the assertion that would close it."))
def test_sizing_REFUSES_when_the_equity_basis_cannot_be_shown_to_match_the_price():
    """STRICT xfail, so it is a detector in both directions.

    Red today: no unit information exists on either side of the division. It flips to XPASS — turning
    the suite red and forcing this file to be re-derived — the moment someone gives the account
    snapshot a currency and teaches sizing to refuse a mismatch. That is the intended outcome, and a
    strict xfail is the only form that NOTICES the fix instead of silently continuing to pass.

    Deliberately NOT written as "assert the runner has a currency attribute": that is satisfied by
    adding an unused field. It asserts the BEHAVIOUR — a refusal to size — which an unused field
    cannot fake.
    """
    r, sent = _fixture(equity=1_000_000.00, allocated=None)
    aio.run(r._submit("2026-08-04", ("AAA",), (), {}, {"AAA": PX_USD}, State.TRADING))
    assert sent == [], (
        "sized against an account whose basis is unknown and unverifiable against the price")
