"""The registration contract, and `is_entry` in particular (kumo-trading-platform issue 39).

`is_entry` is the only genuinely new member and the only one the platform cannot derive for itself.
Cockpit refuses ENTRIES when a strategy is over budget and always permits EXITS, so a wrong answer
here is not a cosmetic mislabel: classify a wind-down as an entry and the gate traps the strategy
above target permanently; classify a flip as an exit and an over-budget strategy can open unbounded
new exposure while calling it a wind-down.
"""

from __future__ import annotations

import inspect

import pandas as pd

import pytest

from kumo_strategies.runtime.nautilus.contract import (
    RegistrationMixin, StrategyRegistration, is_entry_order)
from kumo_strategies.strategies import _layout


def _shipped_adapters():
    """EVERY cockpit lane in this package. The single source for every conformance parametrize below.

    THE POINT OF THE WHOLE FILE. TECHIVOL-005 was built to the standard — 53 tests, mutation bites, a
    BacktestEngine harness, a trade-for-trade diff against the pandas runner — and still failed its
    first live session four separate ways, because every one of those failures was an INTEGRATION
    CONTRACT THAT WAS NEVER WRITTEN DOWN. Three separate tests in this file claimed to assert things
    "for every strategy" and named their strategies by hand; QC27 shipped after all three and was
    subject to none of them.

    A new strategy must inherit every contract here by EXISTING.

    Discovery is `strategies._layout.lanes()` — every `Strategy` subclass defined in a
    `strategies/<name>/nautilus*.py` that declares `EXTERNAL_ID` (ks#211: the lanes left
    `runtime/nautilus`, and a sweep still walking that package would find only shims). Research
    adapters (PennyGapStrategy, IntradayMomentumRotation) have no `EXTERNAL_ID`, cannot be registered
    by cockpit, and are legitimately not subject to the deployment contract; the hole that would open
    — a real lane shipping without one and escaping this suite — is closed by
    `test_a_research_adapter_cannot_quietly_BECOME_a_lane` below, derived rather than listed.
    """
    return _layout.lanes()


def _all_adapters():
    """Every Nautilus Strategy subclass here, lane or research."""
    return _layout.adapters()



@pytest.mark.parametrize("net,side,qty,expected,why", [
    (0, "BUY", 10, True, "opening a long"),
    (0, "SELL", 10, True, "opening a short is still an entry"),
    (10, "BUY", 5, True, "adding to a long"),
    (-10, "SELL", 5, True, "adding to a short"),
    (10, "SELL", 5, False, "partial reduction of a long"),
    (10, "SELL", 10, False, "flattening a long"),
    (-10, "BUY", 5, False, "partial reduction of a short"),
    (-10, "BUY", 10, False, "flattening a short — a BUY that is an exit"),
    (10, "SELL", 50, True, "flip: 10 reduced, 40 of NEW short exposure opened"),
    (-10, "BUY", 50, True, "flip the other way"),
    (10, "SELL", 0, False, "a zero-quantity order changes nothing"),
])
def test_is_entry_order(net, side, qty, expected, why):
    assert is_entry_order(net, side, qty) is expected, why


def test_any_order_that_shrinks_the_position_is_an_exit_whatever_the_side():
    """The property the wind-down depends on. An over-budget strategy must never be blocked from
    selling down, so anything reducing exposure has to classify as an exit — including a BUY."""
    for net in (-100, -7, 7, 100):
        for qty in (1, abs(net) / 2, abs(net)):
            side = "SELL" if net > 0 else "BUY"
            assert not is_entry_order(net, side, qty), f"net={net} {side} {qty} blocked a wind-down"


def test_a_flip_that_GROWS_exposure_is_an_entry():
    """This is what stops an over-budget strategy opening unbounded opposite exposure and calling it
    a wind-down — the dishonesty the platform cannot detect from outside."""
    assert is_entry_order(10, "SELL", 1000), "+10 -> -990 is 980 of new short exposure"
    assert is_entry_order(-10, "BUY", 1000)


def test_a_flip_that_SHRINKS_exposure_is_an_exit_and_that_is_deliberate():
    """Selling 11 against a +10 long leaves -1: strictly smaller exposure than it started with.

    Calling it an entry would let the budget gate block an order that reduces risk, which is the
    trapping failure the asymmetry exists to avoid. The leak is bounded by construction — the result
    is always smaller than the position it replaced, so nothing unbounded can be opened this way,
    and the test above covers the case that can be.
    """
    assert not is_entry_order(10, "SELL", 11)
    assert not is_entry_order(-10, "BUY", 11)


def test_the_contract_has_no_self_policing_member():
    """A strategy polices its allocation correctly until the day it has a bug, and then holds more
    than it was granted, silently. Enforcement is on the order path, which is cockpit's. Cockpit
    asserts the same thing from its side; this is the mirror."""
    members = set(StrategyRegistration.__protocol_attrs__)
    assert members == {"external_id", "label", "claimed_instruments", "warmup_bars", "preflight"}
    # `preflight` was added 2026-08-22 and is NOT a self-policing member: it returns `Probe(name,
    # value, error)` — OBSERVATIONS with no verdict field — and cockpit judges them. The distinction
    # is the whole design (kumo-trading-platform issue 438): a preflight that returned pass/fail would be exactly
    # what this test forbids, one layer up, reporting healthy right until the thing deciding health is
    # the thing that is broken. `test_a_Probe_carries_no_verdict_field` is what keeps that true.
    assert not any("budget" in m or "limit" in m for m in members)
    assert not any(m.startswith("check_") or m.startswith("validate_") for m in members), (
        "a member named check_*/validate_* is the strategy judging itself")


def test_external_id_carries_no_TAG_suffix():
    """The tag is cockpit's to allocate, and a name like "BCTROT-003" is what made 003 look claimed
    from this side and free from cockpit's. Two strategies sharing an order_id_tag does not degrade —
    Nautilus raises at Trader.add_strategy and the node does not boot.

    Digits in the name itself are fine and QC345 keeps them: 345 identifies the QuantConnect
    strategy, it is not an allocation. What must never appear is a trailing `-NNN`.
    """
    import re
    from kumo_strategies.runtime.nautilus import (
        momentum_rotation, momentum_rotation_intraday, qc345_rotation)
    for mod in (momentum_rotation, momentum_rotation_intraday, qc345_rotation):
        ext = mod.EXTERNAL_ID
        assert ext, f"{mod.__name__} publishes no EXTERNAL_ID"
        assert not re.search(r"[-_]\d+$", ext), f"{ext} looks like it claims a tag"


def test_the_default_tag_is_published_separately_from_the_name():
    """Cockpit allocates the tag; this default only exists so an existing deployment keeps the id it
    already has. Under NETTING the position id is `{instrument}-{strategy_id}`, so a tag that moves
    after anything has traded orphans real positions."""
    from kumo_strategies.strategies.momentum_rotation import nautilus as m
    assert m.DEFAULT_ORDER_ID_TAG == "002", "MOMENTUM-002 is live; its default must not move"
    assert m.EXTERNAL_ID == "MOMENTUM"


class _FakeCache:
    def __init__(self, net):
        self._net = net

    def net_position(self, iid):
        if self._net is None:
            raise KeyError(iid)
        return self._net


class _Strat(RegistrationMixin):
    EXTERNAL_ID = "TEST"

    def __init__(self, net=0.0, claims=None, need=7):
        self.cache = _FakeCache(net)
        self._claims = claims
        self._need = need


def test_no_strategy_side_is_entry_SURVIVES_anywhere(cls=None):
    """REMOVED 2026-08-22 (kumo-trading-platform issue 442). Required of every lane, called by nothing.

    Its stated rationale — that only the strategy knows whether an order is an entry, because a SELL
    is an exit on a long and an entry on a short — was TRUE about the arithmetic and FALSE about who
    can do it. Cockpit reads the same net position from the same Nautilus cache
    (`exec_client.py:612` -> `budget_gate.is_entry(net, side, qty)`) and never once asked a strategy.

    So the member was a required piece of the contract that existed only to be conformance-tested.
    That is worse than dead code: `RegistrationMixin` was defined TWICE in this module, and the dead
    copy's `is_entry` had an EMPTY BODY — it would have returned None and classified every order as
    not-an-entry, disabling cockpit's budget gate entirely. Nothing bound to it, so it never fired.
    A member nobody calls is a member whose breakage nobody observes.

    `is_entry_order` — the pure function — STAYS. Cockpit's gate mirrors its rule, and the two are
    meant to agree; deleting the shared statement of the rule is not what this removes.
    """
    assert not hasattr(RegistrationMixin, "is_entry"), (
        "the mixin still carries `is_entry`; cockpit stopped requiring it on 2026-08-22")
    assert "is_entry" not in set(StrategyRegistration.__protocol_attrs__)
    for lane in _shipped_adapters():
        assert not hasattr(lane, "is_entry"), f"{lane.__name__} still exposes `is_entry`"


def test_the_pure_entry_RULE_is_kept_because_cockpit_mirrors_it():
    """Deleting the member must not delete the rule. Cockpit's `budget_gate.is_entry` implements the
    same "does absolute exposure grow" test, and the two agreeing is what makes a wind-down
    un-blockable on both sides."""
    assert is_entry_order(10.0, "SELL", 5) is False, "reducing a long is an exit"
    assert is_entry_order(-10.0, "SELL", 5) is True, "adding to a short is an entry"


def test_the_mixin_exposes_the_declarative_members():
    s = _Strat(claims=["AAPL.NASDAQ"], need=254)
    assert s.external_id == "TEST"
    assert s.label == "TEST", "label falls back to the id rather than being empty"
    assert s.claimed_instruments == ["AAPL.NASDAQ"]
    assert s.warmup_bars == 254


def test_claimed_instruments_is_a_copy_not_the_live_list():
    """Claims are exclusive across the node and an over-broad one stops it booting, so a caller must
    not be able to widen them by mutating what it was handed."""
    claims = ["AAPL.NASDAQ"]
    s = _Strat(claims=claims)
    s.claimed_instruments.append("MSFT.NASDAQ")
    assert s.claimed_instruments == ["AAPL.NASDAQ"]


@pytest.mark.parametrize("cls", _shipped_adapters(), ids=lambda c: c.__name__)
def test_each_shipped_adapter_satisfies_the_registration_contract(cls):
    """Conformance asserted per adapter, since the contract is a Protocol and nothing enforces it at
    import. A strategy missing a member does not fail here — it fails in cockpit at registration.

    DISCOVERED, not listed. This was a hand-written parametrize naming momentum_rotation and
    qc345_rotation; QC27RotationStrategy shipped afterwards and was never subject to it — the third
    hand-written list in this file to miss QC27, after the broker_equity method check and the
    order_id_tag check. A new strategy must inherit every contract here by existing, not by someone
    remembering to add a row.
    """
    for member in StrategyRegistration.__protocol_attrs__:
        assert hasattr(cls, member), f"{cls.__name__} does not implement {member}"
    assert getattr(cls, "EXTERNAL_ID", None), f"{cls.__name__} declares no EXTERNAL_ID"


def test_the_order_id_tag_is_a_constructor_parameter_not_a_constant():
    """Cockpit maps external_id -> tag and must be able to instantiate at the tag it allocated.
    Without this the mapping is decorative: cockpit can map QC345 to 006 and still get a 003."""
    import inspect

    # DISCOVERED, not listed. This named two strategies and QC27 was not one of them — the same blind
    # spot that let QC27's `@property` defect ship, in the test right next door.
    #: Lanes whose `order_id_tag` still carries a default. SHRINK ONLY. Removing a default from a
    #: LIVE strategy's signature is a change to a running lane, so it is recorded rather than made
    #: silently from a test file — but a NEW lane cannot join this list, which is the point.
    known_defaulted = {"MomentumRotationStrategy": "002", "QC345RotationStrategy": "003"}

    for cls in _shipped_adapters():
        sig = inspect.signature(cls.__init__)
        p = sig.parameters.get("order_id_tag")
        assert p is not None, f"{cls.__name__} hardcodes its tag"
        if cls.__name__ in known_defaulted:
            assert p.default is not inspect.Parameter.empty, (
                f"{cls.__name__} no longer defaults its tag — delete it from `known_defaulted`")
            continue
        # PRESENT IS NOT ENOUGH — it must have NO DEFAULT. Giving it one leaves it in `parameters`,
        # so the original check passed while cockpit's allocation was silently optional. A default
        # that happens to agree with cockpit is not an allocation, and the day it disagrees the node
        # does not boot. Found by mutation-biting the template: `order_id_tag: str = '999'` survived.
        assert p.default is inspect.Parameter.empty, (
            f"{cls.__name__}.order_id_tag has a default ({p.default!r}) — cockpit allocates tags, and "
            f"a lane that can be constructed without one will eventually be")


#: The exact key set cockpit publishes on `broker.account` (engine_node.py `_publish_account`).
#: `portfolio_value` is DELIBERATELY ABSENT: cockpit reads Alpaca's `portfolio_value` and republishes
#: it under the name `equity` (exec_client.py:427), so a strategy reading `portfolio_value` off this
#: message gets None every time.
PUBLISHED_ACCOUNT = {
    "equity": 103_428.0, "cash": 12_000.0, "buying_power": 24_000.0, "multiplier": 1.0,
    "long_market_value": 91_428.0, "last_equity": 102_900.0, "ts": 1,
}


def _strategies_with_broker_equity():  # noqa: D401 — kept as an alias, see _shipped_adapters
    """Every strategy in `runtime.nautilus` that defines `broker_equity`, DISCOVERED not listed.

    A hand-written list is what let QC27 through. `test_broker_equity_is_a_METHOD...` named
    MomentumRotationStrategy and QC345RotationStrategy explicitly; QC27RotationStrategy landed on a
    branch merged afterwards, carrying the identical `@property` defect the test existed to catch,
    and the test stayed green because it had never heard of it.
    """
    import inspect

    found = []
    for m in _layout.lane_modules():
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ != m.__name__:
                continue
            if inspect.getattr_static(obj, "broker_equity", None) is not None:
                found.append(obj)
    assert found, "no strategy defines broker_equity — this test no longer describes the code"
    return found


def test_the_published_account_fixture_carries_NO_portfolio_value_alias():
    """The fixture must mirror what cockpit ACTUALLY publishes, or it stops being evidence.

    Adding `portfolio_value` here would make every test above pass while the code read a key that is
    never on the wire — the fixture would agree with the bug instead of catching it. kumo-trading-platform's
    sibling test forbids the alias at the publisher for the same reason: two names for one number IS
    the defect, and aliasing preserves it while hiding it.
    """
    assert "portfolio_value" not in PUBLISHED_ACCOUNT, (
        "the fixture grew a portfolio_value alias — cockpit does not publish that key, and adding it "
        "here makes these tests agree with the defect they exist to catch")
    assert "equity" in PUBLISHED_ACCOUNT


def test_every_known_strategy_is_actually_DISCOVERED():
    """Guards the discovery, DERIVED rather than listed.

    This named three strategies while five lanes existed — it would have passed with BCTROT or the
    template silently dropped. A hand-written floor covers the instances someone remembered on the day
    it was written, which is the exact defect it exists to prevent, one level out. kumo-trading-platform found
    the identical thing in their own `test_broker_equity_seam.py`: hardcoded to two, so QC27 was wired
    months later and the file passed while the live strategy was broken.

    So the floor is now a SECOND, INDEPENDENT DERIVATION: count the files under `strategies/` that
    define a Nautilus `Strategy` subclass, by parsing EVERY .py there rather than importing the ones
    the helper chose. Discovery imports `strategies/<name>/nautilus*.py` with `inspect`; this walks
    the whole strategies tree with `ast`, so a lane written into a file the helper does not look at
    (`live.py`, `engine.py`, a new stem) shows up here and not there. Two derivations of one fact,
    which by our own rule is a detector — if they disagree, one of them is wrong.
    """
    import ast

    on_disk = set()
    for path in sorted(p for p in _layout.strategies_root().rglob("*.py")
                       if "tests" not in p.parts and "backtests" not in p.parts
                       and "__pycache__" not in p.parts):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases}
            if "Strategy" in bases:
                on_disk.add(node.name)

    discovered = {c.__name__ for c in _all_adapters()}
    missing = on_disk - discovered
    assert not missing, (
        f"these define a Nautilus Strategy on disk but discovery does not find them: {sorted(missing)}. "
        f"The scan has narrowed, and every conformance test in this file is silently skipping them.")
    assert len(discovered) >= len(on_disk), (
        f"discovery found {len(discovered)} adapters, the filesystem has {len(on_disk)}")


def test_no_test_here_iterates_a_HAND_WRITTEN_list_of_strategies():
    """Guards the CALL SITES, not just the helper.

    `test_every_known_strategy_is_actually_DISCOVERED` proves the scan finds all three — but a
    mutation that swaps ONE test's loop for a literal tuple survives it, because the helper is still
    correct and the strategies left in the tuple are still the passing ones. That is exactly the edit
    that let QC27's `@property` through: not a broken scan, a test that stopped using it.

    Asserted over this file's own AST. A substring check would be satisfied by deleting the warning
    in this docstring and broken by writing one — see `tests/test_source_assertions_parse.py`.
    """
    import ast
    from pathlib import Path as _Path

    tree = ast.parse(_Path(__file__).read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        it = node.iter
        # `for cls in (A, B):` / `for cls in [A, B]:` over classes named *Strategy
        if isinstance(it, (ast.Tuple, ast.List)):
            names = [e.id for e in it.elts if isinstance(e, ast.Name)]
            if any(n.endswith("Strategy") for n in names):
                offenders.append(f"line {node.lineno}: {names}")
    assert not offenders, (
        "a test iterates a hand-written list of strategies instead of discovering them — this is how "
        f"QC27 was missed: {offenders}")


def test_broker_equity_reads_a_key_the_publisher_ACTUALLY_PUBLISHES():
    """THE SECOND CAUSE OF ONE SYMPTOM, found by kumo-trading-platform on 2026-08-21.

    QC345's morning failure was `broker_equity` being a property called as a method. Fixing that moved
    the failure one step later rather than removing it: at 18:30Z QC345 resumed, reached sizing, and
    refused every entry with "sizing yielded 0 shares at 492.73" — five names, zero submitted, against
    an account reading $103,428. TECHIVOL-005 failed its first live slot the same way.

    Cockpit publishes `equity` on `broker.account`; there is no `portfolio_value` key on that message,
    because cockpit reads Alpaca's `portfolio_value` and REPUBLISHES it under the name `equity`. Two
    names for one number across a repo boundary, with nothing forcing the ends to agree.

    Asserted against the real published key set, on every strategy, so a new one cannot reintroduce
    it. NOT fixed by cockpit adding a `portfolio_value` alias — that preserves the two names and
    hides the defect.
    """
    for cls in _strategies_with_broker_equity():
        strat = cls.__new__(cls)
        strat._broker_account = dict(PUBLISHED_ACCOUNT)
        fn = inspect.getattr_static(cls, "broker_equity")
        got = fn(strat)
        assert got == 103_428.0, (
            f"{cls.__name__}.broker_equity returned {got!r} from the account message cockpit actually "
            f"publishes — it is reading a key that is never on the wire")


def test_a_NON_FINITE_equity_is_REFUSED_by_every_strategy_that_reports_one():
    """A NaN equity DISARMS THE ONLY AUTOMATIC STOP, and `equity <= 0` does not catch it.

    `momentum_rotation.broker_equity` guards with `float(... or 0.0)` then `if equity <= 0: raise`.
    Neither holds against a NaN:

        float("nan") or 0.0   ->  nan          (**NaN is truthy**, so the `or` never fires)
        nan <= 0              ->  False        (every comparison with NaN is False, so no raise)

    so the method RETURNS nan. The daily-loss halt then evaluates `now < start * (1 - frac)` with
    `now = nan`, which is False, and the strategy NEVER HALTS. That is the precise failure the
    method's own docstring says it exists to prevent — "returning 0.0 would disarm it silently" — and
    the guard written for it catches the zero while the NaN walks through.

    Found 2026-08-24 sweeping for the same `or`-default trap behind the NaN-name crash in 4d46290.
    Same operator, same truthiness, third direction: a real 0 read as absent, an absent name read as
    present, and now a non-number read as a number.

    CLASS-AIMED, on the same discovery the property/method check uses. A hand-written list is what let
    QC27 through once already, and a per-strategy version of this test would be written for whichever
    lane was in mind that day.
    """
    for cls in _strategies_with_broker_equity():
        for bad in (float("nan"), float("inf"), float("-inf")):
            strat = cls.__new__(cls)
            strat._broker_account = dict(PUBLISHED_ACCOUNT, equity=bad)
            fn = inspect.getattr_static(cls, "broker_equity")
            try:
                got = fn(strat)
            except Exception:
                continue                      # refused — which is the point
            assert got is None, (
                f"{cls.__name__}.broker_equity returned {got!r} for a published equity of {bad!r}. "
                f"A non-finite equity must raise or return None; returning it disarms the daily-loss "
                f"halt, which compares against it and never fires")


def test_every_adapter_RE_READS_its_decision_slot_rather_than_capturing_it():
    """kumo-trading-platform issue 514: the slot knob was dead without a redeploy.

    Cockpit resolved `*_SLOTS` once at build (`qc27.py:316`, `qc345.py:632`) and passed a VALUE; the
    adapter captured it, and `set_time_alert(..., override=True)` re-armed from the boot-time copy
    forever. An operator could edit the slot, watch it be accepted, and see the lane keep firing at
    the old time. Verified live 2026-08-24 — nothing changed until the engine was recreated.

    Same dead-knob class as `qc27.ALLOCATED_EQUITY`, which survived the lane's entire life because
    the constant happened to equal the setting.

    CLASS-AIMED on the same discovery the other seam tests use. A hand-written list is what let QC27
    through the property/method check once already, and this is exactly the kind of test somebody
    writes for whichever adapter they had in mind that day.

    Asserted on the AST — does `_arm` CALL the re-read — rather than on the presence of an attribute,
    which an unused field would satisfy.
    """
    import ast
    import inspect

    for cls in _strategies_with_broker_equity():
        arm = inspect.getattr_static(cls, "_arm", None)
        if arm is None:
            continue                      # not every adapter schedules its own alert
        called = {n.func.attr for n in ast.walk(ast.parse(inspect.getsource(arm).lstrip()))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert called & {"_reread_slots", "_reread_open_offset"}, (
            f"{cls.__name__}._arm does not re-read its decision slot — it re-arms from the value "
            f"captured at construction, so the settings knob is dead until a redeploy. Calls: "
            f"{sorted(called)}")


def test_every_adapter_NAMES_its_slot_reader_in_its_own_signature():
    """FORWARDING THROUGH `**kwargs` IS NOT ENOUGH — cockpit has to INSPECT before it calls.

    kumo-trading-platform passes the live slot reader only to adapters whose signature accepts it
    (`slot_reader._live_reread_kwargs`), and that guard is not optional: this repo is pinned by
    revision, so passing an unknown kwarg to an older adapter raises `TypeError` at BUILD, before any
    lane registers — one strategy's construction killing every other lane, the #377 shape. It already
    fired once on a real boot with exactly this message:

        TypeError: QC27RotationStrategy.__init__() got an unexpected keyword argument 'read_open_offset'

    So an adapter that only forwards via `**kwargs` is functionally able to take the reader and is
    INVISIBLE to the caller that must check first. It silently keeps the captured-at-build schedule —
    #514 unfixed for that lane, with everything green.

    Measured 2026-08-25 on the running engine after db25413: MomentumRotationStrategy accepted
    `read_slots`, BCTRotationStrategy did not, because it defines its own `__init__` with `**kwargs`.
    BCTROT-004 is the lane with TWO slots, so it is the one where a schedule edit matters most.

    `BCTRotationStrategy.__init__`'s own docstring already states this rule for the three parameters
    above it — *"Named explicitly rather than left to `**kwargs`... cockpit could not tell from the
    signature that `order_id_tag` was honoured at all."* Same defect, same class, one parameter later.
    """
    import inspect

    for cls in _strategies_with_broker_equity():
        if inspect.getattr_static(cls, "_arm", None) is None:
            continue
        params = inspect.signature(cls.__init__).parameters
        named = {"read_slots", "read_open_offset"} & set(params)
        assert named, (
            f"{cls.__name__}.__init__ does not NAME a slot reader. Forwarding through **kwargs is "
            f"invisible to a caller that inspects the signature first, so this lane silently keeps "
            f"the schedule captured at build. Params: {sorted(params)}")


#: Base-adapter parameters a SUBCLASS deliberately leaves to `*args`/`**kwargs`. Recorded, not
#: forbidden — BCTROT's passthrough is intentional, and naming all fourteen would bury the three that
#: actually distinguish it. What must never happen is a NEW one joining this set silently.
#: `history_days` WAS RECORDED HERE AND SHOULD NOT HAVE BEEN (#200). It was accepted as a
#: deliberate passthrough, and its consequence is that BCTROT-004 — a live, trading lane — could
#: never receive a lookback from cockpit, so its inherited `on_start` guarded `request_bars` off and
#: it requested NO HISTORY AT ALL. Measured on ibkr-paper's 16:19Z boot of 2026-09-11: RequestBars by
#: lane — 28 MANUAL, 0 BCTROT.
#:
#: That is the same defect as `read_slots` in db25413, two paragraphs below, recurring on the very
#: next parameter — and this record is what let it. A passthrough is only "deliberate" if someone
#: checked what the base DOES with the parameter; this one was recorded by shape.
_PASSTHROUGH_OK = {
    "BCTRotationStrategy": {
        "account_topic", "bar_type_suffix", "calendar", "cfg", "equity_per_position",
        "external_order_claims", "instrument_ids", "instrument_type",
        "max_stale_days", "open_offset_minutes", "session_jobs", "session_runner", "source",
    },
}


def test_a_subclass_hides_EXACTLY_the_base_parameters_it_is_recorded_as_hiding():
    """A base parameter that reaches a subclass only through `**kwargs` is INVISIBLE to a caller that
    inspects the signature before calling — and kumo-trading-platform must inspect, because this repo is
    pinned by revision and an unknown kwarg raises `TypeError` at build, taking every other lane down
    with it (#377). Cockpit's `_live_reread_kwargs` therefore DROPS what a signature does not name.

    That is fail-open: the lane builds, stays green, and silently keeps the old behaviour. It already
    happened once — `read_slots` was added to the momentum base in db25413, BCTROT forwarded it
    through `**kwargs`, the call worked, every direct-construction test passed, and cockpit skipped
    the one lane with two slots (fixed in cb5ee79).

    EQUALITY, not subset, and that is the whole design: the fourteen below are known and working, so
    forbidding them would be churn against a deliberate passthrough. The failure mode is the
    FIFTEENTH — a parameter added to the base tomorrow that quietly joins the hidden set, exactly as
    `read_slots` did. Adding one to the base now fails here until it is either named on the subclass
    or recorded above as a deliberate passthrough.
    """
    import inspect

    adapters = {c.__name__: c for c in _strategies_with_broker_equity()}
    for name, cls in sorted(adapters.items()):
        bases = [b for b in cls.__mro__[1:] if b.__name__ in adapters]
        if not bases:
            continue
        own = set(inspect.signature(cls.__init__).parameters)
        base = set(inspect.signature(bases[0].__init__).parameters)
        hidden = base - own - {"self", "args", "kwargs"}
        recorded = _PASSTHROUGH_OK.get(name, set())
        assert hidden == recorded, (
            f"{name} hides a different set of {bases[0].__name__} parameters than recorded.\n"
            f"  newly hidden (name it on the subclass, or record it): {sorted(hidden - recorded)}\n"
            f"  no longer hidden (update the record):                {sorted(recorded - hidden)}")


# -- the broker/strategy seam: broker_equity ---------------------------------------------------------
def test_broker_equity_is_a_METHOD_on_every_strategy_the_broker_can_hold():
    """THE 2026-08-21 QC345 OUTAGE, as a contract rather than one patched line.

    `NautilusBroker.equity()` is `return self.strategy.broker_equity()` — it assumes a METHOD.
    MomentumRotationStrategy defines one. QC345RotationStrategy defined a `@property`, so the property
    EVALUATED first and the `()` landed on its RESULT. `broker_equity` legitimately returns None when
    no account frame has arrived, and `None()` raises:

        TypeError: 'NoneType' object is not callable

    That killed the rebalance mid-submit, AFTER the decision row was written — so the journal showed
    five entries decided and no order, no refusal, no error. QC345 has gone live twice and booked
    nothing both times.

    IT WAS LATENT UNTIL THE MASK CAME OFF. `_broker_account` was empty because kumo-trading-platform issue 382 stopped
    publishing cash under the name equity when equity could not be derived. That was the correct fix —
    the old value understated equity by 25% on the channel the daily-loss halt trusts — and it is what
    made this reachable. #382 did not cause it; it removed the wrong value that was hiding it.

    ASSERTED ACROSS ALL STRATEGIES, not just the one that broke: a per-class fix leaves the next
    strategy free to make the same choice, and this seam is silent until an account frame is missing.
    """
    import inspect

    offenders = []
    for cls in _strategies_with_broker_equity():
        attr = inspect.getattr_static(cls, "broker_equity", None)
        if attr is None:
            offenders.append(f"{cls.__name__}: no broker_equity at all")
        elif isinstance(attr, property):
            offenders.append(
                f"{cls.__name__}: broker_equity is a @property — NautilusBroker.equity() calls it, "
                f"so a None result becomes None() and raises TypeError")
    assert not offenders, "; ".join(offenders)


def test_a_missing_account_frame_yields_None_and_does_NOT_raise():
    """The behaviour the caller depends on. Cockpit's `_equity_per_position` is
    `equity = self._broker.equity() or 0.0` — it ALREADY handles None correctly and never got the
    chance, because the TypeError was raised inside `equity()` before that guard was reached.

    So the fix must keep None a legal answer. Turning it into 0.0 here would silently disarm the
    daily-loss halt, which anchors on this number and fails by never halting.
    """
    import inspect

    from kumo_strategies.runtime.nautilus.broker import NautilusBroker
    from kumo_strategies.strategies.qc345_rotation.nautilus import QC345RotationStrategy

    class _Strat:
        _broker_account = None
        broker_equity = inspect.getattr_static(QC345RotationStrategy, "broker_equity")

    broker = NautilusBroker.__new__(NautilusBroker)
    broker.strategy = _Strat()
    assert broker.equity() is None, "a missing account frame must be None, not a plausible zero"


def test_a_present_account_frame_still_returns_the_equity():
    """The working path must not break — and this test itself encoded the defect.

    It fed `{"portfolio_value": "103181.44"}` and asserted 103181.44 came back. That key is NEVER on
    the `broker.account` message, so the test was asserting the broken contract and passing: it
    described a payload the publisher does not produce, and agreed with code that read a name nobody
    writes. A double built from the same wrong assumption as the code confirms the assumption instead
    of testing it.

    Now built from PUBLISHED_ACCOUNT, the key set cockpit actually emits.
    """
    from kumo_strategies.runtime.nautilus.broker import NautilusBroker
    from kumo_strategies.strategies.qc345_rotation.nautilus import QC345RotationStrategy

    class _Strat:
        _broker_account = dict(PUBLISHED_ACCOUNT)
        broker_equity = inspect.getattr_static(QC345RotationStrategy, "broker_equity")

    broker = NautilusBroker.__new__(NautilusBroker)
    broker.strategy = _Strat()
    assert broker.equity() == PUBLISHED_ACCOUNT["equity"]


def test_a_research_adapter_cannot_quietly_BECOME_a_lane():
    """THE HOLE THE EXTERNAL_ID SPLIT WOULD OTHERWISE OPEN.

    `_shipped_adapters()` subjects only strategies declaring `EXTERNAL_ID` to the deployment contract,
    which is right — cockpit maps external_id to a tag and cannot register a strategy without one, so
    research adapters are legitimately exempt.

    But that makes "forgot to declare EXTERNAL_ID" a way to ship a lane that escapes every conformance
    test in this file. The distinguishing feature of a lane is that it can be WIRED TO COCKPIT'S
    GATEWAY: it accepts a `session_runner`, and without one the adapter decides and submits inside
    `_decide_for` with no lifecycle, journal, idempotency, risk or budget in the path.

    So the rule is derived from what the class can DO, not from a list of which is which: anything
    wireable must declare itself. Adding a research adapter needs no maintenance here; making one
    deployable fails until it is declared.
    """
    import inspect

    offenders = []
    for cls in _all_adapters():
        if getattr(cls, "EXTERNAL_ID", None):
            continue                                   # already a declared lane
        if "session_runner" in inspect.signature(cls.__init__).parameters:
            offenders.append(cls.__name__)
    assert not offenders, (
        f"these accept a `session_runner`, so cockpit can wire them as a live lane, but declare no "
        f"EXTERNAL_ID — so every conformance test in this file skips them: {offenders}")


#: Call sites that legitimately read the ACCOUNT's book, keyed by (file, enclosing function) so the
#: entry survives code movement. Each needs a reason, and the reason must be "this is not an ownership
#: decision" — reconciliation, drift, external-activity detection. Cockpit reached the same shape
#: independently: the rule is not "never call it", it is "ownership decisions may not".
#:
#: SHRINK-ONLY IN SPIRIT. Adding an entry is a claim that a read is not about ownership, and that claim
#: is exactly what was wrong in `qc27_runner`. A new gateway needing an entry is a design conversation,
#: not a formality.
_ACCOUNT_LEVEL_EXEMPT = {
    ("strategies/momentum_rotation/runner.py", "run"): (
        "RECONCILIATION, not ownership. Reads the account book in order to COMPARE it against claims "
        "and Nautilus's attribution, then narrows with min(acct_qty, claim, attributed). The account "
        "figure is the thing being reconciled against, never the answer to 'what do we own'."),
}


# -- ownership reads, across EVERY gateway ---------------------------------------------------------
def test_no_gateway_decides_OWNERSHIP_from_the_accounts_book():
    """PROMOTED FROM ONE MODULE TO THE CONTRACT.

    `broker.positions()` is the ACCOUNT's book; `strategy_positions()` is this strategy's, and under
    NETTING Nautilus already keeps the split. TECHIVOL-005 read the first in both of its ownership
    reads and, on its first live session, formed eight liquidation orders against BCTROT's and
    MOMENTUM's positions — SELL 174 CGAU, SELL 65 BDX and six more, none of which it owns. They were
    stopped by an instrument-subscription check written for an entirely different purpose.

    578fbb6 fixed it and guarded `qc27_runner` alone. That is the same mistake one level up: a defect
    found in one gateway, fenced in that gateway, while three others were free to make it. Asserted
    here over EVERY module in `runtime.executor`, so a new gateway inherits the rule.

    AST, not text: these modules explain the unsafe alternative in prose, and a substring check would
    be satisfied by deleting the explanation and broken by writing one.
    """
    import ast

    files = _layout.executor_sources()
    assert files, "executor sweep found nothing"
    offenders = []
    for path in files:
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "positions"
                        and isinstance(node.func.value, ast.Attribute)
                        and node.func.value.attr in {"broker", "_broker"}):
                    if (_layout.rel(path), fn.name) in _ACCOUNT_LEVEL_EXEMPT:
                        continue
                    offenders.append(f"{_layout.rel(path)}:{fn.name}:{node.lineno}")
    assert not offenders, (
        "these read the ACCOUNT's book to decide ownership — use `strategy_positions()`, or the "
        "native `positions_open(strategy_id=...)` it wraps. A strategy that reads the account's book "
        f"will propose, size and submit exits against other strategies' positions: {offenders}")


# -- preflight: the strategy OBSERVES, the platform JUDGES ------------------------------------------
def test_a_Probe_carries_no_verdict_field():
    """THE WHOLE DESIGN, AND THIS FILE ALREADY FORBADE THE ALTERNATIVE.

    contract.py's own principle, 2026-08-15: "THE STRATEGY DECLARES, THE PLATFORM DECIDES ...
    a strategy polices its own allocation correctly right up until the day it has a bug, and then
    holds more than it was granted, silently."

    A `preflight()` returning pass/fail is that anti-pattern one layer up: it reports healthy right
    up until the day the thing deciding health is the thing that is broken. Every defect found in the
    last 48 hours was a component wrong about itself while reporting fine.

    So a Probe carries the OBSERVED VALUE and nothing else. No `ok`, no `passed`, no `status` — if a
    field like that exists, someone eventually sets it from inside the strategy and we are back to
    self-attestation.

    TECHIVOL is the case that proves it is not pedantry: `owned=["AEM","AMGN",...]` returned eight
    symbols and NO error. A pass/fail probe passes it. It is only wrong in the light of "this strategy
    has never filled anything" — knowledge the strategy does not have and cockpit does.
    """
    import dataclasses

    from kumo_strategies.runtime.nautilus.contract import Probe

    fields = {f.name for f in dataclasses.fields(Probe)}
    assert fields == {"name", "value", "error"}, f"Probe grew fields: {fields}"
    for banned in ("ok", "passed", "status", "healthy", "verdict", "valid"):
        assert banned not in fields, (
            f"Probe.{banned} makes the strategy judge itself — contract.py forbids exactly this")


@pytest.mark.parametrize("cls", _shipped_adapters(), ids=lambda c: c.__name__)
def test_every_lane_can_be_preflighted(cls):
    """A new strategy inherits preflight by EXISTING. That is the whole point: cockpit refuses READY
    on a bad observation, and a lane that cannot be probed is a lane declared deployed on the strength
    of its constructor — which is how all four of the last 48 hours' defects reached production."""
    assert hasattr(cls, "preflight"), f"{cls.__name__} cannot be preflighted"


def test_preflight_NEVER_raises_and_reports_the_failure_instead():
    """A probe that raises tells you less than one that reports. `equity` raising and `equity`
    returning None are two different diagnoses — the @property bug and the portfolio_value bug
    respectively — and an exception escaping collapses them into "preflight crashed"."""
    from kumo_strategies.runtime.nautilus.contract import RegistrationMixin

    class _Broken:
        def equity(self):
            raise RuntimeError("no account frame")

        def strategy_positions(self):
            raise RuntimeError("cache gone")

        def last_price(self, sym, **kw):
            raise RuntimeError("no price")

    class _S(RegistrationMixin):
        EXTERNAL_ID = "TEST"
        claimed_instruments: list = []

    probes = {p.name: p for p in _S().preflight(_Broken())}
    assert probes, "preflight returned nothing"
    for name in ("equity", "owned"):
        assert probes[name].error is not None, f"{name} swallowed the failure"
        assert probes[name].value is None


def test_preflight_reports_the_VALUE_when_the_call_succeeds():
    """`equity=None`, `equity=0.0` and `equity=103428` are three diagnoses. A boolean is one."""
    from kumo_strategies.runtime.nautilus.contract import RegistrationMixin

    class _Ok:
        def equity(self):
            return 103_428.0

        def strategy_positions(self):
            return {}

        def last_price(self, sym, **kw):
            return 90.32

    class _S(RegistrationMixin):
        EXTERNAL_ID = "TEST"
        claimed_instruments: list = []

    probes = {p.name: p for p in _S().preflight(_Ok())}
    assert probes["equity"].value == 103_428.0 and probes["equity"].error is None
    assert probes["owned"].value == {}, "an untraded strategy must report owning nothing"


def test_no_class_in_this_package_is_defined_twice():
    """`contract.py` DEFINED `RegistrationMixin` TWICE (found 2026-08-22).

    The second shadowed the first, so every adapter inherited the second and the first was dead code —
    including an `is_entry` with an EMPTY BODY. Had anything bound to it, it would have returned None,
    classified every order as not-an-entry, and cockpit's budget gate would never have refused one.

    It was found only because a `preflight` added to the first silently did not exist on any strategy.
    Nothing in any suite noticed a duplicated class in the file that DEFINES the deployment contract —
    which is the same failure as everything else this week: two derivations of one thing, and the one
    that loses is invisible.
    """
    import ast
    from collections import Counter

    offenders = []
    for path in _layout.all_runtime_sources():
        tree = ast.parse(path.read_text())
        names = Counter(n.name for n in tree.body if isinstance(n, ast.ClassDef))
        offenders += [f"{path.name}:{n} x{c}" for n, c in names.items() if c > 1]
        # METHODS TOO. The first version of this guard checked classes only, and within the hour I
        # added `preflight` to RegistrationMixin TWICE — recreating, in the same file, the exact
        # defect this test was written for. It passed. A duplicated method shadows identically to a
        # duplicated class and is easier to introduce, because the second copy is out of sight.
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            m = Counter(f.name for f in cls.body
                        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)))
            offenders += [f"{path.name}:{cls.name}.{n} x{c}" for n, c in m.items() if c > 1]
    assert not offenders, (
        f"defined more than once — the later one silently wins and the earlier is dead code that "
        f"still reads as live: {offenders}")


@pytest.mark.parametrize("cls", _shipped_adapters(), ids=lambda c: c.__name__)
def test_a_republished_daily_bar_REPLACES_rather_than_appends(cls):
    """Alpaca republishes the CURRENT session's daily bar as it updates.

    Appending every arrival put ~103 copies of one date per symbol after thirteen hours of uptime,
    which gave the trailing window zero variance, produced a NaN score, and left the strategy sitting
    inert while looking healthy. Asserted across every lane rather than in one adapter's own file,
    because it is a property of the FEED and every lane consuming that feed has it.

    Skipped for a lane with no `_ingest`; the point is to cover everything that has one, not to force
    a shape on adapters that read data differently.
    """
    ingest = getattr(cls, "_ingest", None)
    if ingest is None:
        pytest.skip(f"{cls.__name__} has no _ingest")

    from nautilus_trader.model.data import BarType

    bt = BarType.from_str("AAA.XNAS-1-DAY-LAST-EXTERNAL")

    class _B:
        def __init__(self, ts, close):
            self.ts_event = ts
            self.open = self.high = self.low = self.close = close
            self.volume = 1_000
            self.bar_type = bt

    s = cls.__new__(cls)
    s._need = 10
    from collections import defaultdict, deque
    s._bars = defaultdict(lambda: deque(maxlen=64))
    day = pd.Timestamp("2026-01-05", tz="UTC").value
    ingest(s, _B(day, 10.0))
    ingest(s, _B(day, 11.0))
    # COUNT ONLY. Lanes store bars differently — MOMENTUM and BCTROT keep raw `Bar` objects, the
    # newer lanes keep dicts — so asserting `[-1]["close"]` would test a storage shape rather than
    # the property. That divergence is itself worth ending, but not by making this test demand one.
    assert len(s._bars["AAA"]) == 1, (
        f"{cls.__name__} appended a republished bar instead of replacing it — the trailing window "
        f"fills with copies of one date and every score goes NaN")



def _construct_kwargs(cls, cfg):
    """Minimal kwargs to construct any lane, DERIVED from its signature.

    Lanes differ in what else they require — QC345 takes a `source`, others do not — so a fixed
    kwarg set covers whichever lanes happened to exist when it was written. Everything required and
    unrecognised gets None, which is enough to reach a CONSTRUCTOR-TIME guard; a lane that needs more
    than that to raise has moved its guard too late.
    """
    import inspect

    kw = {"cfg": cfg, "instrument_ids": [], "order_id_tag": "999"}
    params = inspect.signature(cls.__init__).parameters
    # A LANE WITH A SHADOW ESCAPE HATCH IS BUILT IN SHADOW. Some lanes refuse to construct against a
    # runner that cannot execute their decisions — CRSISHORT's `ORDER_PATH_COMPLETE`, SMHGLD's
    # delta-execution gate — and that refusal is correct and has its own tests. These generic tests
    # are asking about something else (a price field, a cadence), so they build the lane in the mode
    # that lets them ask. Without this, adding a capability gate to any lane silently breaks every
    # generic conformance test for it, and the obvious repair is to weaken the gate.
    if "shadow_only" in params:
        kw["shadow_only"] = True
    for name, prm in params.items():
        if name in ("self", *kw) or prm.default is not inspect.Parameter.empty:
            continue
        if prm.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        kw[name] = None
    return kw



def _lanes_with_a_price_field():
    """Lanes carrying a price-field setting, and the field's name.

    PARAMETRIZED OVER THIS rather than skipping inside the test. A parametrized test that skips every
    case exits 0, so `if True: pytest.skip(...)` survived every assertion — the coverage cannot be
    checked from inside a test that is not running. With the filter in the parametrize there is no
    skip to widen, and an empty list fails here instead of passing silently.
    """
    import dataclasses
    import typing

    out = []
    for cls in _shipped_adapters():
        cfg_cls = typing.get_type_hints(cls.__init__).get("cfg")
        if not dataclasses.is_dataclass(cfg_cls):
            continue
        names = [f.name for f in dataclasses.fields(cfg_cls)
                 if "price" in f.name and "field" in f.name]
        if names:
            out.append((cls, cfg_cls, names[0]))
    assert len(out) >= 3, (
        f"only {[c.__name__ for c, _, _ in out]} carry a price-field setting — if lanes lost the "
        f"field the guard is moot, and if discovery narrowed, these tests are covering nothing")
    return out


#: A live Nautilus `Bar` carries OHLCV and nothing else. Anything else — `close_adj` above all — can
#: be produced by a backtest and NEVER by the wire.
_BAR_FIELDS = frozenset({"open", "high", "low", "close", "volume"})


@pytest.mark.parametrize("cls,cfg_cls,field", _lanes_with_a_price_field(),
                         ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x))
def test_a_price_field_no_bar_can_supply_is_REFUSED_AT_CONSTRUCTION(cls, cfg_cls, field):
    """THE DEFECT THAT KILLED QC345'S FIRST LIVE SESSION AND NEARLY KILLED QC27'S.

    `build_feature_panel` needs the configured price field as a COLUMN, and the natural default is
    `close_adj` because that is what the backtest reads. A Nautilus bar carries OHLCV and nothing
    else, so the live feed can never produce it.

    Unguarded this does not fail at startup. It fails at the FIRST REBALANCE — weeks later, in front
    of nobody — with `bars missing momentum price field 'close_adj'`. QC345 shipped exactly that shape
    (`KeyError: 'eligible'`) and booked nothing on its first live session. Refusing at CONSTRUCTION
    turns a silent latent crash into a node that will not boot, in front of whoever is deploying.

    ASSERTED ACROSS EVERY LANE THAT HAS SUCH A FIELD, discovered from the config's own dataclass
    fields. It was previously guarded in QC27's own test file only — one adapter fixing a defect and
    fencing it in that adapter, while every other lane stayed free to make it, which is the same
    mistake one level up that `578fbb6` made with the ownership read.
    """
    import dataclasses

    bad = dataclasses.replace(_build(cfg_cls), **{field: "close_adj"})
    assert bad.__getattribute__(field) not in _BAR_FIELDS, "fixture no longer sets an unsupplyable value"
    with pytest.raises(ValueError, match="close_adj"):
        cls(**_construct_kwargs(cls, bad))


@pytest.mark.parametrize("cls,cfg_cls,field", _lanes_with_a_price_field(),
                         ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x))
def test_a_price_field_a_bar_DOES_supply_constructs_fine(cls, cfg_cls, field):
    """The guard must refuse the unsupplyable value, not every value — a lane that cannot be built
    with `close` cannot be built at all."""
    import dataclasses

    good = dataclasses.replace(_build(cfg_cls), **{field: "close"})
    built = cls(**_construct_kwargs(cls, good))
    assert built is not None, "construction with a supplyable price field must succeed"


# -- the account book must NET, not accumulate ------------------------------------------------------
def test_opposing_legs_NET_to_zero_rather_than_summing_to_double():
    """THE LIVE WHD CASE, AND IT IS THE NUMBER 56 AGAIN.

    `_sum_positions` added `p.quantity` — the Nautilus MAGNITUDE, "the current open quantity", never
    negative. `p.signed_qty` is the signed one. So BCTROT's +28 WHD and MOMENTUM's -28 summed to 56
    instead of netting to 0, and the broker reports `GET /v2/positions/WHD -> 404, position does not
    exist`. The account holds none.

    kumo-trading-platform hit the identical defect in market value (#427): a SHORT reported positive and a flat
    book rendered as 56 shares held. Same symbol, same number, same cause, other repo.

    WHY IT MATTERS ON MONDAY. `positions()` feeds `acct_qty` into `own_ceiling`, so an inflated account
    makes the ceiling too PERMISSIVE:

        unsigned  own_ceiling(56, my_claim=28, other=28) = 28   sells 28 of a position that does not exist
        signed    own_ceiling(0,  my_claim=28, other=28) =  0   correct

    I derived "WHD does not self-heal" from the unsigned number and told cockpit so. They checked the
    broker rather than the engine and were right.
    """
    from types import SimpleNamespace

    from kumo_strategies.runtime.nautilus.broker import NautilusBroker

    def _p(sym, signed):
        return SimpleNamespace(
            instrument_id=SimpleNamespace(symbol=SimpleNamespace(value=sym)),
            quantity=abs(signed), signed_qty=float(signed))

    out = NautilusBroker._sum_positions([_p("WHD", 28), _p("WHD", -28), _p("BETA", 79)])
    assert out.get("WHD", 0) == 0, (
        f"opposing legs summed to {out.get('WHD')} instead of netting to 0 — a flat book reads as a "
        f"position and the ownership ceiling is computed against it")
    assert out["BETA"] == 79, "netting broke the ordinary single-leg case"


def test_a_lone_short_reports_NEGATIVE_not_positive():
    """A long-only lane holding a negative is a real condition — MOMENTUM-002 is carrying -28 WHD
    right now. Reporting it as +28 hides it from every check that looks for one."""
    from types import SimpleNamespace

    from kumo_strategies.runtime.nautilus.broker import NautilusBroker

    p = SimpleNamespace(instrument_id=SimpleNamespace(symbol=SimpleNamespace(value="WHD")),
                        quantity=28, signed_qty=-28.0)
    assert NautilusBroker._sum_positions([p])["WHD"] == -28


def _lanes_with_a_cadence_setting():
    """Lanes whose config carries a rebalance CADENCE, and the field's name.

    Filtered in the parametrize rather than skipped inside the test — a parametrized test that skips
    every case exits 0, which is how the price-field guard nearly shipped covering nothing.
    """
    import dataclasses
    import typing

    out = []
    for cls in _shipped_adapters():
        cfg_cls = typing.get_type_hints(cls.__init__).get("cfg")
        if not dataclasses.is_dataclass(cfg_cls) or not hasattr(cls, "_is_rebalance"):
            continue
        names = [f.name for f in dataclasses.fields(cfg_cls) if f.name == "rebalance_period"]
        if names:
            out.append((cls, cfg_cls, names[0]))
    assert out, "no lane carries a rebalance cadence — this guard no longer describes the package"
    return out


@pytest.mark.parametrize("cls,cfg_cls,field", _lanes_with_a_cadence_setting(),
                         ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x))
def test_the_cadence_is_READ_FROM_CONFIG_not_hardwired(cls, cfg_cls, field):
    """QC27'S CADENCE WAS RESEARCH-ONLY AND PRODUCTION WAS HARDWIRED MONTHLY.

    `runner_qc27_verified.run()` took a `rebalance_period` argument, so a sweep could measure monthly,
    weekly and daily and report a winner. BOTH production call sites — the adapter's `_is_rebalance`
    and the executor gateway — called `rebalance_dates()` with NO period and were pinned to monthly.
    Research and production disagreed about the cadence and nothing failed to say so.

    Same shape as `warmup_sessions` on the same branch: declared, swept, never read by the thing that
    trades. The measurement said ship daily; the config to do it existed only in the backtest.

    Asserted BEHAVIOURALLY — a monthly config and a daily config must disagree about which sessions
    are rebalances. A structural check (does it mention the field?) passes on a line that reads the
    config and throws the value away.
    """
    import dataclasses

    panel = pd.DataFrame({"date": pd.date_range("2026-01-01", "2026-03-31", freq="B")})
    monthly = cls(**_construct_kwargs(cls, dataclasses.replace(
        _build(cfg_cls, **_live_price_field(cfg_cls)), **{field: "M"})))
    daily = cls(**_construct_kwargs(cls, dataclasses.replace(
        _build(cfg_cls, **_live_price_field(cfg_cls)), **{field: "D"})))

    m = {d for d in panel["date"] if monthly._is_rebalance(d, panel)}
    d = {x for x in panel["date"] if daily._is_rebalance(x, panel)}
    assert m, f"{cls.__name__} found no monthly rebalance at all"
    assert len(d) > len(m), (
        f"{cls.__name__}: daily ({len(d)}) is not more frequent than monthly ({len(m)}) — the cadence "
        f"is not being read from the config, which is how a sweep can report daily while the live "
        f"strategy rebalances monthly")
    assert m < d, f"{cls.__name__}: monthly rebalances are not a subset of daily ones"


def _live_price_field(cfg_cls) -> dict:
    """`close` wherever the config has a price field, so construction is not refused by the guard that
    rejects a field no live bar can supply."""
    import dataclasses

    return {f.name: "close" for f in dataclasses.fields(cfg_cls)
            if "price" in f.name and "field" in f.name}


def _required(cfg_cls) -> dict:
    """What a config REFUSES to default (#212): `TemplateConfig.market_view` has no default because a
    window is measured per lane, never inherited. These generic constructions supply a stated view so
    the guards below can judge the rest of the config; the number is a test's, measured for nothing."""
    import dataclasses

    from kumo_strategies.strategies.market_view import MarketAction, MarketSignal, MarketViewConfig

    out = {}
    for f in dataclasses.fields(cfg_cls):
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            if f.name == "market_view":
                out[f.name] = MarketViewConfig(signal=MarketSignal.INDEX_VS_MA, window=50,
                                               action=MarketAction.EXIT_ONLY)
    return out


def _build(cfg_cls, **over):
    return cfg_cls(**{**_required(cfg_cls), **over})


def _lanes_with_a_warmup_calculation():
    """Lanes exposing `warmup_bars_needed(cfg)`, with the config's trailing-window fields."""
    import dataclasses
    import importlib
    import typing

    out = []
    for cls in _shipped_adapters():
        fn = getattr(importlib.import_module(cls.__module__), "warmup_bars_needed", None)
        cfg_cls = typing.get_type_hints(cls.__init__).get("cfg")
        if fn is None or not dataclasses.is_dataclass(cfg_cls):
            continue
        windows = [f.name for f in dataclasses.fields(cfg_cls)
                   if any(k in f.name for k in ("lookback", "warmup", "window"))
                   and isinstance(getattr(_build(cfg_cls), f.name, None), int)]
        if len(windows) >= 2:
            out.append((cls, fn, cfg_cls, tuple(windows)))
    assert out, "no lane exposes a warmup calculation — this guard no longer describes the package"
    return out


@pytest.mark.parametrize("cls,fn,cfg_cls,windows", _lanes_with_a_warmup_calculation(),
                         ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x))
def test_warmup_takes_the_MAX_of_its_windows_not_the_sum(cls, fn, cfg_cls, windows):
    """QC345 SUMMED THEM AND REQUIRED 295 BARS WHERE 254 WAS RIGHT.

    Every one of these is a TRAILING window ending at the same bar, so they OVERLAP. Summing overstates
    the requirement, and the cost is not cosmetic: a lane that thinks it needs more history than it does
    sits inert for weeks, warm by every real measure and refusing to rank — which reads exactly like a
    strategy that has found nothing to trade.

    Asserted as `max <= needed < sum`, which is what "takes the max" means without pinning any lane's
    particular arithmetic: some legitimately add a bar or two of slack for a shift(1).
    """
    cfg = _build(cfg_cls, **_live_price_field(cfg_cls))
    values = [getattr(cfg, w) for w in windows]
    needed = fn(cfg)

    assert needed >= max(values), (
        f"{cls.__name__}: warmup {needed} is less than its largest window {max(values)} — it would "
        f"rank on an incomplete trailing window")
    assert needed < sum(values), (
        f"{cls.__name__}: warmup {needed} is at or above the SUM of {dict(zip(windows, values))}. "
        f"These are overlapping trailing windows; summing them is how QC345 required 295 bars where "
        f"254 was right, and a lane that overstates its warmup sits inert while looking healthy")


@pytest.mark.parametrize("cls,cfg_cls,field", _lanes_with_a_price_field(),
                         ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x))
def test_a_config_REFUSES_a_wrong_typed_value(cls, cfg_cls, field):
    """THE CALLEE REFUSES WHAT IT CANNOT USE, rather than trusting the caller to coerce.

    Cockpit's settings overrides reach these configs UNCOERCED — `QC27_PORTFOLIO_SIZE="10"` arrives as
    a string. The mechanism is not a missing cast: cockpit builds the config by walking
    `dataclasses.fields()` and testing `is_dataclass(f.type)`, and because every config here uses
    `from __future__ import annotations`, **`f.type` is a STRING on every field** — so that test is
    unconditionally False, nested dataclasses are never constructed, and raw values pass straight
    through. It degrades silently to "hand the dict over", which is why nobody saw it.

    `portfolio_size="10"` then reaches slot arithmetic. `equity / "10"` raises; `"10" * n`
    concatenates. Neither is a config error at the point anyone would look.

    Cockpit is fixing the coercion with `typing.get_type_hints` — the same resolution I needed for the
    price-field guard this morning. This test is the other half: it makes their fix FALSIFIABLE
    instead of both of us assuming it worked, and it protects the boundary regardless of what any
    caller does.
    """
    import dataclasses

    ints = [f.name for f in dataclasses.fields(cfg_cls)
            if isinstance(getattr(_build(cfg_cls, **_live_price_field(cfg_cls)), f.name, None), int)
            and not isinstance(getattr(_build(cfg_cls, **_live_price_field(cfg_cls)), f.name, None), bool)]
    if not ints:
        pytest.skip(f"{cfg_cls.__name__} has no int field")

    base = _live_price_field(cfg_cls)
    with pytest.raises((TypeError, ValueError), match=ints[0]):
        _build(cfg_cls, **{**base, ints[0]: "10"})


def test_bool_and_int_are_not_interchangeable_to_the_validator():
    """`bool` IS a subclass of `int` in Python, so a naive `isinstance(value, int)` accepts `True` for
    an int field and the mirror accepts `1` for a bool field. A settings domain serves JSON, where
    `true` and `1` are a keystroke apart, so this is the realistic wrong value rather than a curiosity.

    Mutation-bitten: widening the bool check to `(bool, int)` survived every other assertion here,
    because nothing else ever passes a bool.
    """
    import dataclasses

    import pytest as _pytest

    from kumo_strategies.strategies._validate import check_field_types

    @dataclasses.dataclass(frozen=True)
    class _Cfg:
        count: int = 5
        enabled: bool = False

    check_field_types(_Cfg())                       # the honest case must pass

    with _pytest.raises(TypeError, match="count"):
        check_field_types(_Cfg(count=True, enabled=False))
    with _pytest.raises(TypeError, match="enabled"):
        check_field_types(_Cfg(count=5, enabled=1))
