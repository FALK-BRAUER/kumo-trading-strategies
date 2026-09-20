"""Venue access goes through Nautilus. Two modules under `runtime/` still do not, and are recorded.

THE TITLE IS THE GOAL, NOT THE STATE. `calendar.py` and `executor/tradable.py` still reach Alpaca
directly and are listed below with their reasons; both are tracked in #75 and both block ibkr-paper-retired
from running without an Alpaca credential. This file stops the list growing and stops an ORDER path
ever joining it — it does not assert the rule is already met (2026-08-27).

**the operator's ruling, 2026-08-27.** Strategies run on kumo-trading-platform, which runs on Nautilus; Nautilus holds
the connectors (IBKR, Alpaca). A strategy reaches a venue THROUGH Nautilus — orders via
`NautilusBroker`, instruments via `cache.instrument()`, prices via the data client, account and
positions via the `broker.account` topic and `cache.positions_open()`.

WHY IT IS A RULE. A second path to the same venue is a second derivation of the same fact, and the
two disagree silently:

  * Alpaca REST says a symbol is tradable while the node never subscribed it — so it is ranked,
    chosen, sized, and refused at submit as "not a subscribed instrument". TECHIVOL-005 hit that
    EIGHT TIMES in one session.
  * A direct call binds the lane to THAT venue: `calendar.py` hardcodes Alpaca's paper API, so
    BCTROT-004 — which trades on IBKR — takes its trading calendar from Alpaca.
  * It puts a blocking socket wherever it is called. One `urlopen` timeout in `on_start` took three
    lanes down on 2026-08-22, because Nautilus re-raises out of `Trader.START`.

`backtesting/` is deliberately out of scope: offline research fetching reference data is a different
question from a live lane reaching around its own node.

WHAT THIS GUARD CATCHES AND WHAT IT DOES NOT — stated precisely, because three review rounds were
spent on this docstring being more confident than the code.

CATCHES a call whose chain descends from `_NETWORK_PREFIXES` via an IMPORT in the same module:
module aliases (`requests as _rq`), submodules bound from a parent (`from urllib import request`),
callables imported out of those modules under any name (`from urllib.request import urlopen as
anything`), client objects (`requests.Session().get(...)`, `socket.socket().connect(...)`,
`build_opener().open(...)`), and any method name at all — the match is on the module PATH, so an
unknown or new method fails closed rather than slipping through.

DOES NOT CATCH:

  * a venue SDK with its own transport (`alpaca_trade_api`, `ib_insync`) — the chain roots in the
    SDK, not in a stdlib/HTTP module
  * a call reached through a variable or attribute rather than an import (`self._http.get(...)`)
  * a subprocess (`curl`), or any module outside `_NETWORK_PREFIXES`, which is six paths not a registry
  * the call living in a non-`runtime/` helper that `runtime/` imports — the walk is scoped to
    `runtime/`
  * anything named in `_NOT_A_CALL_OUT`

It is a tripwire against the obvious regression, not a proof of the rule. The rule is enforced by
review; this stops the easy version of breaking it from landing unnoticed.
"""

from __future__ import annotations

import ast
import pathlib

_RUNTIME = pathlib.Path(__file__).resolve().parents[1]   # kumo_strategies/runtime/

#: Modules that STILL reach a venue directly, each a recorded decision rather than an oversight.
#: Keyed by path RELATIVE TO `runtime/`, never by bare filename: a second `calendar.py` under another
#: subdirectory would otherwise hide behind the known one (2026-08-27).
#: EQUALITY, not a subset: a new one fails here, and fixing one fails here too so the record cannot
#: drift from the code. Tracked in #75.
#:
#:   calendar.py   the trading calendar for ALL FOUR lanes. Nautilus does not obviously expose a
#:                 venue trading calendar, so removing this needs cockpit — the rule's own escape
#:                 clause. Cockpit DOES pass `require_exchange=True` for all three families, so the
#:                 holiday-unaware fallback cannot engage on a deployed lane — the real risk is the
#:                 inverse: it RAISES on missing APCA credentials and can stop lanes BUILDING on an
#:                 instance that has none, which is every IBKR instance.
#:   tradable.py   Alpaca's tradable universe, fed into `pgrunner.tradable_symbols` for MOMENTUM-002
#:                 and BCTROT-004. The node's own instrument set already answers this and is the
#:                 authority — `NautilusBroker._iid` is what actually decides whether an order can
#:                 be placed.
_KNOWN_DIRECT_VENUE_ACCESS = {"calendar.py": 1, "executor/tradable.py": 2}


#: Network primitives. Nothing under `runtime/` should call one: every byte a live lane needs comes
#: through Nautilus. Matching the CALL rather than a URL is the point — a host can be assembled,
#: aliased, or point at an IBKR gateway on localhost, and it is still a socket this repo opened.
#: Matched on the MODULE PATH, not the method name. An include-list of method names
#: (`get`/`post`/`urlopen`/...) is finite by construction, so `requests.patch(...)`,
#: `httpx.stream(...)` and `build_opener().open(...)` all passed it (codex, round 3). Anything
#: reached through one of these paths counts unless it is explicitly excluded below — unknown names
#: FAIL CLOSED rather than slipping through.
#:
#: Paths, not roots: `urllib.request` is a way out to the network and `urllib.parse` is not.
_NETWORK_PREFIXES = ("urllib.request", "requests", "httpx", "aiohttp", "socket", "http.client")

#: Names on those paths that open NOTHING. `urllib.request.Request(...)` builds a request object; the
#: socket is opened by whatever is handed it. Kept deliberately tiny — every entry is a hole.
_NOT_A_CALL_OUT = ("Request",)


def _alias_maps(tree: ast.AST) -> tuple[dict[str, str], dict[str, str]]:
    """`(module alias -> dotted module path, imported callable -> the module it came from)`.

    TWO MAPS, NOT ONE. `import requests as _rq` needs the first; `from urllib.request import urlopen
    as venue_call` needs the second, and only having the first is how `venue_call(...)` slipped past
    (codex, round 2). A callable pulled out of a network module reaches the network whatever it has
    been renamed to — subject to `_NOT_A_CALL_OUT`.
    """
    mods: dict[str, str] = {}
    fns: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                # FULL dotted target, so `import urllib.request` maps `urllib` -> `urllib` and the
                # attribute chain supplies `.request`; an alias maps straight to the full path.
                mods[(a.asname or a.name).split(".")[0]] = (
                    a.name if a.asname else a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                # AMBIGUOUS BY CONSTRUCTION: `from urllib import request` binds a SUBMODULE, while
                # `from urllib.request import urlopen` binds a CALLABLE, and the AST cannot tell them
                # apart without importing. Recording only the parent missed
                # `from urllib import request; request.urlopen(...)` entirely (codex, round 4).
                # Both readings are kept and either may match.
                fns[a.asname or a.name] = f"{node.module}|{node.module}.{a.name}"
    return mods, fns


def _is_network(path_: str) -> bool:
    """Does any candidate reading of this chain descend from a network module path?"""
    return any(c == q or c.startswith(q + ".")
               for c in path_.split("|") for q in _NETWORK_PREFIXES)


def _dotted(node: ast.AST, mods: dict[str, str], fns: dict[str, str]) -> str:
    """The dotted module path(s) a call chain descends from, `|`-separated, or "".

    RECURSES THROUGH A CALL, which is what catches a client OBJECT rather than a module:
    `requests.Session().get(...)`, `httpx.Client().post(...)`,
    `http.client.HTTPSConnection(...).request(...)`, `socket.socket().connect(...)`. The chain roots
    at a Call, not a Name, so a walker that only accepted a module name saw nothing (codex, round 2).
    """
    parts: list[str] = []
    while True:
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Name):
            base = mods.get(node.id) or fns.get(node.id) or node.id
            tail = list(reversed(parts))
            return "|".join(".".join([b, *tail]) for b in base.split("|"))
        else:
            return ""


#: Modules that STILL reach a venue directly, each a recorded decision rather than an oversight.
#: Keyed by path RELATIVE TO `runtime/`, never by bare filename: a second `calendar.py` under another
#: subdirectory would otherwise hide behind the known one (2026-08-27).
#: EQUALITY, not a subset: a new one fails here, and fixing one fails here too so the record cannot
#: drift from the code. Tracked in #75.
#:
#:   calendar.py   the trading calendar for ALL FOUR lanes. Nautilus does not obviously expose a
#:                 venue trading calendar, so removing this needs cockpit — the rule's own escape
#:                 clause. Cockpit DOES pass `require_exchange=True` for all three families, so the
#:                 holiday-unaware fallback cannot engage on a deployed lane — the real risk is the
#:                 inverse: it RAISES on missing APCA credentials and can stop lanes BUILDING on an
#:                 instance that has none, which is every IBKR instance.
#:   tradable.py   Alpaca's tradable universe, fed into `pgrunner.tradable_symbols` for MOMENTUM-002
#:                 and BCTROT-004. The node's own instrument set already answers this and is the
#:                 authority — `NautilusBroker._iid` is what actually decides whether an order can
#:                 be placed.
_KNOWN_DIRECT_VENUE_ACCESS = {"calendar.py": 1, "executor/tradable.py": 2}


#: Network primitives. Nothing under `runtime/` should call one: every byte a live lane needs comes
#: through Nautilus. Matching the CALL rather than the URL is the point — see `_network_calls_by_module`.
#: Matched on the MODULE PATH, not the method name. An include-list of method names
#: (`get`/`post`/`urlopen`/...) is finite by construction, so `requests.patch(...)`,
#: `httpx.stream(...)` and `build_opener().open(...)` all passed it (codex, round 3). Anything
#: reached through one of these paths counts unless it is explicitly excluded below — unknown names
#: FAIL CLOSED rather than slipping through.
#:
#: Paths, not roots: `urllib.request` is a way out to the network and `urllib.parse` is not.
_NETWORK_PREFIXES = ("urllib.request", "requests", "httpx", "aiohttp", "socket", "http.client")

#: Names on those paths that open NOTHING. `urllib.request.Request(...)` builds a request object; the
#: socket is opened by whatever is handed it. Kept deliberately tiny — every entry is a hole.
_NOT_A_CALL_OUT = ("Request",)


def _network_calls_by_module() -> dict[str, int]:
    """`{runtime-relative path: how many network CALL SITES it has}`.

    COUNTING URL CONSTANTS WAS THE WRONG MEASURE and I shipped it to review believing otherwise.
    A new `urlopen(f"{_BASE}/v2/orders")` inside `calendar.py` reuses the constant already there, so
    the count did not move and the guard passed — while the docstring claimed the opposite. My bite
    missed it because I mutated by adding a CONSTANT, which is what the detector measured, rather
    than a CALL, which is what the rule is about: a mutation aimed at the implementation confirms the
    implementation against itself.

    Counting call sites also states the rule more honestly than "no broker hostnames": nothing here
    should open a socket at all, whatever host it points at. An IBKR local gateway at
    `127.0.0.1:4001` carries no broker hostname and is exactly as much a violation.
    """
    out: dict[str, int] = {}
    for path in _RUNTIME.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                            # pragma: no cover
            continue
        mods, fns = _alias_maps(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            hit = False
            if isinstance(fn, ast.Attribute):
                hit = (fn.attr not in _NOT_A_CALL_OUT
                       and _is_network(_dotted(fn.value, mods, fns)))
            elif isinstance(fn, ast.Name):
                # A callable imported OUT of a network module reaches the network under any name.
                hit = (fn.id not in _NOT_A_CALL_OUT
                       and _is_network(fns.get(fn.id, "")))
            if hit:
                rel = path.relative_to(_RUNTIME).as_posix()
                out[rel] = out.get(rel, 0) + 1
    return out


def test_direct_venue_access_matches_the_record_EXACTLY():
    """Equality on the COUNTS, so the record cannot drift in either direction.

    A new module fails. A new call inside a recorded module fails, for the shapes the module
    docstring says are caught — "any new call fails" is NOT true, and this docstring asserted it
    twice before review caught it. Read the CATCHES/DOES NOT list before trusting a green. Removing a
    violation also fails, because a record still claiming one that is fixed is a lie in the direction
    nobody checks."""
    found = _network_calls_by_module()
    assert found == _KNOWN_DIRECT_VENUE_ACCESS, (
        f"direct venue access under runtime/ changed.\n"
        f"  found:    {dict(sorted(found.items()))}\n"
        f"  recorded: {dict(sorted(_KNOWN_DIRECT_VENUE_ACCESS.items()))}\n"
        f"Route it through Nautilus, or — if a recorded module genuinely gained a call — say why "
        f"here rather than raising the number quietly.")


def test_the_detector_can_actually_see_a_venue_url():
    """The control. `found == recorded` also passes if the walk finds nothing at all — which it would
    if `runtime/` moved, or if `ast.Call` matching broke. Then the test would be green while
    detecting nothing, which is the failure mode this whole file exists to prevent."""
    assert _network_calls_by_module(), "the detector found no network call anywhere — it is not looking"


def test_no_order_path_talks_to_a_venue():
    """The half that must stay empty. A calendar or a symbol list reaching a venue is wrong and
    recoverable; an ORDER path doing it bypasses the RiskEngine, the cache and reconciliation at
    once — nothing downstream would know the order exists."""
    assert "executor/broker.py" not in _network_calls_by_module(), (
        "an order path reaches a venue directly — orders must go through NautilusBroker so the "
        "RiskEngine sees them, the cache holds them, and reconciliation can find them")
