"""A test that reads source must PARSE it, never substring-match it (2026-08-21).

WHY THIS EXISTS AS A MECHANICAL RULE RATHER THAN FOLKLORE. Two independent instances surfaced in one
evening, in two repositories, with OPPOSITE symptoms and one root cause:

    kumo-trading-platform  assert "slot=slot" in src        PASSED with the bug present — it matched a COMMENT
    kumo-trading-strategies  assert "QC27Rotation..." not in src   FAILED with the code correct — it matched a
                                                            comment explaining why not to do that

So a source-substring assertion is satisfied by DELETING an explanation and broken by WRITING one. In
these repos the comments carry the incident history — the reason a thing is done the unsafe-looking
way, the outage that produced the guard — so the incentive it creates is actively harmful: it rewards
stripping commentary out of exactly the code whose intent most needs explaining.

Neither instance was caught by review. Both were caught by mutation-biting the guard.

THE RULE: if a test reads source (`inspect.getsource`, `Path.read_text`), it must assert over the AST —
`ast.walk` for a Call, a keyword, a decorator, an ordering — not `in`/`not in` against the text.

THE ALLOWLIST IS A DEBT LEDGER, NOT AN EXEMPTION. It records the assertions that predate the rule. It
must only ever shrink; adding to it is the thing this file exists to prevent. Keyed by (file, test
name) rather than line number so it stays stable when code moves.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_SRC_TESTS = Path(__file__).resolve().parents[1] / "kumo_strategies"   # tests live beside their packages (ks#211)

#: Reads-source-as-text assertions that predate this rule. SHRINK ONLY.
_KNOWN_WEAK: set[tuple[str, str]] = {
    ("test_executor.py", "test_the_give_back_trail_must_not_live_in_memory"),
    ("test_bctrot.py", "test_the_schedule_is_the_ONLY_thing_that_varies_from_momentum"),
    ("test_momentum_rotation.py", "test_on_start_requests_instrument_definitions_not_only_bars"),
    ("test_decide_wiring_audit.py", "test_the_live_runner_consumes_the_weights_it_asks_for"),
}

_READS_SOURCE = {"getsource", "getsourcelines", "read_text"}


def _source_bound_names(tree: ast.AST) -> set[str]:
    # SCOPED BY CALLER. This is called per FUNCTION, not per module — a module-wide scan let a name
    # bound from `read_text()` in one test flag an unrelated `in <same name>` in another, which is
    # exactly the false positive that teaches people to disable a guard. Caught by the guard firing
    # on this file's own sibling test.
    """Locals assigned from something that reads SOURCE.

    Deliberately narrow: it tracks the ORIGIN of the value rather than the variable's name. An earlier
    version flagged any name containing "text" and reported `assert "TypeError" in text` against a
    JOURNAL SUMMARY — a false positive that would have taught people to rename variables to dodge the
    check, which is worse than not having it.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for call in ast.walk(node.value):
            if not isinstance(call, ast.Call):
                continue
            fn = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)
            if fn in _READS_SOURCE:
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _offenders() -> list[tuple[str, str, int]]:
    out = []
    for path in sorted([*_TESTS.rglob("test_*.py"), *_SRC_TESTS.rglob("test_*.py")]):
        if path.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                        # not ours to police
            continue
        for fn in (n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            bound = _source_bound_names(fn)          # per-function, see the note above
            if not bound:
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assert):
                    continue
                for cmp in (c for c in ast.walk(node.test) if isinstance(c, ast.Compare)):
                    for op, right in zip(cmp.ops, cmp.comparators):
                        if (isinstance(op, (ast.In, ast.NotIn))
                                and isinstance(right, ast.Name) and right.id in bound
                                and (path.name, fn.name) not in _KNOWN_WEAK):
                            out.append((path.name, fn.name, node.lineno))
    return out


def test_no_new_test_asserts_a_substring_against_source():
    offenders = _offenders()
    assert not offenders, (
        "these assertions substring-match SOURCE TEXT, so they are satisfied by deleting a comment "
        "and broken by writing one — bind the AST instead (ast.walk for a Call/keyword/ordering):\n"
        + "\n".join(f"  {f}::{t} line {ln}" for f, t, ln in offenders))


def test_the_allowlist_only_ever_shrinks():
    """A ledger that can grow is not a ledger. Every entry must still correspond to a real, still-weak
    assertion — an entry whose test was fixed or deleted is stale and must come out, or the allowlist
    silently re-permits the pattern under a reused name."""
    live = set()
    for path in sorted([*_TESTS.rglob("test_*.py"), *_SRC_TESTS.rglob("test_*.py")]):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for fn in (n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            bound = _source_bound_names(fn)          # per-function, see the note above
            if not bound:
                continue
            for node in ast.walk(fn):
                for cmp in (c for c in ast.walk(node) if isinstance(c, ast.Compare)):
                    for op, right in zip(cmp.ops, cmp.comparators):
                        if (isinstance(op, (ast.In, ast.NotIn))
                                and isinstance(right, ast.Name) and right.id in bound):
                            live.add((path.name, fn.name))
    stale = _KNOWN_WEAK - live
    assert not stale, (
        f"allowlist entries no longer correspond to a weak assertion — delete them: {sorted(stale)}")
