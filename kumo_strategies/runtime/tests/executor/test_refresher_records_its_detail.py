"""The scheduled refresher must say WHAT it wrote on the row an operator reads.

`PgJobs.refresh` computes `res.detail`, journals it, and does NOT pass it to `refresh_source` — so
`exec_pool_source.detail` stays NULL for every source the in-process refresher maintains, while the
journal has the information.

THAT BLANK IS WHY 2026-08-26 TOOK TWO REPOS AND SEVERAL HOURS. `/pool` showed:

    alpaca-paper   ledger_book  age 0h    detail=None                          <- refreshed, silent
    ibkr-paper-retired  ledger_book  age 57h   detail="seeded from instance config"  <- bootstrap, never since

kumo-trading-platform and I independently concluded staging was "missing a pusher", because the only field
that could have said what maintains a source was empty on the instance where one was working.
Meanwhile the access log showed ZERO POSTs on alpaca-paper, so the HTTP path was not the writer
either, and nothing distinguished "refreshed in-process" from "never touched since bootstrap".

`PgJobs.refresh` reads `SourceSpecRow` from the database and calls `pool.refresh_source` directly —
no HTTP — which is exactly why `detail=None` and zero POSTs occur together.
"""

from __future__ import annotations

import inspect

from kumo_strategies.runtime.executor import pgjobs


def test_the_refresher_passes_its_detail_to_the_health_row():
    """Bound to the AST rather than run against Postgres: the claim is that the call SITE forwards
    the detail, and the write path needs a database while the decision does not — the same split
    `check_shrink` and `own_ceiling` already use."""
    import ast

    tree = ast.parse(inspect.getsource(pgjobs))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "refresh_source"]
    assert calls, "no call to refresh_source found — this test no longer describes the code"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert "detail" in kwargs, (
            f"pgjobs:{call.lineno} calls refresh_source without `detail`, so the source's health row "
            f"is left blank while the journal gets the text. That blank is what made a refreshed "
            f"source and an abandoned one look identical on /pool.")


def test_the_detail_is_the_fetchers_own_words_not_a_constant():
    """A hardcoded string would satisfy the test above and say nothing. It has to be what the source
    actually REPORTED, which is the only thing that distinguishes one refresher from another.

    Asserted on the argument's AST NODE TYPE, not on the source text. The first version read
    `"detail=res.detail" in inspect.getsource(...)`, which `tests/test_source_assertions_parse.py`
    rejects and is right to: a substring check against source is satisfied by deleting a comment and
    broken by writing one. A `Constant` here is a literal; a `Name` or `Attribute` is something the
    fetch produced."""
    import ast

    tree = ast.parse(inspect.getsource(pgjobs))
    for call in (n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "refresh_source"):
        for kw in call.keywords:
            if kw.arg != "detail":
                continue
            assert not isinstance(kw.value, ast.Constant), (
                f"pgjobs:{call.lineno} passes a literal as `detail` — the health row must carry what "
                f"the source reported, not a constant that is true of every refresh")
