# Engineering principles

The rules this repository is built and reviewed by. Each was written after a defect, and each
section names the defect by shape — the tracker it was filed in is private, and the shape is what
transfers. This is the engineering content of the project's operating manual; the manual itself
also carries day-to-day instructions for specific deployments and stays private.

## 1. Never call a broker from this repository

**Venue access goes through Nautilus. Always.** Strategies here run inside a platform that hosts a
NautilusTrader `TradingNode`; Nautilus holds the venue connectors. A strategy reaches a venue
*through Nautilus*, never by calling a broker's REST API itself.

Concretely:

- **Orders** — the Nautilus-backed broker, which submits through the node so the risk engine sees
  the order, the cache holds it and reconciliation can find it. A direct-to-venue broker class once
  existed here. It was never constructed; the hazard was that it existed, and its own docstring
  ("construct explicitly — nothing defaults to this") read as permission. It is deleted, and a test
  asserts no order path regains a socket.
- **Instruments and tradability** — the node's cache and its subscribed instrument set. A symbol
  the node has not subscribed is not tradable, whatever a venue's REST API says about it.
- **Prices** — the data client, through the strategy's own subscriptions.
- **Account and positions** — the account topic and the cache's open positions.

Why it is a rule and not a preference: a second path to the same venue is a second derivation of
the same fact, and the two disagree silently.

- A REST call said a symbol was tradable while the node had never subscribed it. The order was
  ranked, chosen, sized and refused at submit — a decision slot consumed by an order that could never
  fill, eight times in one session.
- A direct venue call binds the lane to *that* venue. A trading calendar fetched from one broker's
  endpoint gave a lane on the *other* broker's instance its calendar — a cross-venue dependency
  nobody chose.
- It puts a blocking socket wherever it is called from. One HTTP timeout inside `on_start` took
  three lanes down, because Nautilus re-raises out of the trader's start.

**If Nautilus does not expose something a strategy needs, that is a conversation with the platform —
not a reason to open a socket here.** The platform owns the node and its clients; it can supply the
value or add the connector.

## 2. What this repository is, and is not

Executable strategy implementations — decision engines, runtime adapters, replay/backtest runners —
and the evidence that promoted them. It is consumed by the platform repository, which owns the
control surface, the API bridge, the command bus and the book projection. Exploratory research
lives elsewhere; this tree keeps only the scripts and write-ups that produced a decision.

- **Decision-pure until bound.** A strategy's decision engine takes a frame and returns a decision.
  It is bound to Nautilus by an adapter, and the adapter owns featurisation and the day-slice — the
  engine enforces the contract it is handed rather than trusting the caller.
- **Strategy names carry no number.** Name a strategy `BCTROT`, never `BCTROT-003`. The order-id tag
  is a global sequence two repositories would have to agree on, and nothing makes them agree; the
  platform is the only place that sees every strategy in one trader, so it allocates. A ticket title
  once made a tag look claimed from this side and free from the other, two strategies were assigned
  the same tag, and Nautilus refused to boot the node. It was correctable only because one of them
  had never filled — after one position the id is permanent.
- **The identity contract** `(account_id, client_id, instrument_id, strategy_id, cycle_id)` is the
  platform's. Every broker order is owned by exactly one strategy at creation. Derived cycle P&L is
  a projection; broker reconciliation is the hard anchor.
- **Data is not in this tree.** Market data and study outputs live in a private data repository and
  are read through `KUMO_DATA_ROOT` — a name, never a path, the way a secret is a keychain name.
  A reader whose root is unset, or whose file is absent, refuses and names what it wanted. There is
  no default: the default this replaced was an absolute path on one machine, and one reader returned
  an *empty frame* for a session whose file was missing, so a missing day and a flat day were one
  value. The count of tests that need data is asserted, so a data-dependent test cannot vanish into
  a skip nobody counted.

## 3. Every bug gets a test — seen red

**A bug is not fixed until a test fails without the fix.** No exceptions for one-liners.

- **Prove it.** Reintroduce the bug, watch the test go red, restore, watch it go green, and say so in
  the PR. A test written after the fix that was never seen red is a guess about what it covers.
- **A test that has not been seen failing is an assertion about nothing.** Three repositories
  converged on this independently in one week. For a *result* rather than a code path — a backtest
  number, a fitted envelope — the equivalent is breaking the input deliberately and confirming the
  number moves.
- **An insufficient bite looks exactly like an unearned test.** Green after a mutation is ambiguous
  both ways: the mutation may not have cut every wire, or the test may have been red already. Green
  baseline first; then cut; then restore and verify the restore byte-for-byte — an edit-run-restore
  inside one second can leave stale bytecode live.
- **Mutation bites only probe what fixtures can express.** An axis no fixture varies is invisible to
  mutation testing, and its green looks identical to real coverage. Assert the fixture's own property
  before asserting the invariance.
- **Test the seam, not the unit.** A double that is more forgiving than production hides a live
  defect; bind the real function to a narrow host rather than loosening the fake. A fake broker that
  accepts what production rejects turns a vulnerability into a claimed incident and a defect into a
  green.
- **Bind the AST, not the source text.** A test that greps the source for a substring is satisfied
  by deleting a comment and broken by writing one. `hasattr` is satisfied by an inherited no-op —
  resolve the definition through the MRO. Forwarding through `**kwargs` is not a public seam: assert
  on the signature.
- **Uniform failure means a broken detector.** All-red or all-green across every subject is evidence
  about the instrument, not the subjects.

## 4. Verify by disagreement, not by inspection

Two derivations of one fact are a detector. **Identical when they should differ** means a dead
mechanism — configuration flags computed and discarded, sweepable, returning byte-identical results.
**Differing when they should match** means a live defect — two implementations of one exit simulator
off by 1.69% exposed a population bias no review had found. Across a dozen measured defects, code
review caught none and a pre-registration freeze caught none; disagreeing implementations, arithmetic
that could not be true, and peers re-checking settled claims caught all of them.

- **Agreement is not connection.** Test a knob with a value the default could not produce. Never
  20 000 when the constant is 20 000. A test once asserted a falsy-`or` account fallback as
  *deliberate* — green, with a stated reason, and both halves wrong; the gate it trusted defaulted
  to `None`.
- **An empty journal is not proof nothing runs.** Verify what is armed; enumerate the siblings. A
  runbook said DISABLED where the code said TRADING.
- **A log records delivery, not handling.** "Received bar" was true and irrelevant; verify an ingest
  by its effect in this tree's own state.
- **Comments can describe machinery that never existed.** Grep the nouns a comment names, especially
  safeguards; a false premise was once quoted into another repository's design.
- **Sequential fatal failures hide the count.** One reported failure in a start sequence is a lower
  bound of one, not a count of one; the first death hides identical siblings.
- **Prose beside a value is part of the value.** When changing a configuration value, change the
  sentence above it in the same commit.

## 5. Absence is not permission

- **`or` does not guard NaN, and it cannot tell unset from zero.** In one day, three directions: `0`
  read as absent, `NaN` read as present, and `NaN` surviving `<= 0` and disarming a halt. Use
  `x if x is not None else y`, guard non-finite explicitly, and prefer refusing to guessing.
- **A default that is a real identity is a wrong answer with good manners.** A live id, a machine
  path, an account — as a default, undetectable downstream. Require the field or use an empty
  sentinel, and guard it structurally.
- **Empty is the bug, so the bug cannot identify it.** A detector that recognises its subject by a
  property the defect destroys can never fire. Learn names where they are populated; assert on
  contents.
- **Three states, never two.** Known-yes, known-no, never-told-us. The third is its own answer — a
  typed refusal rather than `None`, a named skip rather than a pass.
- **Fixed horizons are a parameter, not an outcome.** Stocks have no holding period; end an episode
  on a state condition and report every definition side by side.
- **Default to the current regime.** Measure the most recent year first; older history is a
  robustness check, never the headline.

## 6. The lifecycle gate

A strategy is not ready for platform wiring until it has, in order:

1. a pure decision engine with deterministic tests;
2. a replay/backtest runner using the same decision contract;
3. a performance ledger with explicit fee and slippage assumptions;
4. a Nautilus adapter test or smoke run;
5. a paper-trading gate and a documented live gate.

Every lane declares its market view or refuses to register — no lane opts out. A live lane refuses
a configuration no runner can execute. Every adapter subscribes the handlers it defines; a class
test discovers the lanes rather than listing them, because a hand-kept list was the defect.

## 7. Working in this tree

- **Check Nautilus first.** Before writing order, execution or scheduling logic, prove Nautilus does
  not already do it; read the installed package, not memory.
- **Stage explicit paths.** Several sessions can resolve this package from one working tree (an
  editable install). Never `git add -A`, `git add .` or `git commit -a`: a sweep once carried another
  session's uncommitted edit into an unrelated squash. `git checkout -b` carries uncommitted changes
  onto the new branch; `git status` must be clean before branching, or stash and say so. Before
  committing, read the staged file list against what the message claims; before merging, read
  `git diff --name-only main..branch` against the PR's stated scope.
- **An editable install makes worktrees lie.** Every worktree imports the package from the checkout
  the install points at, so a branch missing a module still imports it and a run there proves nothing
  about that branch. Set `PYTHONPATH` to the worktree root; the suite refuses to run against a
  different checkout's source and prints which one it served.
- **Structured reports.** Status, defects and decisions are structured, not prose: What · Cause ·
  Impact now · Impact if ignored · Fix · When · Risk · Decision needed. Drop any line without content.
  Lead with the problem, not the investigation. Numbers over adjectives. Never present a menu without
  a recommendation.
- **The reviewer is named.** One review per procedure step, findings folded, not re-reviewed. A
  named cross-review by a different reader with different assumptions is the substitute when the
  primary reviewer is unavailable; every PR states which reviewer it got.
- Conventional Commits. Tests beside source. Comments in English. Every non-hidden directory carries
  a README saying what belongs there and what does not. Never commit credentials, broker keys, raw
  market data or generated backtest artefacts — `bin/check-public-tree.sh` refuses them.
