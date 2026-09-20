"""Candidate sources — WHAT the strategy is allowed to trade on a given day.

This is the load-bearing abstraction. Measurement (issue #15) showed the rotation rule is not an
edge on its own: run identically over random liquidity-matched universes it LOST money, median
-6.5%, while BCT's pool returned +53% with 0 of 40 random seeds beating it. The pool is the edge.

So the pool must be swappable without touching the ranking or execution logic. A source answers one
question: which symbols are eligible at the close of session `d`? It must be strictly point-in-time
— returning anything knowable only after `d` silently manufactures performance.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class CandidateSource(Protocol):
    """Eligible symbols as of the close of a session. Must never look forward."""

    name: str

    def eligible(self, d: pd.Timestamp) -> set[str]:
        ...


_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Register a source under its `name`. Mirrors the cockpit registry pattern."""
    _REGISTRY[cls.name] = cls
    return cls


def build(name: str, **params) -> CandidateSource:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown candidate source {name!r} — registered: {sorted(_REGISTRY)}") from None
    return cls(**params)


def registered() -> list[str]:
    return sorted(_REGISTRY)


@register
class BctPositions:
    """Whatever BCT held at the close of `d`, from the position ledger.

    Uses the ledger's explicit buy_date/sell_date, NOT the action stream. ledger-tool's holdings.md
    warns the action stream over-counts, because buys go unmatched to sells and stale positions
    accumulate — reconstructing from it inflated the book from ~30 names to 76.

    `lag_days` exists because his posts are same-day: a position dated d is not actionable until the
    next session. Measured, this matters enormously — without the lag the book returns +80.9%, with
    it +7.4%, because names move +3.8% on average on the day his buy is recorded.

    `max_open_days` fixes a staleness bias that is NOT optional. Sells go unrecorded, so a position
    with no sell_date stays in the book forever and the pool inflates: 121 names on 2026-06-03
    against a real book of roughly 40 (ledger-tool tracking/ledger-holdings.md, re-seeded from the
    weekend video where he shows the whole book). Worse, the names that silently persist are the
    ones that kept running, so the stale tail is biased toward winners.

    The cutoff is taken from his OWN behaviour rather than picked: across 665 recorded closes the
    median hold is 12 calendar days, p90 is 39 and p95 is 58. A position still "open" past that is
    far more likely to be a missed sell than a genuine hold, so it drops out of the pool.
    """

    name = "bct_positions"

    def __init__(self, ledger_path: str, lag_days: int = 1, max_open_days: int | None = 58) -> None:
        p = pd.read_csv(ledger_path)
        p["buy_date"] = pd.to_datetime(p["buy_date"], errors="coerce")
        p["sell_date"] = pd.to_datetime(p["sell_date"], errors="coerce")
        self._p = p.dropna(subset=["buy_date"])
        self._lag = pd.Timedelta(days=lag_days)
        self._max_open = pd.Timedelta(days=max_open_days) if max_open_days else None

    def eligible(self, d: pd.Timestamp) -> set[str]:
        ref = d - self._lag
        open_now = self._p.sell_date.isna() | (self._p.sell_date > ref)
        m = (self._p.buy_date <= ref) & open_now
        if self._max_open is not None:
            # only unsold rows are suspect — a recorded sell_date is evidence, not a guess
            unsold = self._p.sell_date.isna()
            m &= ~unsold | ((ref - self._p.buy_date) <= self._max_open)
        return set(self._p.loc[m, "symbol"])


@register
class StaticList:
    """A fixed watchlist. The simplest source — useful for a hand-curated industry list, and the
    baseline any smarter source has to beat."""

    name = "static_list"

    def __init__(self, symbols: list[str]) -> None:
        self._s = set(symbols)

    def eligible(self, d: pd.Timestamp) -> set[str]:
        return set(self._s)


@register
class IndustryMembers:
    """Members of one or more industries, resolved by which sector/industry ETF each name actually
    tracks rather than by index membership.

    Assignment is by trailing correlation to an ETF panel (measured: FIG->IGV 0.65, ESTC->IGV 0.74,
    TEAM->IGV 0.70, NVDA->SMH 0.68, MPC->XOP 0.73, median best-fit 0.54 across 2,400 names). That
    sidesteps point-in-time holdings entirely — no N-PORT 58-day lag, no scraping, and a recent IPO
    belonging to no index is still placed.

    This is the source the operator asked for: industry-based, not capitalisation-based.
    """

    name = "industry_members"

    def __init__(self, assignment: dict[str, str], industries: list[str]) -> None:
        want = set(industries)
        self._members = {s for s, etf in assignment.items() if etf in want}
        if not self._members:
            raise ValueError(f"no members for industries {sorted(want)}")

    def eligible(self, d: pd.Timestamp) -> set[str]:
        return set(self._members)


@register
class UnionSource:
    """Several sources combined. Lets an industry screen run alongside a curated book."""

    name = "union"

    def __init__(self, sources: list[CandidateSource]) -> None:
        self._sources = list(sources)

    def eligible(self, d: pd.Timestamp) -> set[str]:
        out: set[str] = set()
        for s in self._sources:
            out |= s.eligible(d)
        return out


@register
class LedgerToolHoldings:
    """BCT's book as maintained by ledger-tool — the authoritative source.

    ledger-tool owns this ledger. `tracking/ledger-holdings.md` is re-seeded every weekend from the
    members video, where he shows his entire book, and daily posts are applied as deltas between
    seeds. Its own maintenance protocol is explicit that the signal CSV is not the book:

        "The ledger-signals-backtest.csv is NOT the book - it over-counts (dangling buys never
         matched to a sell). Use it only for the delta stream, not net positions."

    Which is exactly the failure that inflated the pool to 121 names against a real book of ~40.

    Two representations are parsed:
      table rows   `| TKR | Jul10 | note |` — open unless struck through
      strikethrough `~~TKR~~ | CLOSED Jul30 +1.2%` — closed, with the date
      delta lines  `2026-07-31 (Fri): ADDED A, B; CLOSED C 7.2%, D 5.3%`

    Point-in-time is only as good as the delta history. Where the file has no deltas for a date the
    book is carried forward from the nearest earlier seed, so a window predating the file cannot be
    reconstructed from it — use the ledger for that and say so.
    """

    name = "ledger_tool_holdings"

    _MONTHS = {m: i for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

    def __init__(self, path: str, year: int = 2026) -> None:
        import re

        text = open(path).read()
        self._closed: dict[str, pd.Timestamp] = {}
        self._open: set[str] = set()
        self._deltas: list[tuple[pd.Timestamp, str, str]] = []

        for tkr, mon, day in re.findall(r"~~([A-Z.]{1,6})~~[^|\n]*\|\s*CLOSED\s+([A-Za-z]{3})(\d{1,2})", text):
            if mon in self._MONTHS:
                self._closed[tkr] = pd.Timestamp(year, self._MONTHS[mon], int(day))

        for row in re.findall(r"^\|\s*([A-Z][A-Z.]{0,5})\s*\|", text, re.M):
            if row not in self._closed:
                self._open.add(row)

        for line in re.findall(r"^\s*(\d{4}-\d{2}-\d{2})[^\n]*", text, re.M):
            pass  # dates captured below with their payload
        for d, body in re.findall(r"^\s*(\d{4}-\d{2}-\d{2})[^:]*:\s*(.+)$", text, re.M):
            ts = pd.Timestamp(d)
            for m in re.finditer(r"ADDED\s+([^;]+)", body):
                for t in re.findall(r"\b([A-Z][A-Z.]{0,5})\b", m.group(1)):
                    self._deltas.append((ts, "ADD", t))
            for m in re.finditer(r"CLOSED\s+(.+)$", body):
                for t in re.findall(r"\b([A-Z][A-Z.]{0,5})\b", m.group(1)):
                    self._deltas.append((ts, "CLOSE", t))
        self._deltas.sort()

    @property
    def closed(self) -> dict[str, pd.Timestamp]:
        """Symbol -> close date, for correcting a ledger whose sells went unrecorded."""
        return dict(self._closed)

    def eligible(self, d: pd.Timestamp) -> set[str]:
        """The book as of `d`, replaying deltas forward from the seed."""
        held = set(self._open)
        for ts, action, tkr in self._deltas:
            if ts > d:
                break
            held.add(tkr) if action == "ADD" else held.discard(tkr)
        return {t for t in held if self._closed.get(t, pd.Timestamp.max) > d}


@register
class LedgerBook:
    """BCT's book from ledger-tool's machine-readable export — the source to use.

    `research/ledger/ledger-book.csv`: symbol, open_date, close_date, close_pct,
    still_open, close_source. 550 intervals reconstructed from the full 833-post corpus and verified
    against ground truth (2026-07-15 yields exactly PENG -9.7%, PBF -9.5%, BNY open, ZION open).
    Regenerated nightly, so it stays current. Prefer this over parsing the markdown.

    TWO THINGS LEDGER_TOOL MEASURED THAT THIS CLASS HAS TO RESPECT:

    Over-count. Comparing derived membership against weekend members-video snapshots — where he
    shows his whole book — the post-derived book runs 3x to 11x too large:

        date        video   derived   over
        2026-05-25     17       119   7.0x
        2026-06-14     29       118   4.1x
        2026-07-10     42       121   2.9x

    The 07-10 row is the honest one: that review is the most complete and matches his stated "~40
    positions". So ~3x is the FLOOR, and video counts are a lower bound, not a count.

    Expiry calibration. An earlier version expired unsold positions at the p95 of recorded holds.
    ledger-tool pointed out that p95 is itself computed from the biased tail — theirs comes out 118d
    against my 58d for exactly that reason. So the cutoff is calibrated against a SNAPSHOT instead:
    pick the age limit that reproduces the observed book size on a date where the video is complete.

    Coverage. An earlier version cut this off at 2026-03 on ledger-tool's report that Oct 2025 - Feb
    2026 was genuinely absent. They corrected it: the sparse months were a shallow fetch, not a
    silent author - he posts every trading day, and 1-2 posts a month is a puller that stopped early.
    A deeper pull took community coverage from 55 to 201 date dirs, and Dec/Jan/Feb are now
    effectively complete at ~20 trading-day posts each.

    So the cutoff is 2025-12-01: the month density first reaches ~20 posts. Oct/Nov 2025 stay
    excluded - they recovered only partially and YouTube's continuation caps at 244 posts, which is
    a platform ceiling rather than something to push on. `coverage_start` refuses to answer below it
    instead of returning a book that quietly invents membership.
    """

    name = "ledger_book"
    COVERAGE_START = pd.Timestamp("2025-12-01")

    def __init__(self, csv_path: str, lag_days: int = 1, max_open_days: int | None = None,
                 coverage_start: str | pd.Timestamp | None = COVERAGE_START) -> None:
        b = pd.read_csv(csv_path)
        b["open_date"] = pd.to_datetime(b["open_date"], errors="coerce")
        b["close_date"] = pd.to_datetime(b["close_date"], errors="coerce")
        # rows with no observed open cannot be placed in time; they carry a close only
        self._b = b.dropna(subset=["open_date"]).copy()
        self._lag = pd.Timedelta(days=lag_days)
        self._max_open = pd.Timedelta(days=max_open_days) if max_open_days else None
        self._coverage_start = pd.Timestamp(coverage_start) if coverage_start is not None else None

    def eligible(self, d: pd.Timestamp) -> set[str]:
        if self._coverage_start is not None and d < self._coverage_start:
            return set()          # refuse rather than invent
        ref = d - self._lag
        m = (self._b.open_date <= ref) & (self._b.close_date.isna() | (self._b.close_date > ref))
        if self._max_open is not None:
            # The age cap models a MISSED SELL: the post stream under-records exits, so a row with
            # no close_date may well be closed in reality. It must therefore apply ONLY to rows with
            # no recorded close.
            #
            # Applying it to every row also evicted positions whose exit IS recorded further out.
            # 56.7% of his closed trades run past 20 days (median 26, p75 65, p90 153), so a
            # 19-day cap dropped more than half his live book from the pool while he was
            # demonstrably still holding — and it is why we exited shared names ~50 days before he
            # did, taking +1.9% on them against his +7.3%.
            unclosed = self._b.close_date.isna()
            m &= ~unclosed | ((ref - self._b.open_date) <= self._max_open)
        return set(self._b.loc[m, "symbol"])

    @staticmethod
    def calibrate_expiry(csv_path: str, snapshots_path: str, anchors: list[str] | str,
                         candidates: range = range(5, 121, 1)) -> int:
        """Fit one expiry across the COMPLETE video reviews.

        Calibrating against observed snapshots avoids the circularity ledger-tool flagged: the hold
        distribution is drawn from the same biased data the expiry is meant to correct.

        Fit only against complete reviews. Counts are LOWER bounds — a thin review means he
        annotated fewer names that week, not that the book shrank. 2026-05-10 shows 6 and
        2026-07-25 shows 8; fitting to those would drive the expiry far too short. The reviews
        consistent with his stated "~40 positions" are 2026-03-15 (37) and 2026-07-10 (42), which
        also bracket the tradeable window at both ends.
        """
        import json

        if isinstance(anchors, str):
            anchors = [anchors]
        snaps = json.load(open(snapshots_path))["snapshots"]   # {date: {held: [...], count: n}}
        targets = {}
        for a in anchors:
            snap = snaps.get(a)
            if not snap:
                raise ValueError(f"no snapshot for {a} — have {sorted(snaps)}")
            targets[pd.Timestamp(a)] = snap.get("count") or len(snap.get("held", []))

        best, best_err = None, None
        for days in candidates:
            # lag_days=0 here on purpose. The trading path lags by a session because a post is
            # only actionable the next morning, but a SNAPSHOT is a statement about the book on
            # its own date. Calibrating through the default lag compared a 2026-07-10 snapshot
            # against eligibility as of 07-09 and miscounted same-day opens and closes.
            src = LedgerBook(csv_path, lag_days=0, max_open_days=days, coverage_start=None)
            err = sum(abs(len(src.eligible(d)) - t) for d, t in targets.items())
            if best_err is None or err < best_err:
                best, best_err = days, err
        return int(best)

    def validate_pre_coverage(self, snapshots_path: str) -> int:
        """Resolve positions opened before coverage using the nearest FOLLOWING video snapshot.

        ledger-tool's suggestion, and better than either option I offered. For an open with no recorded
        close, look at the next review: if the name appears he genuinely still held it; if it is
        absent he was out by then, so close it at the snapshot date. That replaces a guess with an
        observation.

        Absence is weaker evidence than presence — reviews do not annotate everything — so this only
        closes a position when a snapshot exists after its open and omits it. Returns how many rows
        were resolved.
        """
        import json

        snaps = json.load(open(snapshots_path))["snapshots"]
        by_date = sorted((pd.Timestamp(k), set(v.get("held", []))) for k, v in snaps.items())
        if not by_date:
            return 0

        resolved = 0
        for i, row in self._b.iterrows():
            if pd.notna(row.close_date):
                continue
            nxt = next(((d, held) for d, held in by_date if d > row.open_date), None)
            if nxt is None:
                continue
            d, held = nxt
            if row.symbol not in held:
                self._b.at[i, "close_date"] = d      # observed absent -> he was out by then
                resolved += 1
        return resolved
