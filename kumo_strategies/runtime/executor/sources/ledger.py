"""ledger-provider's book as a pool source — the trader whose holdings we trade a concentrated slice of.

Reads ledger-tool's machine-readable export, which is regenerated nightly. Point-in-time by
construction: a position counts only if it opened on or before the reference date and has not closed
by then. The reference date lags by a session because a post is only actionable the next morning.
"""

from __future__ import annotations

import hashlib
import json
import os

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kumo_strategies.runtime.executor.sources.base import (
    RefreshResult, SourceError, register)

#: Where the export is, when the source is not told explicitly. NO DEFAULT PATH (#211): the one
#: this replaced was an absolute path on the author's machine, and every deployed instance read it
#: because their pool specs pass only `max_open_days`. A default that is a real identity is a wrong
#: answer with good manners — nothing downstream can tell "configured" from "assumed". Unset is a
#: `SourceError` naming this variable; the refresher records it per source and keeps the last set.
ENV_CSV = "KUMO_LEDGER_CSV"
#: ledger-tool writes this AFTER the csv, so its presence means the export finished. It is how a
#: half-failed run is distinguished from a genuinely smaller book — without it, "the export
#: produced 63 of 130 names" and "he really holds 63" are identical on the wire.
META_SUFFIX = ".meta.json"
MAX_META_AGE_HOURS = 30.0



@register
@dataclass
class LedgerBookSource:
    """Names ledger-provider held as of the reference date.

    `max_open_days` models a MISSED SELL — the post stream under-records exits, so a row with no
    close date may already be closed. It applies ONLY to rows without a recorded close: 56.7% of his
    closed trades run past 20 days, and capping every row evicted names he demonstrably still held,
    which cost ~24 points of return before it was found.
    """

    kind = "ledger_book"
    name: str = "ledger_book"
    csv_path: str | None = None
    """Explicit ledger path. `None` reads `KUMO_LEDGER_CSV`; neither is a refusal by name."""
    lag_days: int = 1
    max_open_days: int = 57
    """Backstop only, and UNVALIDATED. Fitting it against ledger-tool's full-review anchors was not
    possible: the error is dominated by unclosed-interval accumulation, not holding period, and
    every value from 20d to 180d carries the same ~+50 bias. It now applies solely to lots that no
    full review ever confirmed."""
    confirm_window_days: int = 45
    """How far behind the newest full review a confirmation may be and still count as current.
    Compared against the sidecar's `latest_full_review`, not against today -- reviews are irregular,
    so measuring against wall-clock would retire names simply because the reviewer took a break."""

    def _csv(self) -> Path:
        """The ledger path: explicit, else the environment, else a refusal that names the variable."""
        if self.csv_path:
            return Path(self.csv_path)
        raw = os.environ.get(ENV_CSV)
        if raw is None or raw.strip() == "":
            raise SourceError(
                f"no ledger path: csv_path not given and {ENV_CSV} is unset. This source has no "
                f"default — set {ENV_CSV} to the export's csv (the instance's env owns it).")
        return Path(raw).expanduser()

    def _check_meta(self, csv: Path) -> dict:
        """Refuse a suspect export rather than treating it as a smaller book.

        A partial run is indistinguishable from a genuine contraction by looking at the rows alone,
        and this source's departures are acted on as SELL signals — so trusting a half-written
        export would liquidate names the trader still holds.
        """
        # ledger-tool writes ledger-book.meta.json — REPLACING .csv, not appending to it. My first
        # version looked for ledger-book.csv.meta.json, found nothing, and returned {} — the guard
        # was there and silently did nothing, which is worse than not having it.
        candidates = [csv.with_suffix(META_SUFFIX), csv.with_suffix(csv.suffix + META_SUFFIX)]
        meta_path = next((c for c in candidates if c.exists()), None)
        if meta_path is None:
            raise SourceError(
                f"no sidecar beside {csv.name} (looked for {[c.name for c in candidates]}). "
                f"Without it a half-failed export is indistinguishable from a smaller book, and "
                f"this source's departures are acted on as sell signals.")
        try:
            m = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise SourceError(f"unreadable sidecar {meta_path}: {e}") from None
        if m.get("status") != "ok":
            raise SourceError(
                f"export status={m.get('status')!r}: {m.get('detail') or 'no detail'} — "
                f"keeping the previous set")
        gen = m.get("generated_at")
        if gen:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(gen)).total_seconds() / 3600
            if age > MAX_META_AGE_HOURS:
                raise SourceError(
                    f"export is {age:.0f}h old (generated {gen}) — the upstream job has not run. "
                    f"Refusing rather than deciding on a stale book.")
        declared = m.get("schema")
        if declared:
            missing = [c for c in declared if c not in self._columns(csv)]
            if missing:
                raise SourceError(f"sidecar declares columns absent from the csv: {missing}")

        # The CSV and the sidecar are SEPARATE atomic writes. Reading between them pairs a new book
        # with the previous sidecar, and every check above then validates the wrong file: status,
        # generated_at and schema all pass while the rows are from a different export. row_count
        # only catches it when the counts happen to differ, which a same-size rewrite does not.
        #
        # This is the same shape as the pool being four hours behind ledger-tool's fix earlier today --
        # each layer internally consistent, the pairing wrong, and nothing able to say so.
        want = m.get("csv_sha256")
        if want:
            got = hashlib.sha256(csv.read_bytes()).hexdigest()
            if got != want:
                raise SourceError(
                    f"csv does not match its sidecar (sha256 {got[:12]} vs declared {want[:12]}) — "
                    f"the two are written separately, so this is a torn read or a partial write. "
                    f"Keeping the previous set.")
        rows = m.get("row_count")
        if rows is not None:
            actual = sum(1 for _ in csv.open()) - 1        # minus the header
            if actual != rows:
                raise SourceError(
                    f"csv has {actual} rows, sidecar declares {rows} — refusing a book that does "
                    f"not match its own manifest")
        return m

    @staticmethod
    def _columns(csv: Path) -> list[str]:
        with csv.open() as f:
            return f.readline().strip().split(",")

    def upstream_changed_at(self) -> datetime | None:
        """When ledger-tool last published, from the sidecar's `generated_at`.

        This is what lets the job runner notice our cache is BEHIND the source rather than merely
        old. The two are different questions and they came apart in practice: a book cached at 18:26
        was well inside its 24h cadence while ledger-tool published a corrected file at 22:20, so the
        pool kept serving a symbol upstream had already fixed and reported itself healthy.

        Returns None rather than raising -- a missing or unreadable sidecar is fetch()'s problem to
        report properly, and it does. Failing here would only turn a fetchable source into a
        silently skipped one, which is the failure mode this whole method exists to remove.
        """
        try:
            csv = self._csv()
            candidates = [csv.with_suffix(META_SUFFIX), csv.with_suffix(csv.suffix + META_SUFFIX)]
            meta_path = next((c for c in candidates if c.exists()), None)
            if meta_path is None:
                return None
            raw = json.loads(meta_path.read_text()).get("generated_at")
            if not raw:
                return None
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except (OSError, json.JSONDecodeError, ValueError, SourceError):
            return None

    def fetch(self, asof: datetime | None = None) -> RefreshResult:
        p = self._csv()
        if not p.exists():
            raise SourceError(f"ledger not found: {p}")
        meta = self._check_meta(p)
        try:
            b = pd.read_csv(p)
        except Exception as e:                                     # noqa: BLE001
            raise SourceError(f"unreadable ledger {p}: {e}") from e
        for col in ("symbol", "open_date", "close_date"):
            if col not in b.columns:
                raise SourceError(f"ledger missing column {col!r}")

        b["open_date"] = pd.to_datetime(b["open_date"], errors="coerce")
        b["close_date"] = pd.to_datetime(b["close_date"], errors="coerce")
        b = b.dropna(subset=["open_date"])
        ref = pd.Timestamp((asof or datetime.now(timezone.utc)).date()) - pd.Timedelta(days=self.lag_days)

        open_now = b["close_date"].isna() | (b["close_date"] > ref)
        m = (b["open_date"] <= ref) & open_now
        unclosed = b["close_date"].isna()
        m &= ~unclosed | ((ref - b["open_date"]) <= pd.Timedelta(days=self.max_open_days))

        # CONFIRMATION BEATS AGE. `last_confirmed_held` is the date of the most recent FULL review
        # that confirmed this lot held, scoped to the lot's own life. A name confirmed at the corpus
        # edge is held, however long it has been open -- which is the cohort expiry could never
        # keep: AEM, CGAU, SCCO, WPM are long holds he simply stops posting about, and an age-based
        # rule retires exactly those.
        #
        # Expiry survives only as the backstop for lots that were never confirmable. That is the
        # honest scope for a constant I have no way to validate: it now decides the names no
        # evidence covers, instead of overruling the evidence.
        latest = meta.get("latest_full_review")
        if "last_confirmed_held" in b.columns and latest:
            conf = pd.to_datetime(b["last_confirmed_held"], errors="coerce")
            edge = pd.Timestamp(latest)
            fresh = conf.notna() & ((edge - conf) <= pd.Timedelta(days=self.confirm_window_days))
            # re-admit anything the age rule dropped but a recent full review still confirms
            m |= (b["open_date"] <= ref) & open_now & fresh
        held = b.loc[m]
        if held.empty:
            raise SourceError(f"no open positions as of {ref.date()} — refusing to empty the pool")
        syms = {str(r.symbol): {"opened": str(r.open_date.date()),
                                "days_held": int((ref - r.open_date).days)}
                for r in held.itertuples()}
        through = meta.get("source_through")
        return RefreshResult(
            self.name, syms,
            f"{len(syms)} held as of {ref.date()} (lag {self.lag_days}d, "
            f"expiry {self.max_open_days}d"
            + (f", corpus through {through}" if through else "") + ")")
