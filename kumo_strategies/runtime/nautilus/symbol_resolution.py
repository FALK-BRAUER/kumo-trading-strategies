"""Symbol -> InstrumentId, resolved through Nautilus on whatever broker is attached (platform issue 622).

Shared by every rotation strategy, because the alternative is four copies of a rule that must not
drift — and this repo's recurring defect is two derivations of one fact disagreeing.

WHY THIS EXISTS AT ALL. `instrument_ids: list[InstrumentId]` demanded *symbol + venue* at
CONSTRUCTION, and the venue half is a runtime fact only a connected adapter can produce. Cockpit met
that obligation the only way it could offline — by importing Alpaca's asset list into a strategy — so
an IBKR-only instance could not build a single lane without an Alpaca API key. Measured twice on
ibkr-paper-retired: `build_node()` raised, 13 restarts, `/positions` served [] against 22 non-flat
positions held at the broker.

A SYMBOL is what this layer knows. The pool is a research artifact; it has no opinion about XNAS vs
XNYS and must not be forced to have one.

WHY `on_start` IS WHERE IT WORKS, read from the installed nautilus_trader rather than assumed:

    system/kernel.py:1022   self._connect_clients()
    system/kernel.py:1024   await self._await_engines_connected()    <- awaits connect
    system/kernel.py:1039   self._trader.start()                     <- on_start runs here

    adapters/interactive_brokers/data.py:147
        await self.instrument_provider.initialize()
        for instrument in self._instrument_provider.list_all():
            self._handle_data(instrument)                            <- into the Cache

So by `on_start` every declared instrument is in the Cache — on a COLD start, with no durable cache
and no HTTP call. Resolving from a WARM cache at build time was considered and rejected: it works
only because a previous run connected, so it fixes one boot and re-breaks on the next.
"""

from __future__ import annotations


class SymbolResolutionMixin:
    """`symbols` in, `_iids` out, resolved at `on_start` from the attached adapter."""

    def _init_symbols(self, instrument_ids, symbols) -> None:
        """Exactly one of the two. RAISES on both or neither.

        A default silently picks a path, which is how a half-migrated caller looks healthy — the
        exact failure shape this ticket is about. `instrument_ids` stays supported so the pin can
        move without changing the tenant that is actually trading.
        """
        if (instrument_ids is None) == (symbols is None):
            raise ValueError(
                f"{type(self).__name__} needs exactly one of `symbols` or `instrument_ids`. "
                "`symbols` is preferred: an id demands a venue at construction, which only a "
                "connected adapter can supply (kumo-trading-platform issue 622)."
            )
        self._symbols = list(symbols) if symbols is not None else []
        self._iids = list(instrument_ids) if instrument_ids is not None else []
        self._unresolved_symbols: list[str] = []
        #: {symbol: [every candidate id]} for symbols the venue lists more than once. Populated even
        #: when the ambiguity is resolved WELL — a correct pick and no pick are equally silent
        #: otherwise, and 64 symbols carrying two identities is a fact an operator should be able to
        #: read back rather than something absorbed quietly (kumo-trading-platform issue 625).
        self._ambiguous_symbols: dict[str, list[str]] = {}

    def _resolve_symbols(self, cache, prefer=None) -> None:
        """DROP AND NAME, never default and never raise.

        Defaulting was the original defect: an id whose venue was guessed matches no loaded
        instrument, so bars never arrive, warmup never completes and any order is refused — while the
        lane looks healthy. Raising was worse: `JEPO`, a vision misread of JEPQ, crash-looped the
        engine and took every other strategy and the entire UI feed with it. One bad row costs that
        row.

        The dropped names land in `_unresolved_symbols` as well as the log, because `self.log` is
        Nautilus's logger and nothing in-process can read it back — which is how #613's inert node
        stayed invisible for 26 minutes.
        """
        if not self._symbols:
            return                                    # constructed from ids; nothing to resolve

        # GROUPED, NOT FLATTENED. This was `{i.symbol.value: i for i in ...}`, a dict comprehension,
        # so a symbol the venue lists twice silently kept whichever the cache iterated LAST —
        # arbitrary, and not stable across boots. Measured on an IBKR paper instance 2026-08-28: 64 symbols carry
        # two identities, every pair a real venue plus a spurious XNAS, because IB is asked with
        # `exchange="SMART"` and answers with contract details per listing. That is IB being
        # correct; the defect was entirely what we did with the answer.
        #
        # It cost real requests: BCTROT asked for `SPY.XNAS`, which matches an instrument with no
        # data, and the venue answered with an empty array and no error.
        candidates: dict[str, list] = {}
        for iid in cache.instrument_ids():
            candidates.setdefault(iid.symbol.value, []).append(iid)

        resolved, missing, ambiguous = [], [], {}
        for sym in self._symbols:
            found = candidates.get(sym) or []
            if not found:
                missing.append(sym)
                continue
            if len(found) == 1:
                resolved.append(found[0])
                continue
            pick = self._prefer_one(sym, found, prefer)
            ambiguous[sym] = [str(i) for i in sorted(found, key=str)]
            resolved.append(pick)

        self._iids = resolved
        self._unresolved_symbols = missing
        self._ambiguous_symbols = ambiguous
        if ambiguous:
            self.log.warning(
                f"{self.id}: {len(ambiguous)} symbols are listed on more than one venue here. One "
                f"identity was chosen per symbol and the others are NOT traded: "
                + "; ".join(f"{s} -> {next(str(i) for i in resolved if i.symbol.value == s)} "
                            f"(of {', '.join(v)})" for s, v in sorted(ambiguous.items()))
            )
        if missing:
            self.log.error(
                f"{self.id}: {len(missing)} of {len(self._symbols)} symbols have no instrument on "
                f"this venue and will NOT be traded: {', '.join(missing)}"
            )
        if not resolved:
            # A lane that can buy nothing is broken; the NODE is not. Reported, not raised — the
            # equivalent failure used to happen inside build_node and killed everything with it.
            self.log.error(
                f"{self.id}: not one pool symbol resolved to an instrument on this venue. The lane "
                f"can place no orders. Other strategies and the data feed are unaffected."
            )

    def _prefer_one(self, sym: str, found: list, prefer):
        """Choose one identity for a symbol the venue lists several times.

        THE PREFERENCE IS INJECTED AND THIS MODULE DOES NOT KNOW THE VENDOR. IB answers the question
        properly — `Instrument.info["contract"]["primaryExchange"]` is `ARCA` for SPY, so the right
        rule is "keep the identity whose venue is the primary listing". But comparing that to a venue
        needs `exchange_to_mic_venue`: `"ARCA"` is not `"ARCX"`, and a naive string compare would
        match NOTHING, fall through to the fallback on every symbol, and look exactly like a working
        preference. That mapping lives in `nautilus_trader.adapters.interactive_brokers`, which
        imports `ibapi` — not installed here, and it must not be.

        Teaching this module IB's exchange names would re-create kumo-trading-platform issue 622 inside the resolver
        written to fix it. The layer that knows the RULE is not the layer that knows the VENDOR. So
        cockpit supplies `prefer`, next to the `exchange="SMART"` declaration that causes the
        duplicates in the first place.

        THE FALLBACK IS DETERMINISTIC AND IS NEVER THE ANSWER. `sorted()[0]` gives ARCX for SPY by
        luck and XNAS for RVTY by bad luck. Its only job is that the wrong answer is the SAME wrong
        answer every boot: instrument ids are what positions and reconciliation key on, so a
        stable-wrong id is recoverable and one that moves under a restart is not.
        """
        fallback = sorted(found, key=str)[0]
        if prefer is None:
            return fallback
        try:
            chosen = prefer(sym, list(found))
        except Exception as exc:                                       # noqa: BLE001
            # REPORTED, NOT SWALLOWED. A preference that raises and is quietly caught degrades to
            # the arbitrary pick while looking principled — the failure is then invisible precisely
            # because a plausible answer still comes out.
            self.log.error(
                f"{self.id}: the venue preference raised on {sym} ({exc!r}); falling back to "
                f"{fallback}. The pick is deterministic but NOT informed.")
            return fallback
        if chosen not in found:
            # Includes `None`, which is how a preference says "I cannot tell" — a real answer, and
            # one that must still leave a trace rather than passing for a decision.
            self.log.error(
                f"{self.id}: the venue preference did not choose among the candidates for {sym} "
                f"(returned {chosen!r}); falling back to {fallback}.")
            return fallback
        return chosen

    def _resolve_symbols_if_needed(self) -> None:
        """Call at the TOP of `on_start`.

        Guarded here rather than only inside `_resolve_symbols`, because `self.cache` is a read-only
        Cython attribute on Nautilus's Actor: merely EVALUATING it raises on any host that has none.
        A strategy constructed from ids has nothing to resolve and must not need one.
        """
        if getattr(self, "_symbols", None):
            self._resolve_symbols(self.cache, getattr(self, "_prefer_venue", None))
