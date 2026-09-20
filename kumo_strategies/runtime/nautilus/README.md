# nautilus

The SHARED NautilusTrader adapter layer: `adapter.py`, `broker.py` (`NautilusBroker`), `contract.py`
(registration, session outcome, protective close), and the mixins every lane composes (`held_seed`,
`market_aware`, `slot_reread`, `symbol_resolution`, `order_provenance`, `sides`, `capabilities`).
Keep it thin: translate pure decisions into Nautilus strategy/config/order APIs, preserve
strategy/cycle attribution. Use Context7 before implementing real Nautilus APIs.

The lanes themselves live in `strategies/<name>/nautilus.py` (ks#211). `bctrot_rotation.py`,
`crsi_short.py`, `momentum_rotation.py`, `momentum_rotation_intraday.py`,
`qc27_rotation.py`, `qc345_rotation.py`, `smhgld_sleeve.py`, `template_rotation.py` here are
re-export shims — cockpit imports those paths — and must hold no code.
