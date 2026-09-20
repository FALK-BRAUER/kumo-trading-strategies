# Nautilus Adapter Notes

Cockpit currently pins `nautilus_trader==1.229.0`; this repo's optional `nautilus` dependency matches that pin.

Implementation rules:

- Use Context7 before adding real Nautilus API code.
- Keep pure decision logic independent from Nautilus classes.
- Preserve Cockpit's strategy identity contract and explicit `cycle_id` projection.
- Do not treat Nautilus native position ids as cycle ids; they are strategy-level position identifiers.
- Every submitted order must have exactly one strategy owner.

The adapter placeholder exists so strategy code can evolve under test before being bound to runtime-specific APIs.
