"""The template lane must exist and must conform on day one (2026-08-22).

    "We need integration tests, protocols, interfaces, templates... At the time a strategy gets wired
     into trading it needs to be ready."

THE ARGUMENT IS IN THE DATA, not in tidiness. kumo-trading-platform measured 19 decided slots since 2026-07-31,
10 of the 17 live ones bad — and EIGHT OF THE TEN fall on 08-19..08-21, the three days in which three
hand-built lanes were added. MOMENTUM, the oldest, is the only lane that trades reliably. That is a
manufacturing-defect distribution rather than a bug distribution.

There is no scaffold in this repo. Five strategies were each hand-built from whichever one someone
happened to read, and the result is four `run()` signatures, three journal schemas, two ownership
conventions, two names for equity, and a contract class defined twice.

A template does not detect any of that. It makes a new lane START conforming, so a divergence has to
be introduced deliberately instead of inherited by accident.
"""

from __future__ import annotations


def test_a_template_lane_exists():
    from kumo_strategies.strategies.template.nautilus import TemplateRotationStrategy

    assert TemplateRotationStrategy.EXTERNAL_ID == "TEMPLATE"


def test_the_template_is_covered_by_the_DEPLOYMENT_CONFORMANCE_suite():
    """It must be discovered by the SAME machinery real lanes are, not by a bespoke test. A template
    checked by its own private assertions proves only that its private assertions pass."""
    from kumo_strategies.strategies._layout import lanes   # what test_contract parametrizes over

    names = {c.__name__ for c in lanes()}
    assert "TemplateRotationStrategy" in names, (
        f"the template is not discovered by the conformance suite, so it proves nothing about what a "
        f"new lane inherits. Found: {sorted(names)}")


def test_the_template_is_NOT_registrable_as_a_real_lane():
    """It conforms like a lane so the suite can check it, and must never be TRADED. Cockpit's registry
    is an explicit tuple, so it cannot be auto-registered — this pins the intent from this side."""
    from kumo_strategies.strategies.template.nautilus import TemplateRotationStrategy

    assert TemplateRotationStrategy.EXTERNAL_ID.startswith("TEMPLATE"), (
        "the external id must be recognisably a template, so a registry entry for it is obviously "
        "wrong to a reader")
    assert getattr(TemplateRotationStrategy, "IS_TEMPLATE", False) is True


def test_the_template_pure_layer_is_importable_without_nautilus():
    """The pure/runtime split every strategy here follows: decisions must be testable and backtestable
    with no engine present."""
    import importlib

    m = importlib.import_module("kumo_strategies.strategies.template")
    assert hasattr(m, "decide") and hasattr(m, "TemplateConfig")
