"""Package marker.

Two test files share the basename `test_engine.py` — one here, one in the sibling strategy
package. Without `__init__.py` pytest derives a module name from the basename alone, so the second
one collides with the first and collection ABORTS THE WHOLE SUITE rather than skipping a file.

That surfaced when momentum-003 merged alongside qc345: both branches were green in isolation and
main was broken by the combination.
"""
