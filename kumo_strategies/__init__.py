"""Executable Kumo strategy implementations and runtime contracts."""

from kumo_strategies.provenance import digest, digest_tree, files

__all__ = ["__version__", "digest", "digest_tree", "files"]

#: NOT a build identity. It has never changed and cannot answer "what code is this container
#: running" -- use `digest()`, which measures the installed bytes. See `provenance`.
__version__ = "0.1.0"
