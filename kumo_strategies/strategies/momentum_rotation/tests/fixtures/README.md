# fixtures

Small, committed cuts of research data that tests drive the engine over, so a test never reads
`research/` at run time (ks#211 moves research data out of the public tree).

Goes here: one CSV per fixture, named `<ticker>_<start>_to_<end>_<what>.csv`, with the cut's
provenance in the test that uses it. Does not go here: anything a test could generate itself, or
any file larger than a few KB.
