# families

One module per strategy family the session engine runs: `rotation` (momentum / BCT, two clocks), `monthly` (QC27 / QC345), `short` (CRSISHORT). A family owns scoring, decision, exits, sizing and the fill rule; the engine owns the loop, the book, the venue and the report. Nothing in here touches cash.
