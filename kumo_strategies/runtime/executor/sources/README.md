# sources

Pluggable contributors to the symbol pool. Each owns a set of symbols and knows how to refresh it.

| kind | what it contributes |
|---|---|
| `ledger_book` | names ledger-provider held as of a reference date, from ledger-tool's nightly export |
| `static_list` | a hand-maintained watchlist, inline or from a file |

Two rules a source must obey:

- **`fetch()` returns the WHOLE set**, never add/remove deltas. The pool replaces that source's
  contribution wholesale, which is what makes a refresh idempotent.
- **A source that cannot fetch RAISES `SourceError`.** Returning an empty set would silently empty
  the pool — far more dangerous than running on a stale one.

Add a kind by decorating a dataclass with `@register` and giving it `kind`, `name` and `fetch()`.

Does NOT hold: the pool itself (→ `../pool.py`) or override logic — a source contributes, the
operator's pins and blacklist sit above it.
