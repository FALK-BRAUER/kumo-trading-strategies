"""`bin/check-public-tree.sh` refuses what a public tree must never hold — proven per rule (#1045).

Each test plants ONE violation in a throwaway git repo and asserts the script refuses AND names the
exact file:line. A refusal that does not name the line is a refusal nobody can act on, and a script
that refuses on a planted hit but names the wrong one would pass a vaguer test.

Three properties the per-rule tests cannot cover on their own, so they get their own tests:

- CLEAN PASSES. Without it, a script that always refuses passes every test above.
- `.public-exclude` IS HONOURED. The private tree carries the operating record by design; the check
  must skip exactly what the export omits, and nothing else — a planted hit under an excluded path
  passes, the same hit one directory over refuses.
- A DENYLIST THAT IS NAMED BUT MISSING REFUSES. "0 hits" from a check that silently skipped its one
  literal rule is the confident wrong answer this whole file exists to prevent.

The repo is built with real `git` because the script uses `git grep` / `git ls-files`: a fixture that
wrote files without tracking them would test nothing — the script cannot see an untracked file.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "bin" / "check-public-tree.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "tree"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    # The script excludes ITSELF from the scan by path, so it must exist at the same relative path.
    (r / "bin").mkdir()
    (r / "bin" / "check-public-tree.sh").write_text(SCRIPT.read_text())
    (r / "README.md").write_text("# clean\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "clean")
    return r


def plant(repo: Path, rel: str, body: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    _git(repo, "add", "-A")


def run(repo: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    e = {k: v for k, v in os.environ.items() if k != "KUMO_PUBLIC_DENYLIST"}
    if env:
        e.update(env)
    return subprocess.run(["bash", str(SCRIPT), str(repo)], capture_output=True, text=True, env=e)


def test_the_script_under_test_is_the_one_in_bin() -> None:
    # The fixture copies the real script in; if the path moves this file tests a stale copy of nothing.
    assert SCRIPT.is_file(), SCRIPT


def test_a_clean_tree_passes_and_says_what_it_scanned(repo: Path) -> None:
    r = run(repo)
    assert r.returncode == 0, r.stderr
    assert "10 checks, 0 hits" in r.stdout
    # It counts what it scanned, so "0 hits" over 0 files cannot read as clean.
    assert "1 tracked files scanned" in r.stdout  # README.md; the script itself is excluded


# THE PLANTED STRINGS ARE ASSEMBLED, NOT WRITTEN. This file is tracked in the same tree the script
# scans, so a literal `DU` + seven digits here would be a hit in the test that proves hits are caught
# — the first version of this file did exactly that and refused its own commit. Every planted value
# is built from parts at import time; none of them is an account id, host, token or key of anything.
_IB_ID_A = "DUP" + "7" * 6
_IB_ID_B = "DU" + "1234567"
_UUID = "0123abcd-89ab" + "-4cde-8f01-0123456789ab"
_HOST = "some-host" + ".tail" + "abc123" + ".ts.net"
_TOKEN = "123456789" + ":" + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
_KEY_ID = "PK" + "ABCDEFGHIJKLMNOPQR"
_PEM = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
_LIVE_ID = "U" + "7654321"
_HOME = "/Users/" + "someone" + "/projects/x"
_LAN = "192.168." + "1.23"


@pytest.mark.parametrize(
    ("rel", "body", "label"),
    [
        ("backend/x.py", f'ACCOUNT = "{_IB_ID_A}"\n', "IB paper account id shape found:"),
        ("backend/x.py", f'ACCOUNT = "{_IB_ID_B}"\n', "IB paper account id shape found:"),
        ("deploy/.env", f"ALPACA_ACCOUNT_ID={_UUID}\n", "account uuid assigned"),
        ("ui/c.test.ts", f'load("https://{_HOST}:3000/")\n', "private hostname found:"),
        ("backend/n.py", f'TOKEN = "{_TOKEN}"\n', "telegram bot token shape found:"),
        ("backend/a.py", f'KEY = "{_KEY_ID}"\n', "alpaca key id shape found:"),
        ("deploy/id_rsa", f"{_PEM}\n", "private key material found:"),
        # The three the first version was blind to (peer review of #1081):
        ("deploy/.env", f"IBKR_ACCOUNT_ID={_LIVE_ID}\n", "IB live account id shape found:"),
        ("deploy/compose.yml", f"      - {_HOME}:{_HOME}:ro\n", "home-directory path found:"),
        ("ui/c.test.ts", f'load("http://{_LAN}:3000/")\n', "private network address found:"),
    ],
)
def test_each_shape_rule_refuses_and_names_the_line(repo: Path, rel: str, body: str, label: str) -> None:
    plant(repo, rel, body)
    r = run(repo)
    assert r.returncode == 1, r.stdout + r.stderr
    assert label in r.stderr
    assert f"{rel}:1:" in r.stderr, r.stderr  # the exact file AND line


def test_an_id_shape_inside_a_longer_word_is_not_a_hit(repo: Path) -> None:
    # The word edges are the whole point of spelling `[^A-Za-z0-9_]` out: `DUTEST001` and `xDU1234567y`
    # are not account ids, and a rule without edges would refuse every fixture the scrub introduced.
    plant(repo, "backend/x.py", f'A = "DUTEST001"; B = "x{_IB_ID_B}y"\n')
    r = run(repo)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    ("rel", "label"),
    [
        ("data/soxx.csv", "market-data file tracked"),
        ("zz_handoffs/2026-01-01.md", "operating record tracked:"),
        ("FOR_SOMEONE.md", "operating record tracked:"),
        ("instances/x/RUNBOOK.md", "operating record tracked:"),
    ],
)
def test_data_and_operating_record_paths_refuse_and_are_named(repo: Path, rel: str, label: str) -> None:
    plant(repo, rel, "x\n")
    r = run(repo)
    assert r.returncode == 1
    assert label in r.stderr
    assert rel in r.stderr


def test_a_csv_under_a_fixtures_directory_is_allowed(repo: Path) -> None:
    plant(repo, "backend/tests/fixtures/bars.csv", "t,o,h,l,c\n")
    assert run(repo).returncode == 0


def test_public_exclude_skips_exactly_the_listed_paths(repo: Path) -> None:
    plant(repo, ".public-exclude", "# private\nzz_handoffs/\nCLAUDE.md\n")
    plant(repo, "zz_handoffs/h.md", f"login {_IB_ID_A}\n")   # excluded: the export omits it
    plant(repo, "CLAUDE.md", f"host {_HOST}\n")              # excluded
    r = run(repo)
    assert r.returncode == 0, r.stderr
    assert "2 exclusion(s)" in r.stdout
    # THE SAME HIT ONE DIRECTORY OVER STILL REFUSES — the exclusion is a list of paths, not a licence.
    plant(repo, "docs/h.md", f"login {_IB_ID_A}\n")
    r = run(repo)
    assert r.returncode == 1
    assert "docs/h.md:1:" in r.stderr
    assert "zz_handoffs/h.md" not in r.stderr


def test_denylist_terms_refuse_case_insensitively_and_name_the_line_but_not_the_content(repo: Path, tmp_path: Path) -> None:
    deny = tmp_path / "deny.txt"
    deny.write_text("someLogin42\nmy_pool_name\n")
    plant(repo, "backend/p.py", 'SOURCES = ("MY_POOL_NAME",)\n')
    r = run(repo, {"KUMO_PUBLIC_DENYLIST": str(deny)})
    assert r.returncode == 1
    assert "denylist term found at:" in r.stderr
    assert "backend/p.py:1" in r.stderr
    # FILE:LINE ONLY. The shape rules print the line because a shape is not the secret; a denylist
    # term IS, and a refusal in a public CI log that quotes it is a second channel for it.
    assert "MY_POOL_NAME" not in r.stderr
    assert "SOURCES" not in r.stderr


def test_a_denylist_term_in_a_tracked_filename_is_a_hit(repo: Path, tmp_path: Path) -> None:
    # git grep reads content only. A file NAMED after a login or a host passed the first version.
    deny = tmp_path / "deny.txt"
    deny.write_text("someLogin42\n")
    plant(repo, "docs/notes-somelogin42.md", "nothing inside\n")
    r = run(repo, {"KUMO_PUBLIC_DENYLIST": str(deny)})
    assert r.returncode == 1
    assert "denylist term in a tracked FILENAME:" in r.stderr
    assert "docs/notes-somelogin42.md" in r.stderr


def test_a_named_but_missing_denylist_refuses_rather_than_skipping(repo: Path, tmp_path: Path) -> None:
    r = run(repo, {"KUMO_PUBLIC_DENYLIST": str(tmp_path / "nope.txt")})
    assert r.returncode == 1
    assert "no such file" in r.stderr


def test_the_denylist_file_itself_is_never_inside_the_tree(repo: Path) -> None:
    # If the denylist were tracked, the scan would find every term in it and refuse the tree for
    # carrying the list of what it must not carry — and exporting it would publish the list. The
    # script reads it from OUTSIDE via the env var; this pins that a tracked copy is a hit.
    plant(repo, "bin/denylist.txt", "someLogin42\n")
    deny = repo.parent / "deny.txt"
    deny.write_text("someLogin42\n")
    r = run(repo, {"KUMO_PUBLIC_DENYLIST": str(deny)})
    assert r.returncode == 1
    assert "bin/denylist.txt:1" in r.stderr


def test_an_exclusion_that_empties_the_scan_refuses(repo: Path) -> None:
    # `*.md` in .public-exclude drops README.md — the only tracked file besides the script — and the
    # first version printed "0 hits" over nothing, with a clean line. 0 of 0 is not clean.
    plant(repo, ".public-exclude", "*.md\n.public-exclude\n")
    r = run(repo)
    assert r.returncode == 1
    assert "nothing left to scan" in r.stderr


def test_a_fatal_pathspec_in_the_exclusion_list_refuses_rather_than_passing_every_scan(repo: Path) -> None:
    # `../x` is outside the repo; git grep exits ≥2 on it. The first version read every non-zero exit as
    # "no hit", so one bad exclude line made ALL scans silently pass — the peer planted an id behind it
    # and the tree came back clean.
    plant(repo, ".public-exclude", "../x\n")
    plant(repo, "backend/x.py", f'ACCOUNT = "{_IB_ID_A}"\n')
    r = run(repo)
    assert r.returncode == 1
    assert "could not run" in r.stderr


def test_a_tracked_symlink_refuses_even_when_its_target_is_the_only_hit(repo: Path) -> None:
    # git grep reads a symlink as a zero-length blob: a link whose TARGET is a home-directory path
    # passed every content rule (peer review of #1083, M2). The target is the leak and no scan reads
    # it, so the link itself is refused.
    (repo / "docs").mkdir(exist_ok=True)
    os.symlink(_HOME + "/private/notes.md", repo / "docs" / "link.md")
    _git(repo, "add", "-A")
    r = run(repo)
    assert r.returncode == 1
    assert "symlink tracked" in r.stderr
    assert "docs/link.md" in r.stderr


def test_an_identifier_inside_base64_encoded_content_is_a_hit(repo: Path, tmp_path: Path) -> None:
    # Two production-captured Redis dumps carried the real account ids inside base64 msgpack blobs,
    # and every shape rule, trufflehog and gitleaks passed over them (2026-09-14). Encoded content
    # is content: the run is decoded and held to the same shapes and denylist.
    import base64
    deny = tmp_path / "deny.txt"
    deny.write_text("someLogin42\n")
    padding = b" " * 40   # non-word bytes: the word edges in the shapes are real, an id glued to letters is not one
    blob_term = base64.b64encode(padding + b"account someLogin42 held" + padding).decode()
    blob_shape = base64.b64encode(padding + b"INTERACTIVE_BROKERS-" + _IB_ID_A.encode() + padding).decode()
    plant(repo, "backend/api/fixtures/dump.json", f'{{"a": "{blob_term}", "b": "{blob_shape}"}}\n')
    r = run(repo, {"KUMO_PUBLIC_DENYLIST": str(deny)})
    assert r.returncode == 1
    assert "inside BASE64-encoded content" in r.stderr
    assert "dump.json (base64 run #1): somelogin42" in r.stderr
    assert "dump.json (base64 run #2): shape" in r.stderr
    # and the SAME blobs under an excluded path are skipped, like every other rule
    plant(repo, ".public-exclude", "backend/api/fixtures/\n.public-exclude\n")
    r = run(repo, {"KUMO_PUBLIC_DENYLIST": str(deny)})
    assert r.returncode == 0, r.stderr


def test_ordinary_base64_content_is_not_a_hit(repo: Path) -> None:
    # A png or a font in base64 must not refuse the tree — only decoded content that matches a rule.
    import base64
    plant(repo, "ui/src/logo.ts", f'export const LOGO = "{base64.b64encode(bytes(range(256)) * 2).decode()}";\n')
    r = run(repo)
    assert r.returncode == 0, r.stderr
