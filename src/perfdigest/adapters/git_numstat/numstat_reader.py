"""Read a saved ``git diff --numstat`` export -> NormalizedUnit (repo_change).
Pure Python, no deps.

RepoState semantics (owner directive): a token-efficient BACKWARD look at what
a working session did to a repo, so a re-orienting agent (post-compaction, new
session) does not re-read raw diffs. The raw diff is the token sink; numstat
is its digest. This reader emits FACTS (paths, line counts) and never
verdicts — "is this footprint releasable" is the model's conclusion.

Artifact-first: we bind to a SAVED file produced by::

    git diff --numstat -M <base>..<tip> > session.numstat   # a session's range
    git diff --numstat -M > session.numstat                 # uncommitted only

Line grammar, verified against REAL ``git diff --numstat -M`` output
(git 2.53, throwaway-lab capture committed as
``tests/fixtures/repo_lab_renames_sample.numstat``)::

    <added>\\t<deleted>\\t<path spec>

  * added/deleted are decimal counts, or ``-`` for a BINARY file. The dash is
    the headline honesty case: a binary has no line counts, so both metrics
    stay ``None`` -> 'not_available_in_this_export' — NEVER 0.0. A genuine
    ``0`` (e.g. a pure rename with no edits) is preserved as measured 0.0.
  * The path spec has three real shapes:
      - plain path              (spaces verbatim: ``docs/read me.txt``)
      - ``old => new``          (whole-path rename: ``NOTES => TASKS.md``)
      - ``pre{old => new}post`` (partial rename: ``src/{a.py => b.py}``,
        ``{src => docs}/util.py``, ``docs/{ => api}/config.py`` — either brace
        side may be empty; rebuilding a path over an empty side can produce
        ``//``, which git's own display collapses, so we do too)
    A renamed unit is NAMED BY ITS NEW PATH (where the file lives now — the
    reorientation answer); the old path stays reachable via ``expand``.
  * Non-ASCII paths arrive C-quoted by git (``"docs/\\303\\266l\\303\\247..."``,
    core.quotePath default) and are kept VERBATIM as printed — the name is
    honest to the artifact; we do not unquote.
  * Known ambiguity, inherent to the unquoted artifact: a plain file whose
    NAME literally contains ``" => "`` (``notes about a => b.txt``) is
    indistinguishable from a whole-path rename in numstat output — git prints
    both identically — and parses as a rename here. git's own limitation;
    ``expand`` shows ``path_as_printed`` so the raw line is always recoverable.

``duration_us`` is ``None`` for every unit: a change digest has no time
dimension at all, so ``summarize_report`` falls back to file order and ranking
by size is done by reading ``lines_added`` from the metrics — a fact the agent
sorts by, not a hotness verdict we bake in.
"""

from __future__ import annotations

import re
from typing import Any

from perfdigest.core.metrics import DOMAIN_REPO_CHANGE, NormalizedUnit

_LINE = re.compile(r"^(?P<added>\d+|-)\t(?P<deleted>\d+|-)\t(?P<path>.+)$")
# One brace group per line is all real git emits (the common prefix/suffix
# around the single differing middle segment).
_BRACED = re.compile(r"^(?P<pre>.*)\{(?P<old>.*) => (?P<new>.*)\}(?P<post>.*)$")


def _count(field: str) -> float | None:
    """A numstat count column: int, or None for the binary-file dash."""
    return None if field == "-" else float(field)


def _split_path(path_spec: str) -> tuple[str, str | None]:
    """-> (new_path, old_path or None if not a rename)."""
    m = _BRACED.match(path_spec)
    if m:
        new = f"{m['pre']}{m['new']}{m['post']}".replace("//", "/")
        old = f"{m['pre']}{m['old']}{m['post']}".replace("//", "/")
        return new, old
    if " => " in path_spec:
        old, _, new = path_spec.partition(" => ")
        return new, old
    return path_spec, None


def _parse(report_path: str) -> list[dict[str, Any]]:
    with open(report_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    records: list[dict[str, Any]] = []
    malformed = 0
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        m = _LINE.match(raw_line)
        if not m:
            malformed += 1
            continue
        new_path, old_path = _split_path(m["path"])
        records.append(
            {
                "path": new_path,
                "old_path": old_path,
                "path_as_printed": m["path"],
                "lines_added": _count(m["added"]),
                "lines_deleted": _count(m["deleted"]),
            }
        )

    if not records:
        raise ValueError(
            f"{report_path} does not look like a `git diff --numstat` export: no "
            "line matched the expected '<added>\\t<deleted>\\t<path>' shape "
            "(tab-separated counts, '-' for binary files). An empty diff also "
            "produces an empty file — if the session made no changes there is "
            "nothing to digest. Produce this artifact with: "
            "git diff --numstat -M <base>.. > session.numstat"
        )
    if malformed:
        # Mixed content (a numstat pasted into some other log) digests the
        # matching lines but must not do so silently.
        raise ValueError(
            f"{report_path}: {malformed} line(s) do not match the numstat "
            f"'<added>\\t<deleted>\\t<path>' shape next to {len(records)} that "
            "do — this file is not a clean `git diff --numstat` export. "
            "Re-capture with: git diff --numstat -M <base>.. > session.numstat"
        )
    return records


def load_units(report_path: str) -> list[NormalizedUnit]:
    units: list[NormalizedUnit] = []
    for index, rec in enumerate(_parse(report_path)):
        units.append(
            NormalizedUnit(
                name=rec["path"],
                index=index,
                duration_us=None,  # a change digest has no time dimension
                raw_ref=report_path,
                metrics={
                    "lines_added": rec["lines_added"],
                    "lines_deleted": rec["lines_deleted"],
                },
                domain=DOMAIN_REPO_CHANGE,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """The parsed line behind one unit — old path and rename/binary facts."""
    records = _parse(report_path)
    if kernel_index >= len(records):
        raise IndexError(f"unit index {kernel_index} not present in {report_path}")
    rec = records[kernel_index]
    out: dict[str, Any] = {
        "path": rec["path"],
        "path_as_printed": rec["path_as_printed"],
        "renamed": rec["old_path"] is not None,
        "old_path": rec["old_path"],
        "is_binary": rec["lines_added"] is None and rec["lines_deleted"] is None,
        "lines_added": rec["lines_added"],
        "lines_deleted": rec["lines_deleted"],
    }
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    if wanted is not None:
        out = {k: v for k, v in out.items() if wanted in k.lower()}
    return out
