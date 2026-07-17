"""cargo --message-format=json fields -> standard build-diagnostic terms.

The vocabulary is counts only, on purpose: the digest tells WHERE errors and
warnings concentrate (which crate; per-file via ``expand``); reading the
actual diagnostic TEXT is the agent's editor/compiler job, not perfdigest's
(translator, not judge — token efficiency is the product). There is no timing
vocabulary here at all because the stream carries none (see the reader
docstring): ``duration_us`` stays honestly absent, never 0.0.

Genuine-zero rule (same as ptxas's ``Used ...`` line): the message stream is a
complete enumeration of the diagnostics this build emitted, so a crate that
appears in the stream with no messages has a MEASURED ``errors``/``warnings``
of 0.0 — not a gap.
"""

from __future__ import annotations

# standard term -> the cargo-record arithmetic behind it (the `expand` hint)
STANDARD_TO_VENDOR: dict[str, str] = {
    "errors": "count of compiler-message records for this crate whose "
    "message.level starts with 'error' (plain errors and internal compiler "
    "errors alike)",
    "warnings": "count of compiler-message records with message.level == "
    "'warning'",
    "notes_helps": "count of compiler-message records with any other level "
    "(note, help, failure-note e.g. rustc's 'aborting due to previous error')",
}

# What get_metrics(metrics=None) returns for a cargo-diag unit.
DEFAULT_CORE_SET: list[str] = [
    "errors",
    "warnings",
    "notes_helps",
]
