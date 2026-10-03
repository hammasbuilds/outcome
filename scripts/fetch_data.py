"""Pull both ECtHR tasks from the LexGLUE parquet conversion.

    python scripts/fetch_data.py
    OUTCOME_DATA=/somewhere/else python scripts/fetch_data.py

Interrupted? Run it again: each file resumes from its ``.part``.

Six files, 102 MB, ungated. `coastalcph/lex_glue` is a loading-script dataset,
so the data lives on HuggingFace's `refs/convert/parquet` branch rather than on
`main`.

Both tasks are required and neither is optional: every finding in this
repository is a comparison between what was alleged and what was found, and
half a corpus supports none of it.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def data_dir() -> Path:
    """``$OUTCOME_DATA`` if set, else ``data/`` — the same rule the loader uses."""
    override = os.environ.get("OUTCOME_DATA")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / "data"

BASE = (
    "https://huggingface.co/datasets/coastalcph/lex_glue/"
    "resolve/refs%2Fconvert%2Fparquet"
)
TASKS = ("ecthr_a", "ecthr_b")
SPLITS = ("train", "validation", "test")

# Rows each split must have. LexGLUE's ECtHR is a fixed published split; a
# different count means a different corpus and different numbers.
EXPECT = {"train": 9_000, "validation": 1_000, "test": 1_000}


CHUNK = 4_000_000


def expected_size(url: str) -> int:
    """Content-Length from a HEAD, following redirects."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "outcome"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def fetch(task: str, split: str, data: Path) -> Path:
    """One parquet, in byte ranges, verified against Content-Length.

    Ranged rather than a single GET because a plain GET over this link
    truncates silently on a bad day: curl and urllib both report success and a
    plausible size, and the failure only surfaces much later as a parquet with
    no footer. Asking for explicit ranges and checking the total at the end
    turns that into an error here, where it can be retried.

    Bytes go to ``<name>.part`` and the file is renamed into place only once it
    is complete, so an interrupted run never leaves a truncated parquet under
    the real name. Re-running resumes from the end of the ``.part``.
    """
    out = data / f"{task}_{split}.parquet"
    part = out.with_name(out.name + ".part")
    url = f"{BASE}/{task}/{split}/0000.parquet"

    total = expected_size(url)
    if out.exists() and out.stat().st_size == total:
        print(f"  {out.name} already complete ({total / 1e6:.0f} MB)")
        return out
    if out.exists():
        out.unlink()  # wrong size: a leftover from an older, non-atomic fetch

    written = part.stat().st_size if part.exists() else 0
    if written > total:
        part.unlink()
        written = 0
    resumed = f" (resuming at {written / 1e6:.0f} MB)" if written else ""
    print(f"  {out.name}  {total / 1e6:.0f} MB{resumed} ", end="", flush=True)
    with part.open("ab") as handle:
        while written < total:
            end = min(written + CHUNK, total) - 1
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "outcome", "Range": f"bytes={written}-{end}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    block = response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"\n    failed at byte {written:,}: {exc}. Re-run to resume.")
                raise
            if not block:
                raise OSError(f"{out.name}: empty response at byte {written:,}")
            handle.write(block)
            handle.flush()
            written += len(block)
            print(".", end="", flush=True)

    got = part.stat().st_size
    if got != total:
        part.unlink()
        raise OSError(f"{out.name}: got {got:,} bytes, expected {total:,}. Removed.")
    os.replace(part, out)
    print(" ok")
    return out


def main() -> None:
    data = data_dir()
    data.mkdir(parents=True, exist_ok=True)
    print(f"into {data}")
    print(f"fetching {len(TASKS) * len(SPLITS)} files from LexGLUE")
    for task in TASKS:
        for split in SPLITS:
            fetch(task, split, data)

    import pyarrow.parquet as pq

    print("\nchecking")
    ok = True
    for split in SPLITS:
        rows = {
            task: pq.read_table(data / f"{task}_{split}.parquet").num_rows
            for task in TASKS
        }
        want = EXPECT[split]
        good = set(rows.values()) == {want}
        ok &= good
        print(f"  {split:<11} {rows}  {'ok' if good else f'EXPECTED {want}'}")

    if not ok:
        print(
            "\nThe splits are not the published ones. Every number in the README "
            "was computed on those, so stop here rather than measure something else.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("\nboth tasks present and aligned by row count")


if __name__ == "__main__":
    main()
