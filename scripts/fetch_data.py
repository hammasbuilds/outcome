"""Pull both ECtHR tasks from the LexGLUE parquet conversion.

    python scripts/fetch_data.py

Six files, 102 MB, ungated. `coastalcph/lex_glue` is a loading-script dataset,
so the data lives on HuggingFace's `refs/convert/parquet` branch rather than on
`main`.

Both tasks are required and neither is optional: every finding in this
repository is a comparison between what was alleged and what was found, and
half a corpus supports none of it.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
BASE = (
    "https://huggingface.co/datasets/coastalcph/lex_glue/"
    "resolve/refs%2Fconvert%2Fparquet"
)
TASKS = ("ecthr_a", "ecthr_b")
SPLITS = ("train", "validation", "test")

# Rows each split must have. LexGLUE's ECtHR is a fixed published split; a
# different count means a different corpus and different numbers.
EXPECT = {"train": 9_000, "validation": 1_000, "test": 1_000}


def fetch(task: str, split: str) -> Path:
    out = DATA / f"{task}_{split}.parquet"
    if out.exists():
        print(f"  {out.name} already here ({out.stat().st_size / 1e6:.0f} MB)")
        return out

    url = f"{BASE}/{task}/{split}/0000.parquet"
    print(f"  {out.name} ...", end="", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "outcome"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            out.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f" failed: {exc}")
        raise
    print(f" {out.stat().st_size / 1e6:.0f} MB")
    return out


def main() -> None:
    DATA.mkdir(exist_ok=True)
    print(f"fetching {len(TASKS) * len(SPLITS)} files from LexGLUE")
    for task in TASKS:
        for split in SPLITS:
            fetch(task, split)

    import pyarrow.parquet as pq

    print("\nchecking")
    ok = True
    for split in SPLITS:
        rows = {
            task: pq.read_table(DATA / f"{task}_{split}.parquet").num_rows
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
