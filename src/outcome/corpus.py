"""ECtHR judgments from LexGLUE, both tasks, aligned.

The corpus ships as two datasets that are really one. Task A labels the
articles the Court **found violated**; Task B labels the articles the applicant
**alleged** were violated. They are built from the same 11,000 judgments, in the
same order, with byte-identical facts — which this module asserts rather than
assumes, because everything interesting here is a comparison between the two.

    from outcome import corpus

    train = corpus.load("train")        # 9,000 cases, both label sets
    train[0].violated                   # articles the Court upheld
    train[0].alleged                    # articles the applicant claimed

Each case is the facts section of the judgment, already split into numbered
paragraphs by the Court's own formatting.

The label ids are indices into `ARTICLES`. That mapping is not distributed with
the parquet; it was recovered from the data by taking every single-label case
and asking which article its facts mention most often, which gives an
unambiguous answer for eight of the ten and matches the published LexGLUE label
set for all of them. `scripts/verify_mapping.py` re-derives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"

VIOLATED, ALLEGED = "a", "b"
SPLITS = ("train", "validation", "test")

# Index -> the Convention article the label stands for.
ARTICLES = (
    "2",      # right to life
    "3",      # prohibition of torture
    "5",      # liberty and security
    "6",      # fair trial
    "8",      # private and family life
    "9",      # thought, conscience and religion
    "10",     # freedom of expression
    "11",     # assembly and association
    "14",     # prohibition of discrimination
    "P1-1",   # protection of property
)


class CorpusMissingError(FileNotFoundError):
    """The parquet files are not on disk."""


@dataclass(frozen=True)
class Case:
    """One judgment, with both label sets attached."""

    facts: tuple[str, ...]
    violated: frozenset[int]
    alleged: frozenset[int]

    @property
    def text(self) -> str:
        return "\n".join(self.facts)

    @property
    def no_violation(self) -> bool:
        """The Court found nothing. About one case in ten."""
        return not self.violated

    @property
    def upheld(self) -> frozenset[int]:
        """Alleged and found. The intersection is the question the case asks."""
        return self.violated & self.alleged

    @property
    def rejected(self) -> frozenset[int]:
        """Alleged and not found."""
        return self.alleged - self.violated

    @property
    def unalleged(self) -> frozenset[int]:
        """Found without having been alleged.

        Rare and legally real: the Court may re-characterise a complaint under
        an article the applicant did not invoke. It is also the only reason
        `violated` is not always a subset of `alleged`, so it is the thing that
        stops 'copy the allegations' from being a perfect ceiling.
        """
        return self.violated - self.alleged


def _path(task: str, split: str) -> Path:
    return DATA / f"ecthr_{task}_{split}.parquet"


@lru_cache(maxsize=8)
def load(split: str = "train") -> tuple[Case, ...]:
    """One split, with both tasks joined row by row."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; have {SPLITS}")

    import pyarrow.parquet as pq

    for task in (VIOLATED, ALLEGED):
        if not _path(task, split).exists():
            raise CorpusMissingError(
                f"{_path(task, split)} is missing. Run scripts/fetch_data.py, "
                "which pulls both tasks from the LexGLUE parquet on HuggingFace."
            )

    violated = pq.read_table(_path(VIOLATED, split)).to_pylist()
    alleged = pq.read_table(_path(ALLEGED, split)).to_pylist()

    if len(violated) != len(alleged):
        raise ValueError(
            f"{split}: task A has {len(violated)} rows and task B has {len(alleged)}. "
            "The two tasks are only comparable row by row."
        )

    out = []
    for a, b in zip(violated, alleged, strict=True):
        if a["text"] != b["text"]:
            raise ValueError(
                f"{split}: the two tasks disagree about the facts of a case. "
                "Every comparison in this repository assumes they are aligned."
            )
        out.append(
            Case(
                facts=tuple(a["text"]),
                violated=frozenset(a["labels"]),
                alleged=frozenset(b["labels"]),
            )
        )
    return tuple(out)


def article(label: int) -> str:
    return ARTICLES[label]


def counts(split: str = "train") -> dict[str, int]:
    cases = load(split)
    return {
        "cases": len(cases),
        "no_violation": sum(1 for c in cases if c.no_violation),
        "alleged_claims": sum(len(c.alleged) for c in cases),
        "upheld_claims": sum(len(c.upheld) for c in cases),
        "unalleged_findings": sum(len(c.unalleged) for c in cases),
    }
