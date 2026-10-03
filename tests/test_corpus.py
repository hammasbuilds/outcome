"""The two tasks are one corpus, and everything here depends on that.

Every finding in this repository is a comparison between what an applicant
alleged and what the Court found. That comparison is only meaningful if the two
parquet files line up row for row, so the alignment is asserted rather than
trusted — a silent off-by-one would produce plausible numbers about nothing.

The loader's guards are tested on tiny parquet written to a temp dir; the
checks on the real corpus skip when it is not on disk.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from outcome import corpus

A_ROWS = [(["para one", "para two"], [3]), (["other facts"], [])]
B_ROWS = [(["para one", "para two"], [3, 4]), (["other facts"], [1])]


def _write(where, task, split, rows):
    table = pa.table({"text": [r[0] for r in rows], "labels": [r[1] for r in rows]})
    pq.write_table(table, where / f"ecthr_{task}_{split}.parquet")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A three-split corpus in a temp dir, selected through OUTCOME_DATA."""

    def build(b_rows=B_ROWS):
        for split in corpus.SPLITS:
            _write(tmp_path, "a", split, A_ROWS)
            _write(tmp_path, "b", split, b_rows)
        return tmp_path

    monkeypatch.setenv("OUTCOME_DATA", str(tmp_path))
    return build


# --- pure: no corpus needed ---------------------------------------------------


def test_loader_reads_from_outcome_data(fake):
    fake()
    assert corpus.available()
    cases = corpus.load("test")
    assert len(cases) == 2
    assert cases[0].violated == {3} and cases[0].alleged == {3, 4}
    assert cases[0].upheld == {3} and cases[0].rejected == {4}
    assert cases[1].no_violation and cases[1].rejected == {1}
    assert corpus.counts("test") == {
        "cases": 2, "no_violation": 1, "alleged_claims": 3,
        "upheld_claims": 1, "unalleged_findings": 0,
    }


def test_loader_rejects_misaligned_tasks(fake):
    """The guard that makes the whole comparison safe: if the two tasks
    disagree about a case's facts, `load` refuses."""
    fake(b_rows=list(reversed(B_ROWS)))
    with pytest.raises(ValueError, match="disagree about the facts"):
        corpus.load("validation")


def test_loader_rejects_row_count_drift(fake):
    fake(b_rows=B_ROWS[:1])
    with pytest.raises(ValueError, match="only comparable row by row"):
        corpus.load("train")


def test_missing_corpus_names_the_fetch_script(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOME_DATA", str(tmp_path / "nowhere"))
    assert not corpus.available()
    with pytest.raises(corpus.CorpusMissingError, match="fetch_data.py"):
        corpus.load("train")


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        corpus.load("dev")


def test_ten_articles():
    assert len(corpus.ARTICLES) == 10
    assert corpus.article(3) == "6"
    assert corpus.article(9) == "P1-1"


# --- on the real corpus (skip without it) -------------------------------------


@pytest.mark.parametrize("split,expected", [("train", 9000), ("validation", 1000), ("test", 1000)])
def test_split_sizes(splits, split, expected):
    assert len(splits[split]) == expected


@pytest.mark.parametrize("split", corpus.SPLITS)
def test_every_case_has_facts(splits, split):
    assert all(c.facts for c in splits[split])


@pytest.mark.parametrize("split", corpus.SPLITS)
def test_labels_are_in_range(splits, split):
    for case in splits[split]:
        assert all(0 <= label < len(corpus.ARTICLES) for label in case.violated)
        assert all(0 <= label < len(corpus.ARTICLES) for label in case.alleged)


@pytest.mark.parametrize("split", corpus.SPLITS)
def test_violations_are_almost_always_alleged(splits, split):
    """A violation of an article nobody invoked is legally possible and rare.

    If this ever rose sharply it would mean the two files had drifted out of
    alignment, which is the failure this corpus is most exposed to.
    """
    cases = splits[split]
    unalleged = sum(1 for c in cases if c.unalleged)
    assert unalleged / len(cases) < 0.05


@pytest.mark.parametrize("split", corpus.SPLITS)
def test_upheld_is_the_intersection(splits, split):
    for case in splits[split]:
        assert case.upheld == case.violated & case.alleged
        assert case.rejected == case.alleged - case.violated
        assert case.upheld.isdisjoint(case.rejected)


def test_about_a_tenth_of_cases_find_no_violation(train):
    counted = corpus.counts("train")
    assert 0.05 < counted["no_violation"] / counted["cases"] < 0.20


def test_most_allegations_succeed(train):
    """The number the headline rests on."""
    counted = corpus.counts("train")
    share = counted["upheld_claims"] / counted["alleged_claims"]
    assert 0.70 < share < 0.85
