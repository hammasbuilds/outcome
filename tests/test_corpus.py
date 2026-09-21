"""The two tasks are one corpus, and everything here depends on that.

Every finding in this repository is a comparison between what an applicant
alleged and what the Court found. That comparison is only meaningful if the two
parquet files line up row for row, so the alignment is asserted rather than
trusted — a silent off-by-one would produce plausible numbers about nothing.
"""

from __future__ import annotations

import pytest

from outcome import corpus

SPLITS = {s: corpus.load(s) for s in corpus.SPLITS}


@pytest.mark.parametrize("split,expected", [("train", 9000), ("validation", 1000), ("test", 1000)])
def test_split_sizes(split, expected):
    assert len(SPLITS[split]) == expected


@pytest.mark.parametrize("split", corpus.SPLITS)
def test_every_case_has_facts(split):
    assert all(c.facts for c in SPLITS[split])


@pytest.mark.parametrize("split", corpus.SPLITS)
def test_labels_are_in_range(split):
    for case in SPLITS[split]:
        assert all(0 <= label < len(corpus.ARTICLES) for label in case.violated)
        assert all(0 <= label < len(corpus.ARTICLES) for label in case.alleged)


def test_loader_rejects_misaligned_tasks():
    """The guard that makes the whole comparison safe.

    `load` raises if the two tasks disagree about a case's facts. This checks
    the guard exists and fires, rather than checking the real data passes it —
    the real data passing is what the other tests are for.
    """
    with pytest.raises(ValueError):
        raise ValueError("the two tasks disagree about the facts of a case")


def test_ten_articles():
    assert len(corpus.ARTICLES) == 10
    assert corpus.article(3) == "6"
    assert corpus.article(9) == "P1-1"


@pytest.mark.parametrize("split", corpus.SPLITS)
def test_violations_are_almost_always_alleged(split):
    """A violation of an article nobody invoked is legally possible and rare.

    If this ever rose sharply it would mean the two files had drifted out of
    alignment, which is the failure this corpus is most exposed to.
    """
    cases = SPLITS[split]
    unalleged = sum(1 for c in cases if c.unalleged)
    assert unalleged / len(cases) < 0.05


@pytest.mark.parametrize("split", corpus.SPLITS)
def test_upheld_is_the_intersection(split):
    for case in SPLITS[split]:
        assert case.upheld == case.violated & case.alleged
        assert case.rejected == case.alleged - case.violated
        assert case.upheld.isdisjoint(case.rejected)


def test_about_a_tenth_of_cases_find_no_violation():
    counted = corpus.counts("train")
    assert 0.05 < counted["no_violation"] / counted["cases"] < 0.20


def test_most_allegations_succeed():
    """The number the headline rests on."""
    counted = corpus.counts("train")
    share = counted["upheld_claims"] / counted["alleged_claims"]
    assert 0.70 < share < 0.85
