"""The metrics, and the claim the repository is built on.

`test_copy_alleged_beats_every_facts_only_baseline` is the finding. If it ever
fails, either a facts-only method got much better or the corpus changed, and
either way the README needs rewriting.
"""

from __future__ import annotations

import pytest

from outcome import baselines as B
from outcome import corpus

TRAIN = corpus.load("train")
VAL = corpus.load("validation")
TEST = corpus.load("test")
GOLD = [c.violated for c in TEST]


def test_micro_f1_on_a_worked_example():
    gold = [frozenset({0, 1}), frozenset({2})]
    pred = [frozenset({0}), frozenset({2, 3})]
    # tp = 2 (0 and 2), fp = 1 (3), fn = 1 (1)
    f1, precision, recall = B.micro_f1(gold, pred)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def test_micro_f1_is_perfect_on_a_perfect_prediction():
    f1, precision, recall = B.micro_f1(GOLD, GOLD)
    assert (f1, precision, recall) == (1.0, 1.0, 1.0)


def test_macro_f1_punishes_ignoring_rare_articles():
    """Predicting only the commonest article scores far worse on macro."""
    top = B.most_common(TRAIN)
    only_top = [frozenset({top})] * len(TEST)
    assert B.macro_f1(GOLD, only_top) < B.micro_f1(GOLD, only_top)[0]


def test_mentioned_finds_protocol_one():
    """'Article 1 of Protocol No. 1' must not be read as Article 1."""
    case = corpus.Case(
        facts=("The applicant complained under Article 1 of Protocol No. 1.",),
        violated=frozenset(),
        alleged=frozenset(),
    )
    assert B.mentioned(case) == frozenset({corpus.ARTICLES.index("P1-1")})


def test_mentioned_reads_lists():
    case = corpus.Case(
        facts=("He relied on Articles 6 and 14 of the Convention.",),
        violated=frozenset(),
        alleged=frozenset(),
    )
    assert B.mentioned(case) == {corpus.ARTICLES.index("6"), corpus.ARTICLES.index("14")}


def test_mentioned_ignores_articles_outside_the_label_set():
    case = corpus.Case(
        facts=("The Court referred to Article 41 and Article 35.",),
        violated=frozenset(),
        alleged=frozenset(),
    )
    assert B.mentioned(case) == frozenset()


def test_cues_are_deterministic():
    """Same guard as contract-reader: equal-weight terms must not be ordered
    by set iteration, which Python randomises per process."""
    a = B.Cues(top=20).fit(TRAIN[:500])
    b = B.Cues(top=20).fit(TRAIN[:500])
    assert a.terms == b.terms


def test_copy_alleged_beats_every_facts_only_baseline():
    """The finding, asserted.

    Reading the facts by any means here tops out around 0.31 micro-F1. Copying
    the applicant's own complaint scores about 0.86 without reading them.
    """
    top = B.most_common(TRAIN)
    cues = B.Cues().fit(TRAIN).tune(VAL)
    facts_only = [
        B.micro_f1(GOLD, [frozenset({top})] * len(TEST))[0],
        B.micro_f1(GOLD, [B.mentioned(c) for c in TEST])[0],
        B.micro_f1(GOLD, [cues.predict(c) for c in TEST])[0],
    ]
    copied = B.micro_f1(GOLD, [B.copy_alleged(c) for c in TEST])[0]
    assert copied > 0.80
    assert max(facts_only) < 0.40
    assert copied > max(facts_only) + 0.40


def test_copy_alleged_has_near_perfect_recall_and_imperfect_precision():
    """It cannot miss, because a violation is almost always alleged. What it
    gets wrong is everything the Court rejected — which is the real task."""
    _, precision, recall = B.micro_f1(GOLD, [B.copy_alleged(c) for c in TEST])
    assert recall > 0.98
    assert precision < 0.85
