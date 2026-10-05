"""The metrics, and the claim the repository is built on.

`test_copy_alleged_beats_every_facts_only_baseline` is the finding. If it ever
fails, either a facts-only method got much better or the corpus changed, and
either way the README needs rewriting. It needs the real corpus and skips
without it; the metric and baseline mechanics below run on toy cases.
"""

from __future__ import annotations

import pytest

from outcome import baselines as B
from outcome import corpus


def _case(text: str, violated=(), alleged=()) -> corpus.Case:
    return corpus.Case(facts=(text,), violated=frozenset(violated), alleged=frozenset(alleged))


# --- pure: no corpus needed ---------------------------------------------------


def test_micro_f1_on_a_worked_example():
    gold = [frozenset({0, 1}), frozenset({2})]
    pred = [frozenset({0}), frozenset({2, 3})]
    # tp = 2 (0 and 2), fp = 1 (3), fn = 1 (1)
    f1, precision, recall = B.micro_f1(gold, pred)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def test_micro_f1_empty_prediction_scores_zero_not_nan():
    assert B.micro_f1([frozenset({1})], [frozenset()]) == (0.0, 0.0, 0.0)


def test_micro_f1_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        B.micro_f1([frozenset({1})], [frozenset({1}), frozenset()])


def test_per_article_reports_support_and_f1():
    six, two = corpus.ARTICLES.index("6"), corpus.ARTICLES.index("2")
    gold = [frozenset({six}), frozenset({six, two}), frozenset()]
    pred = [frozenset({six}), frozenset({six}), frozenset({two})]
    out = B.per_article(gold, pred)
    assert out["6"] == (2, 1.0)
    assert out["2"] == (1, 0.0)


def test_mentioned_finds_protocol_one():
    """'Article 1 of Protocol No. 1' must not be read as Article 1."""
    case = _case("The applicant complained under Article 1 of Protocol No. 1.")
    assert B.mentioned(case) == frozenset({corpus.ARTICLES.index("P1-1")})


def test_mentioned_reads_lists():
    case = _case("He relied on Articles 6 and 14 of the Convention.")
    assert B.mentioned(case) == {corpus.ARTICLES.index("6"), corpus.ARTICLES.index("14")}


def test_mentioned_ignores_articles_outside_the_label_set():
    case = _case("The Court referred to Article 41 and Article 35.")
    assert B.mentioned(case) == frozenset()


def test_cues_learn_and_tune_on_a_toy_corpus():
    """Article 3 cases talk about ill-treatment; the fitted cue model must find it."""
    art3 = corpus.ARTICLES.index("3")
    train = [_case("police beating detention ill-treatment injuries", {art3})] * 20 + [
        _case("property expropriation compensation land")
    ] * 20
    cues = B.Cues(top=5).fit(train).tune(train)
    assert {t for t, _ in cues.terms[art3]} >= {"beating", "ill-treatment", "injuries"}
    assert art3 in cues.predict(_case("beating in police detention, ill-treatment, injuries"))
    assert art3 not in cues.predict(_case("expropriation of land without compensation"))


def test_tune_matches_a_brute_force_threshold_search():
    """The single-pass tune must pick the cut an exhaustive search picks."""
    art6 = corpus.ARTICLES.index("6")
    texts = [
        "delay proceedings",
        "delay hearing",
        "hearing",
        "land",
        "delay",
        "trial delay",
        "proceedings land",
        "hearing trial",
    ]
    gold = [{art6}, {art6}, set(), set(), {art6}, set(), {art6}, set()]
    cases = [_case(t, g) for t, g in zip(texts, gold, strict=True)]
    cues = B.Cues(top=10).fit(cases).tune(cases)
    for label in range(len(corpus.ARTICLES)):
        scored = sorted((cues.score(c, label), label in c.violated) for c in cases)
        best, best_at = -1.0, 0.0
        for cut, _ in scored:
            tp = sum(1 for s, g in scored if s >= cut and g)
            fp = sum(1 for s, g in scored if s >= cut and not g)
            fn = sum(1 for s, g in scored if s < cut and g)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best:
                best, best_at = f1, cut
        assert cues.cut[label] == best_at


# --- on the real corpus (skip without it) -------------------------------------


def test_micro_f1_is_perfect_on_a_perfect_prediction(test_split):
    gold = [c.violated for c in test_split]
    assert B.micro_f1(gold, gold) == (1.0, 1.0, 1.0)


def test_macro_f1_punishes_ignoring_rare_articles(train, test_split):
    """Predicting only the commonest article scores far worse on macro."""
    gold = [c.violated for c in test_split]
    only_top = [frozenset({B.most_common(train)})] * len(test_split)
    assert B.macro_f1(gold, only_top) < B.micro_f1(gold, only_top)[0]


def test_cues_are_deterministic(train):
    """Same guard as contract-reader: equal-weight terms must not be ordered
    by set iteration, which Python randomises per process."""
    a = B.Cues(top=20).fit(train[:500])
    b = B.Cues(top=20).fit(train[:500])
    assert a.terms == b.terms


def test_copy_alleged_beats_every_facts_only_baseline(train, validation, test_split):
    """The finding, asserted.

    Reading the facts by any means here tops out around 0.31 micro-F1. Copying
    the applicant's own complaint scores about 0.86 without reading them.
    """
    gold = [c.violated for c in test_split]
    top = B.most_common(train)
    cues = B.Cues().fit(train).tune(validation)
    facts_only = [
        B.micro_f1(gold, [frozenset({top})] * len(test_split))[0],
        B.micro_f1(gold, [B.mentioned(c) for c in test_split])[0],
        B.micro_f1(gold, [cues.predict(c) for c in test_split])[0],
    ]
    copied = B.micro_f1(gold, [B.copy_alleged(c) for c in test_split])[0]
    assert copied > 0.80
    assert max(facts_only) < 0.40
    assert copied > max(facts_only) + 0.40


def test_copy_alleged_has_near_perfect_recall_and_imperfect_precision(test_split):
    """It cannot miss, because a violation is almost always alleged. What it
    gets wrong is everything the Court rejected — which is the real task."""
    gold = [c.violated for c in test_split]
    _, precision, recall = B.micro_f1(gold, [B.copy_alleged(c) for c in test_split])
    assert recall > 0.98
    assert precision < 0.85
