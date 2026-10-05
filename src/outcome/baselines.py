"""Predictors that do not read the law, so the benchmark can be sized.

Four of them, in increasing order of how much they actually know:

* **`most_common`** — always predict Article 6. A floor.
* **`mentioned`** — predict any article the facts name in so many words.
  Reads the text, understands nothing.
* **`cues`** — per-article terms learned from the training split.
* **`copy_alleged`** — predict exactly what the applicant alleged. Uses the
  Task B labels, which the standard Task A formulation does not give a model.

The last one is the point of the module. It is not a legal system and it is not
cheating either: in any real deployment you know what the applicant complained
about, because that is what starts the case. What it measures is how much of
the published benchmark is the gap between *alleging* and *winning*, and how
much is reading facts at all.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .corpus import ARTICLES

TOKEN = re.compile(r"[a-z][a-z'-]{2,}")
STOP = frozenset(
    """the and for any all not that this with had has have was were been which
    who whom his her its their they them she from are also would could been
    when there then than upon into under over such other applicant court
    government case article convention""".split()
)

# "Article 6", "Articles 6 and 13", "Article 1 of Protocol No. 1".
_PROTOCOL = re.compile(r"\barticle\s+1\s+of\s+protocol\s+(?:no\.?\s*)?1\b", re.I)
_ARTICLE = re.compile(r"\barticles?\s+((?:\d{1,2})(?:\s*(?:,|and)\s*\d{1,2})*)", re.I)
_NUMBER = re.compile(r"\d{1,2}")

_BY_NUMBER = {name: index for index, name in enumerate(ARTICLES)}


def words(text: str) -> set[str]:
    return {w for w in TOKEN.findall(text.lower()) if w not in STOP}


# ---------------------------------------------------------------------------


def most_common(train) -> int:
    counted = Counter(label for case in train for label in case.violated)
    return counted.most_common(1)[0][0]


def mentioned(case) -> frozenset[int]:
    """Every Convention article the facts name explicitly.

    Only 27% of cases name one at all, and naming an article is not the same
    as the Court finding it violated — the facts section recites the
    applicant's complaints, and most of the recitation is what they lost on.
    """
    found: set[int] = set()
    text = case.text
    if _PROTOCOL.search(text):
        found.add(_BY_NUMBER["P1-1"])
    for group in _ARTICLE.findall(text):
        for number in _NUMBER.findall(group):
            if number in _BY_NUMBER:
                found.add(_BY_NUMBER[number])
    return frozenset(found)


def copy_alleged(case) -> frozenset[int]:
    """Predict exactly what the applicant said was violated."""
    return case.alleged


class Cues:
    """Per-article terms, by log-odds, fitted on train only."""

    def __init__(self, top: int = 60) -> None:
        self.top = top
        self.terms: dict[int, tuple[tuple[str, float], ...]] = {}
        self.cut: dict[int, float] = {}

    def fit(self, train) -> Cues:
        bags = [words(c.text) for c in train]
        everywhere = Counter()
        for bag in bags:
            everywhere.update(bag)
        for label in range(len(ARTICLES)):
            pos = Counter()
            n_p = 0
            for case, bag in zip(train, bags, strict=True):
                if label in case.violated:
                    pos.update(bag)
                    n_p += 1
            neg = everywhere - pos
            n_n = len(bags) - n_p
            scored = [
                (
                    term,
                    math.log(
                        ((pos[term] + 1) / (max(n_p, 1) + 2))
                        / ((neg[term] + 1) / (max(n_n, 1) + 2))
                    ),
                )
                for term in set(pos) | set(neg)
            ]
            # Tie-break on the term. Without it, equal-weight terms are ordered
            # by set iteration, which Python randomises per process, and the
            # whole model moves between runs.
            scored.sort(key=lambda t: (-t[1], t[0]))
            self.terms[label] = tuple(scored[: self.top])
        return self

    def score(self, case, label: int) -> float:
        return self._score(words(case.text), label)

    def _score(self, bag: set[str], label: int) -> float:
        return sum(w for term, w in self.terms[label] if term in bag)

    def tune(self, validation) -> Cues:
        """One threshold per article, maximising that article's F1.

        F1 rather than accuracy: several articles appear in under 2% of cases,
        where a threshold chosen on accuracy predicts 'no' forever and scores
        98%.

        Candidate cuts are the observed scores. Walking them from the top down
        keeps running tp/fp counts, so each article costs one sort rather than
        one pass over the split per candidate.
        """
        bags = [words(c.text) for c in validation]
        for label in range(len(ARTICLES)):
            scored = sorted(
                (
                    (self._score(bag, label), label in c.violated)
                    for c, bag in zip(validation, bags, strict=True)
                ),
                key=lambda t: -t[0],
            )
            positives = sum(1 for _, gold in scored if gold)
            best, best_at = -1.0, 0.0
            tp = fp = 0
            i = 0
            while i < len(scored):
                cut = scored[i][0]
                # Every case scoring exactly `cut` is predicted together.
                while i < len(scored) and scored[i][0] == cut:
                    if scored[i][1]:
                        tp += 1
                    else:
                        fp += 1
                    i += 1
                fn = positives - tp
                f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
                # >= keeps the lowest cut among equal F1s, matching an
                # ascending scan that takes the first maximum.
                if f1 >= best:
                    best, best_at = f1, cut
            self.cut[label] = best_at
        return self

    def predict(self, case) -> frozenset[int]:
        bag = words(case.text)
        return frozenset(
            label
            for label in range(len(ARTICLES))
            if self._score(bag, label) >= self.cut.get(label, 0.0)
        )


# ---------------------------------------------------------------------------


def micro_f1(gold, predicted) -> tuple[float, float, float]:
    """Micro-averaged over article-claims, which is what LexGLUE reports."""
    tp = sum(len(g & p) for g, p in zip(gold, predicted, strict=True))
    fp = sum(len(p - g) for g, p in zip(gold, predicted, strict=True))
    fn = sum(len(g - p) for g, p in zip(gold, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def macro_f1(gold, predicted) -> float:
    """Averaged over articles, so the rare ones count as much as Article 6.

    Worth reporting next to micro: Article 6 is half of all findings, and a
    model that only knows Article 6 looks respectable on micro and collapses
    here.
    """
    scores = []
    for label in range(len(ARTICLES)):
        tp = sum(1 for g, p in zip(gold, predicted, strict=True) if label in g and label in p)
        fp = sum(1 for g, p in zip(gold, predicted, strict=True) if label not in g and label in p)
        fn = sum(1 for g, p in zip(gold, predicted, strict=True) if label in g and label not in p)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores)


def per_article(gold, predicted) -> dict[str, tuple[int, float]]:
    out = {}
    for label, name in enumerate(ARTICLES):
        tp = sum(1 for g, p in zip(gold, predicted, strict=True) if label in g and label in p)
        fp = sum(1 for g, p in zip(gold, predicted, strict=True) if label not in g and label in p)
        fn = sum(1 for g, p in zip(gold, predicted, strict=True) if label in g and label not in p)
        support = tp + fn
        out[name] = (support, 2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return out
