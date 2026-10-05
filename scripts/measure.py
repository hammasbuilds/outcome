"""Every number in the README.

    python scripts/measure.py

No model is involved. The claim is about what the benchmark measures, and it is
settled by counting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outcome import baselines as B  # noqa: E402
from outcome import corpus  # noqa: E402


def rule(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def the_corpus() -> None:
    rule("the corpus")
    print(
        f"{'split':<12}{'cases':>7}{'no violation':>14}{'alleged':>10}"
        f"{'upheld':>9}{'unalleged':>11}"
    )
    for split in corpus.SPLITS:
        c = corpus.counts(split)
        print(
            f"{split:<12}{c['cases']:>7,}{c['no_violation']:>14,}"
            f"{c['alleged_claims']:>10,}{c['upheld_claims']:>9,}"
            f"{c['unalleged_findings']:>11,}"
        )
    train = corpus.counts("train")
    share = train["upheld_claims"] / train["alleged_claims"]
    print(f"\n  {share:.1%} of alleged article-violations are upheld by the Court.")
    print("  'unalleged' is the Court finding a violation of an article nobody")
    print("  invoked — legally real, and rare enough not to matter below.")


def the_baselines() -> None:
    rule("what each baseline knows, and what it scores")
    train = corpus.load("train")
    validation = corpus.load("validation")
    test = corpus.load("test")
    gold = [c.violated for c in test]

    top = B.most_common(train)
    cues = B.Cues().fit(train).tune(validation)

    rows = [
        ("always Article 6", "nothing", [frozenset({top})] * len(test)),
        ("articles named in the facts", "the text, literally", [B.mentioned(c) for c in test]),
        ("word cues from the facts", "the text, statistically", [cues.predict(c) for c in test]),
        ("copy what was alleged", "the applicant's complaint", [B.copy_alleged(c) for c in test]),
    ]

    print(f"{'baseline':<30}{'knows':<28}{'micro-F1':>10}{'macro':>8}{'P':>7}{'R':>7}")
    for name, knows, predicted in rows:
        f1, precision, recall = B.micro_f1(gold, predicted)
        print(
            f"{name:<30}{knows:<28}{f1:>10.3f}{B.macro_f1(gold, predicted):>8.3f}"
            f"{precision:>7.3f}{recall:>7.3f}"
        )

    facts_best = max(B.micro_f1(gold, p)[0] for _, k, p in rows if k != "the applicant's complaint")
    copy = B.micro_f1(gold, rows[-1][2])[0]
    print(f"\n  ^ reading the facts, by any means here, tops out at {facts_best:.3f}.")
    print(f"    Knowing what the applicant alleged scores {copy:.3f} without reading")
    print("    them at all. The benchmark's difficulty is mostly the difficulty of")
    print("    guessing the complaint, not of deciding it.")


def the_real_question() -> None:
    rule("the question that is actually hard")
    test = corpus.load("test")
    claims = sum(len(c.alleged) for c in test)
    upheld = sum(len(c.upheld) for c in test)
    print(f"article-claims made in the test split   {claims:>6,}")
    print(f"  upheld                                {upheld:>6,}   {upheld / claims:.1%}")
    print(
        f"  rejected                              {claims - upheld:>6,}   {1 - upheld / claims:.1%}"
    )
    print(
        f"\n  ^ given the complaint, 'the Court agreed' is right {upheld / claims:.1%} of the time."
    )
    print("    So the task worth posing is not 'which articles were violated' but")
    print("    'which of these specific complaints failed' — a harder question with")
    print("    a 75/25 base rate instead of a ten-way multi-label one, and one where")
    print("    a wrong answer is a real error rather than a missing guess.")


def the_articles() -> None:
    rule("per-article, copy-alleged vs reading the facts")
    train = corpus.load("train")
    validation = corpus.load("validation")
    test = corpus.load("test")
    gold = [c.violated for c in test]
    cues = B.Cues().fit(train).tune(validation)

    copy = B.per_article(gold, [B.copy_alleged(c) for c in test])
    read = B.per_article(gold, [cues.predict(c) for c in test])

    print(f"{'article':<10}{'support':>9}{'copy-alleged F1':>18}{'facts-only F1':>16}")
    for name in corpus.ARTICLES:
        support, copy_f1 = copy[name]
        _, read_f1 = read[name]
        print(f"{name:<10}{support:>9,}{copy_f1:>18.3f}{read_f1:>16.3f}")
    print("\n  ^ the gap is everywhere, not concentrated in the rare articles.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not corpus.available():
        print(
            f"No corpus in {corpus.data_dir()}. Run: python scripts/fetch_data.py "
            "(or set OUTCOME_DATA).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    the_corpus()
    the_baselines()
    the_real_question()
    the_articles()
    print()


if __name__ == "__main__":
    main()
