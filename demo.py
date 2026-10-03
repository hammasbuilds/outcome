"""The finding in one screen.

    python demo.py

Without the corpus it walks one illustrative case (made up, labelled as such)
through every baseline, so you can see what "copy what was alleged" means before
downloading 102 MB. With the corpus (`python scripts/fetch_data.py`, or
OUTCOME_DATA pointing at it) it prints the real test-split headline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from outcome import baselines as B  # noqa: E402
from outcome import corpus  # noqa: E402


def names(labels) -> str:
    return ", ".join(sorted((corpus.article(i) for i in labels), key=corpus.ARTICLES.index)) or "-"


def illustrative() -> None:
    art = corpus.ARTICLES.index
    case = corpus.Case(
        facts=(
            "1. The applicant was arrested and held in police custody for four days.",
            "2. He alleged that officers beat him; a medical report recorded bruising.",
            "3. His appeal against the detention order was dismissed without a hearing.",
        ),
        violated=frozenset({art("3"), art("5")}),
        alleged=frozenset({art("3"), art("5"), art("6")}),
    )
    print("No corpus on disk - an ILLUSTRATIVE case, not from the dataset:\n")
    print("  " + "\n  ".join(case.facts))
    print(f"\n  alleged by the applicant : Articles {names(case.alleged)}")
    print(f"  found violated by Court  : Articles {names(case.violated)}")
    print(f"  rejected complaints      : Articles {names(case.rejected)}\n")
    for label, predicted in [
        ("articles named in the facts", B.mentioned(case)),
        ("copy what was alleged", B.copy_alleged(case)),
    ]:
        f1, p, r = B.micro_f1([case.violated], [predicted])
        print(f"  {label:<28} -> {names(predicted):<10} F1 {f1:.2f}  P {p:.2f}  R {r:.2f}")
    print(
        "\n  Copying the complaint cannot miss a violation; its only errors are the\n"
        "  complaints the Court rejected. Fetch the corpus to see that at scale:\n"
        "    python scripts/fetch_data.py && python demo.py"
    )


def real() -> None:
    test = corpus.load("test")
    gold = [c.violated for c in test]
    copy = B.micro_f1(gold, [B.copy_alleged(c) for c in test])
    named = B.micro_f1(gold, [B.mentioned(c) for c in test])
    claims = sum(len(c.alleged) for c in test)
    upheld = sum(len(c.upheld) for c in test)
    print(f"ECtHR test split, {len(test):,} judgments (LexGLUE), micro-F1 on Task A:\n")
    print(f"  copy what was alleged        {copy[0]:.3f}  (P {copy[1]:.3f}, R {copy[2]:.3f})")
    print(f"  articles named in the facts  {named[0]:.3f}")
    print(f"\n  {upheld:,} of {claims:,} alleged article-claims upheld ({upheld / claims:.1%}).")
    print("  The trained facts-only baseline and per-article table: python scripts/measure.py")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    real() if corpus.available() else illustrative()


if __name__ == "__main__":
    main()
