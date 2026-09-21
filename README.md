# outcome

> The ECtHR outcome benchmark scores **0.859 micro-F1** if you ignore the facts entirely and just repeat what the applicant complained about. Reading the facts, by every method here, tops out at **0.309**.

**Status:** complete as a measurement. It needs no language model, and the finding is a
reason to be careful about what one would be scored on. See
[What to measure instead](#what-to-measure-instead).

## The corpus

[LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue)'s ECtHR tasks — 11,000
European Court of Human Rights judgments, facts section only, labelled with Convention
articles.

The corpus ships as two datasets that are really one, built from the same judgments in the
same order with byte-identical facts:

| | |
|---|---|
| **Task A** | the articles the Court **found violated** |
| **Task B** | the articles the applicant **alleged** were violated |

| Split | Cases | No violation | Alleged claims | Upheld | Unalleged findings |
|---|---:|---:|---:|---:|---:|
| train | 9,000 | 914 | 13,165 | 10,371 | 269 |
| validation | 1,000 | 175 | 1,391 | 1,055 | 5 |
| test | 1,000 | 153 | 1,435 | 1,085 | 5 |

**78.8% of alleged article-violations are upheld.** The Court almost never finds a
violation of an article nobody invoked — 269 times in 9,000 cases — which is what makes the
next table possible.

```
python scripts/fetch_data.py   # 102 MB of parquet, not in git
python scripts/measure.py      # every table below
python -m pytest               # 28 tests
```

## What each baseline knows, and what it scores

Test split, micro-F1 over article-claims, which is what LexGLUE reports.

| Baseline | Knows | micro-F1 | macro | P | R |
|---|---|---:|---:|---:|---:|
| always Article 6 | nothing | 0.286 | 0.046 | 0.299 | 0.274 |
| articles named in the facts | the text, literally | 0.234 | 0.220 | 0.330 | 0.181 |
| word cues from the facts | the text, statistically | 0.309 | 0.267 | 0.195 | 0.743 |
| **copy what was alleged** | **the applicant's complaint** | **0.859** | **0.822** | 0.756 | **0.995** |

The gap is not a modelling result. It is the shape of the task. `copy_alleged` has **0.995
recall** — it cannot miss, because a violation is nearly always among the things alleged.
Everything it gets wrong is a complaint the Court rejected.

Per article, the gap is everywhere rather than concentrated in the rare ones:

| Article | Support | copy-alleged F1 | facts-only F1 |
|---|---:|---:|---:|
| 2 — life | 56 | 0.848 | 0.450 |
| 3 — torture | 189 | 0.894 | 0.486 |
| 5 — liberty | 166 | 0.912 | 0.285 |
| 6 — fair trial | 299 | 0.863 | 0.460 |
| 8 — private life | 123 | 0.791 | 0.219 |
| 9 — religion | 5 | 0.625 | 0.200 |
| 10 — expression | 77 | 0.842 | 0.149 |
| 11 — assembly | 37 | 0.925 | 0.200 |
| 14 — discrimination | 16 | 0.667 | 0.000 |
| P1-1 — property | 122 | 0.852 | 0.217 |

## Is that cheating?

It uses Task B's labels, which the standard Task A formulation does not hand the model. But
it is not information nobody has: **the complaint is what starts the case.** Any real system
deciding an ECtHR application knows which articles were invoked, because the application
says so. The benchmark's difficulty comes substantially from withholding something that is
never actually missing.

So the honest reading is not "the benchmark is broken". It is that a score on Task A is
mostly a measure of *guessing the complaint from the facts*, and only slightly a measure of
*deciding it*.

## What to measure instead

Given the complaint, "the Court agreed" is right **75.6%** of the time on the test split.

| Test split | n | |
|---|---:|---:|
| Article-claims made | 1,435 | |
| Upheld | 1,085 | 75.6% |
| **Rejected** | **350** | **24.4%** |

The task worth posing is **"which of these specific complaints failed"** — a binary
judgement with a 75/25 base rate, conditioned on the article actually in dispute. It is
harder than the published task, it is the question a court actually answers, and a wrong
answer on it is a real error rather than a missing guess in a ten-way multi-label set.

That is also the task where a 14B model has something to contribute and a bag of words does
not, which is the next thing to build here.

## Notes on the data

**The label mapping is not distributed with the parquet.** It was recovered by taking every
single-label case and asking which article its facts mention most often; eight of the ten
come out unambiguous, and all ten match the published LexGLUE label set — Articles 2, 3, 5,
6, 8, 9, 10, 11, 14 and P1-1.

**The two tasks are joined row by row and the loader refuses to proceed if they drift.**
Every number above is a comparison between the two files, so a silent off-by-one would
produce plausible numbers about nothing. `corpus.load` raises if any case's facts differ
between the two, and `test_violations_are_almost_always_alleged` would fail loudly if the
files came apart.

**"Article 1 of Protocol No. 1" is not Article 1.** The literal-mention baseline matches the
protocol form first; without that it would read every property case as an Article 1 case,
which is not in the label set at all.

## Layout

```
scripts/fetch_data.py        both tasks, all three splits, from the LexGLUE parquet
src/outcome/corpus.py        the two tasks joined; alleged / violated / upheld / rejected
src/outcome/baselines.py     four predictors and the micro/macro metrics
scripts/measure.py           every table above
tests/                       28 tests, incl. the alignment guard
```
