# FIJI Event Engagement Analytics

Predicting per-event-type attendance likelihood from chapter survey data, to help plan a Fall event
calendar members will actually attend.

## Overview

A ~100-member fraternity chapter collected 43 anonymous survey responses rating 8 event types on a
1–5 scale. This project cleans that data, validates real predictive patterns, responsibly expands the
small sample with a verified synthetic data generator, trains an ordinal predictive model, and produces
an actionable per-event-type engagement table for the chapter's exec board — along with an honest
accounting of the project's limitations.

**Result:** the model beats a naive baseline (always guessing the most common rating) by ~46% on mean
absolute error, and surfaces two concrete, actionable findings: community service events are rated
uniformly low across every class year (a format problem, not a scheduling one), and alumni events show
the sharpest divide by class year of any event type (Sophomores highest, Seniors lowest).

## Key Findings

| Metric | Value |
|---|---|
| Model MAE (held-out real test set) | 0.80 |
| Baseline MAE (always predict most common rating) | 1.47 |
| Improvement over baseline | ~46% |
| Exact-match accuracy | 33.7% (vs. ~20% random-guess floor across 5 classes) |

- **Validated predictive levers:** class year (a consistent "senioritis" pattern — Rising Seniors rate
  most event types lower) and reason for skipping events. Day-of-week preference and total events
  attended were tested and explicitly ruled out as unreliable at this sample size — documented as
  negative findings, not gaps.
- **Community service** events are predicted to rate 4 or 5 for **0% of members, across every class
  year** — a uniform, chapter-wide problem worth a format rethink.
- **Alumni** events show the largest class-year divide in the dataset (100% of Sophomores vs. 0% of
  Seniors predicted to rate 4+) — a case for class-year-targeted outreach rather than one blanket
  approach.

## Methodology

1. **Cleaning:** renamed raw Google Forms columns, resolved a free-text edge case, encoded a
   multi-select day-preference field into independent binary flags, ordinally encoded attendance
   buckets, and reshaped from wide (one row per person) to long (one row per person × event type) for
   analysis.
2. **Lever validation:** checked candidate predictors against ratings using group means and sample
   sizes before trusting any of them, using a minimum group-size threshold to avoid drawing conclusions
   from too few responses.
3. **Synthetic data generation:** with only 43 real responses, built a generator that resamples real
   respondents' full profiles and generates new ratings from Laplace-smoothed, blended probability
   distributions based on the two validated levers (300 synthetic people, verified against real
   aggregate patterns on three separate checks before use).
4. **Leakage-safe train/test split:** since the synthetic generator's distributions were built from
   all 43 real people, a naive random split would let the model re-learn the generator's own math
   rather than demonstrate real predictive power. Real respondents were split *by person* (not by row)
   before combining with synthetic data — training on 30 real people + all synthetic data, evaluating
   only on 13 held-out real people never used as training rows.
5. **Modeling:** ratings are ordinal (1 < 2 < 3 < 4 < 5), so an ordinal logistic regression model
   (`mord.LogisticAT`) was used instead of standard multiclass classification, which would penalize a
   near-miss the same as a wildly wrong prediction.
6. **Evaluation:** Mean Absolute Error as the primary metric (since it respects the ordinal distance
   between predicted and actual ratings), benchmarked against a naive baseline.
7. **Deliverable:** predictions aggregated into a per-event-type, per-class-year likelihood table,
   plus a transparent "suggested relative emphasis" ranking weighted by the chapter's actual class-year
   composition.

## Limitations

- Based on 43 of ~100 chapter members — not a full census; response bias wasn't assessable from the
  data itself.
- The 13-person held-out test set was never used as direct training rows, but was still part of the
  pool that shaped the synthetic generator's probability distributions — not a fully independent test
  in the textbook sense, given the sample size involved.
- Only two levers (class year, reason for skipping) were validated strongly enough to use; others were
  tested and ruled out.
- The "suggested emphasis" ranking is a transparent heuristic (predicted rating × real class-year
  makeup), not a constrained optimization — it doesn't account for budget, venue capacity, or event
  frequency/fatigue, since the survey didn't collect that data.

## Project Structure

```
fiji-event-engagement-analytics/
├── data/
│   ├── raw/            # original survey export (excluded from git — see .gitignore)
│   └── processed/       # synthetic_data.csv, generated and verified in the notebook
├── notebooks/
│   └── eda.ipynb        # full pipeline: cleaning → EDA → synthetic generation → modeling → outputs
├── outputs/              # final CSV deliverables (per-event/class-year tables, model metrics)
├── requirements.txt
└── README.md
```

## Tech Stack

Python, pandas, numpy, scikit-learn, [mord](https://pypi.org/project/mord/) (ordinal regression),
matplotlib, Jupyter.

## Running This Project

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Open `notebooks/eda.ipynb` in VS Code or Jupyter and select the `venv` kernel. The notebook runs
top-to-bottom: cleaning → EDA/lever validation → synthetic data generation → modeling → final outputs.

## Author

Kevin Wehrle
