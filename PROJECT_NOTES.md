# FIJI Event Engagement Analytics — Project Notes

Living status doc. Update after each work session to pick up context instantly.

## Goal
Predict per-event-type attendance likelihood (1–5 rating) to help plan a Fall event calendar members will actually attend. Chose this over predicting overall `events_attended` bucket because it's actionable (gives a per-event-type table for exec board) and forces a resume-relevant wide-to-long reshape.

## Data
- Anonymous Google Form, ~100-member chapter, currently 43 real responses (still collecting)
- Raw CSV lives in `data/raw/` (gitignored — never commit raw survey exports)
- Survey covers: class year, events attended last semester (bucketed), 8 event-type ratings (1-5), day/time preferences (multi-select), reason to skip, how they hear about events, satisfaction, 2 optional free-text fields

## Cleaning decisions made so far (in `notebooks/eda.ipynb`)
- Renamed all 17 raw Google Forms columns to snake_case (`column_map` dict)
- `reason_skip_event`: one free-text "Other" response ("Lazy af") manually folded into `"Too tired/burnt out"` — judgment call, documented in a code comment, since it's semantically equivalent self-reported low motivation
- `preferred_days` (multi-select, e.g. "Friday, Saturday, Sunday"): split into 5 binary columns (`pref_friday`, `pref_saturday`, `pref_sunday`, `pref_weekday_afternoon`, `pref_weekday_evening`) via `.str.get_dummies(', ')` — this is multi-label (multiple can be true per person), NOT the same as one-hot encoding
- `class_year` (nominal, no order): one-hot encoded via `pd.get_dummies` → `class_Rising Sophomore/Junior/Senior` (kept all 3 columns for now, not `drop_first`, since model type isn't locked in yet — trees don't need it dropped, linear/logistic models would)
- `events_attended` (ordinal — "0-3" < "4-6" < "7-9" < "10+"): converted to `pd.Categorical(ordered=True)`, then `.cat.codes` → new column `events_attended_ordinal` (0-3). Kept both the labeled and numeric versions.
- Reshaped the 8 `rating_*` columns from wide to long via `pd.melt()` → `df_long`, 43 people × 8 event types = 344 rows. New columns: `event_type` (stripped of redundant `rating_` prefix) and `rating` (the target variable, 1-5).

## Known data quirks to remember
- `events_attended` is heavily skewed: 29/43 (67%) selected "10+". Real signal, not an error — synthetic generator should respect this skew rather than inventing an even spread, and it's worth a sentence in the final write-up.
- `wished_event` (21/43 filled) and `community_service_interest` (13/43 filled) are optional free text — expected missingness, not used as model features, maybe mined for qualitative color in the write-up later.

## Synthetic data plan (not started yet)
Plan is to derive conditional probabilities for the generator from patterns actually observed in the real 43 responses (e.g., does class year visibly shift ratings by event type in `df_long`?) rather than inventing "believable" assumptions from scratch — stronger to say "derived from EDA" than "hand-tuned." Must include an `is_synthetic` boolean flag so real vs. generated rows stay distinguishable, and document every conditional assumption as a comment.

## Environment
- Local venv (pandas, numpy, matplotlib, seaborn, scikit-learn, ipykernel), VS Code, git initialized and pushed to GitHub
- Learning style: step-by-step, concept-explained-before-code, no unexplained finished code dumps
