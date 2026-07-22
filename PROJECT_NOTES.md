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
- `class_year` (nominal, no order): one-hot encoded via `pd.get_dummies` → `class_Rising Sophomore/Junior/Senior` (kept all 3 columns for now, not `drop_first`, since model type isn't locked in yet — trees don't need it dropped, linear/logistic models would). Original text `class_year` column was also kept (not dropped) — needed later for groupby-based EDA and for the synthetic data generator.
- `events_attended` (ordinal — "0-3" < "4-6" < "7-9" < "10+"): converted to `pd.Categorical(ordered=True)`, then `.cat.codes` → new column `events_attended_ordinal` (0-3). Kept both the labeled and numeric versions.
- Reshaped the 8 `rating_*` columns from wide to long via `pd.melt()` → `df_long`, 43 people × 8 event types = 344 rows. New columns: `event_type` (stripped of redundant `rating_` prefix) and `rating` (the target variable, 1-5).
- `df` (wide, pre-melt, 43 rows × 26 cols) is kept alongside `df_long` — needed by the synthetic generator to resample full real person-profiles.

## Known data quirks to remember
- `events_attended` is heavily skewed: 29/43 (67%) selected "10+". Real signal, not an error.
- `wished_event` (21/43 filled) and `community_service_interest` (13/43 filled) are optional free text — expected missingness, not used as model features, maybe mined for qualitative color in the write-up later.

## EDA: lever validation for synthetic data (done — see `notebooks/eda.ipynb`)
Checked several candidate variables as potential predictors of `rating`, using groupby mean + count (flagging any group with too few people to trust), before deciding what to bake into the synthetic generator. Rule of thumb used throughout: a real lever should shift ratings in a *consistent direction across most of the 8 event types*, not just 1-2, and be backed by a reasonably-sized group (roughly 9+ people).

**Validated levers (used in generator):**
- **`class_year`** — strong, consistent signal. Rising Seniors rate lower than Juniors/Sophomores in 6 of 8 event types ("senioritis" pattern). Groups: Junior n=15, Senior n=11, Sophomore n=17.
- **`reason_skip_event`** — moderate signal. People citing "Not interested in event type" rate lower in 7 of 8 event types vs. "Conflicts with class/work" baseline. Groups: Conflicts n=21, Not interested n=12, Too tired n=9, **Cost n=1 (too small to trust — see below)**.

**Checked, ruled out (documented as negative/inconclusive findings for write-up, not used in generator):**
- **Day-of-week preferences** — Friday (40/43 yes) and Saturday (42/43 yes) are near-universal consensus, no usable comparison group. Sunday (24/19 split) and weekday evening (29/14 split) had enough people to check but showed no consistent directional pattern across event types (gaps small and inconsistent in direction) — real negative result, not just insufficient data.
- **`events_attended_ordinal`** — tempting (intuitive that engagement history predicts ratings) but ruled out: not monotonic (the 4-6 events bucket, n=4, frequently rated *higher* than the 10+ bucket, n=29, across several event types) and bottom 3 buckets are too small (n=4, 4, 6) to trust. One flagged observation for write-up: the n=4 lowest-attendance group showed notably low ratings specifically for community_service/philanthropy, but sample too small to confirm as general pattern.

## Synthetic Data Generator (built and verified — see `notebooks/eda.ipynb`, to be split into its own notebook soon)

**Approach:** resample real people's full profiles (Approach A) — every synthetic person is built by randomly copying one real respondent's entire profile (class_year, reason_skip_event, day prefs, events_attended_ordinal), then only the 8 event ratings are freshly generated. Chosen over generating fully-independent synthetic people from scratch, since we've only validated 2 real levers — inventing new feature *combinations* wholesale would risk unvalidated correlations.

**Rating generation mechanism:**
1. For each (class_year, event_type) pair: build a probability distribution over ratings 1-5 from the 43 real responses, using **Laplace (additive) smoothing** with **alpha = 0.3** (chosen deliberately lower than the textbook default of 1, based on domain knowledge that alpha=1 over-diluted strongly-skewed real patterns like socials, e.g. pulled Rising Sophomore/socials from a real 82% "rated 5" down to 68%; alpha=0.3 keeps it close to 77%). Smoothing exists so no rating is ever *literally* impossible (0% chance) for a group just because none of the ~11-17 real respondents in that small sample happened to pick it.
2. Same smoothing process repeated for `reason_skip_event` × event_type.
3. An **overall (event_type-only) fallback distribution** is used specifically when `reason_skip_event == "Cost"`, since that category only has 1 real respondent and its own distribution isn't trustworthy.
4. The two distributions (class_year-based, skip_reason-based) are combined via a **weighted average**: 0.65 × class_year distribution + 0.35 × skip_reason distribution. Class_year weighted higher since it showed the stronger, more consistent real pattern.
5. One actual rating (1-5) is drawn from the final blended distribution via `np.random.choice`, per event_type, per synthetic person.

**Verification done along the way:** hand-calculated smoothing math confirmed against code output; confirmed the alpha parameter was actually wired into the smoothing formula after catching a bug where changing the default didn't change output (default was declared but never referenced in the function body — fixed); confirmed blended distributions sum to 1.0 and matched hand-calculated values; ran 1000 samples from a known distribution and confirmed proportions matched expected probabilities; generated a 5-person test batch and visually verified structure (8 rows/person, consistent copied fields, plausible varied ratings).

**Output format:** long format (same shape as `df_long`) — chosen because it's directly usable for the modeling/insights work ahead, since a real synthetic survey-response "look" isn't needed, just usable rows for analysis.

**n_people = 300** (2,400 long-format rows). Chosen as roughly 3x actual chapter size (~100 members) — enough for stable aggregate stats and future modeling without implying more precision than 43 real responses can actually support. Documented as a deliberate, justifiable choice rather than an arbitrary round number.

**Final verification (step 6, complete):** compared real vs. synthetic aggregates on 3 checks —
1. Mean rating by class_year: ranking preserved in synthetic (Sophomore > Junior > Senior), values within ~0.1-0.2 of real
2. Mean rating by reason_skip_event: ranking preserved (Conflicts highest, Not interested lowest), spread compressed as expected from blending/smoothing
3. Overall 1-5 rating distribution shape: real and synthetic bar charts nearly identical (both climb steadily from ~3-5% at rating 1 to ~30% at rating 5)

All three passed — generator is considered done and verified.

## Project structure
Everything currently lives in `notebooks/eda.ipynb` (cleaning → EDA/lever validation → synthetic data generator), organized internally with markdown section headers (collapsible) rather than split across files/notebooks. Decided against splitting generator code into `src/` for now — that adds import/path complexity without changing the actual analysis, and isn't worth the friction mid-project. Plan is to do a single reorganization pass (proper `src/` split, multiple notebooks per the cheatsheet's intended structure) at the very end, once the full project (through modeling + write-up) is done and its real shape is known — easier to organize well in hindsight than to guess upfront.

`data/processed/` still used for saved outputs (`df_wide.csv`, `df_long.csv`, `synthetic_data.csv`) via `.to_csv()` — this part's worth keeping regardless of notebook structure, since it lets any later notebook (e.g. modeling) load results without re-running everything from scratch.

## Next phase: predictive modeling / insights (not started)
Goal per original scope: per-event-type attendance likelihood table to drive Fall event planning. Planned next steps:
- New notebook, likely `notebooks/modeling.ipynb`
- Decide model type (likely classification/ordinal regression on `rating`, features = class_year, reason_skip_event, event_type, possibly day prefs even though not validated as strong solo levers — worth revisiting with a model that can weigh multiple weak signals together, unlike the simple groupby checks used for the generator)
- Train on `synthetic_data.csv` (or a combination of real + synthetic), evaluate
- Produce final actionable output: per-event-type likelihood table for exec board, plus write-up covering methodology, validated/ruled-out levers, and the "why 300, why alpha=0.3, why weight=0.65" design choices

## Environment
- Local venv (pandas, numpy, matplotlib, seaborn, scikit-learn, ipykernel), VS Code, git initialized and pushed to GitHub
- Learning style: step-by-step, concept-explained-before-code, no unexplained finished code dumps
