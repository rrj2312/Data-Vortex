# Social Engine — Round 1, Phase 1: Data Recovery & EDA

Team submission for Round 1 of the Social Engine data hackathon: recovering and restoring a deliberately corrupted social-media dataset, then performing exploratory analysis.

## Repo Contents

| File | Description |
|---|---|
| `Social_Engine_Round1.ipynb` | **Main deliverable.** Full notebook: load → clean → save → EDA, with inline commentary and charts. Start here. |
| `clean_data.py` | Standalone script version of the cleaning pipeline (same logic as the notebook, for running outside Jupyter). |
| `extended_eda.py` | Deeper EDA: distributions, correlation analysis, cross-cuts by platform/text-presence, follower-vs-engagement scatter. Generates the `chart_*.png` files. |
| `Social_Engine_Posts_Cleaned.csv` / `.json` | Cleaned posts dataset (final output). |
| `Social_Engine_Users_Cleaned.csv` | Cleaned users dataset. |
| `eda_summary.json` | Machine-readable summary stats from the base cleaning pass. |
| `extended_eda.json` | Machine-readable stats from the deeper analysis (correlations, cross-cuts). |
| `chart_*.png` | Saved chart images (engagement distributions, platform comparison, correlation matrix, posts-over-time, followers-vs-likes). |
| `EDA_Report.md` | Written report: corruption inventory, cleaning decisions with justification, and key findings. |

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the Results

**Option A — Notebook (recommended):**
```bash
jupyter notebook Social_Engine_Round1.ipynb
```
Run all cells top to bottom. This regenerates the cleaned CSV/JSON files and inline charts.

**Option B — Scripts:**
```bash
python clean_data.py       # cleans raw data, produces Social_Engine_Posts_Cleaned.* and eda_summary.json
python extended_eda.py     # deeper EDA + chart images, requires clean_data.py to have run first
```

Input files expected in the same directory: `Social_Engine_Posts_Corrupted.csv`, `Social_Engine_Users.csv`.

## Cleaning Assumptions (summary — full reasoning in `EDA_Report.md`)

- **Duplicate `post_id`s** (360 found): kept first occurrence, treating repeats as corruption artifacts of a unique key rather than distinct posts.
- **Negative `likes`** (509 found): corrected via absolute value, assuming a sign-flip corruption rather than a different metric.
- **Missing `platform`**: labeled `"Unknown"` explicitly rather than imputed, to avoid biasing platform-level stats.
- **Missing `text_content`**: row kept, flagged with a `has_text` boolean rather than dropped, since engagement metrics are still valid.
- **Mixed timestamp formats** (Unix epoch / ISO 8601 / `DD-MM-YYYY`): unified to ISO 8601 UTC; day-only dates lose time precision (set to midnight), which is documented, not fabricated.
- **Text corruption** (HTML entities, mojibake, stray tags): unescaped, re-encoded, and stripped respectively.

## Key EDA Finding

Correlations between `likes`, `shares`, `comments`, and follower count are all near zero (~±0.02) — engagement does not scale with audience size, and the engagement metrics don't move together either. This points to the values being randomly generated for the challenge rather than reflecting organic behavior, which is itself a useful observation about the dataset's structure.
