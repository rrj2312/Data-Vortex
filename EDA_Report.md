# Social Engine — Round 1, Phase 1: EDA Report

## 1. Corruption Inventory (Dataset 1)

| Issue | Count | Column(s) |
|---|---|---|
| Duplicate `post_id` | 360 | posts |
| Missing `platform` | 1,846 | posts |
| Missing `text_content` | 1,746 | posts |
| Missing `likes` | 1,858 | posts |
| Negative `likes` (impossible) | 509 | posts |
| Mixed timestamp formats (epoch / ISO 8601 / DD-MM-YYYY) | all 12,360 rows | posts |
| HTML entities (`&amp;`) | 341 | text_content |
| Mojibake (`Ã©` → `é`) | 316 | text_content |
| Stray HTML tags (`<div>`, `<br>`) | 663 | text_content |
| Literal string `"NULL"` used as a null marker | 1,900 | posts (multiple cols) |

## 2. Cleaning Decisions & Justification

- **Null standardization**: `""`, `"NULL"`, `"None"`, whitespace-only → real `NaN`. Needed because three different missing-value conventions were mixed in the same file, which would otherwise break `.isna()`/`.fillna()` logic silently.
- **Deduplication**: kept the **first** occurrence of each `post_id`. A post_id is treated as a unique key, so repeats are corruption artifacts rather than distinct posts. *(Alternative: keep-last, if later duplicates are treated as corrected versions — worth stating as a limitation.)*
- **Negative likes → absolute value**: assumed a sign-flip corruption rather than a different metric, since `likes` is defined as a non-negative count. Chosen over dropping the row (loses shares/comments/text signal) or nulling (loses the value entirely). *(Alternative: treat as missing if a more conservative policy is preferred.)*
- **Timestamp normalization**: unified three formats (Unix epoch, ISO 8601, `DD-MM-YYYY`) into a single ISO 8601 UTC string. `DD-MM-YYYY` rows lose time-of-day precision (set to `00:00:00Z`) — documented, not fabricated.
- **Missing `platform` → `"Unknown"`**: explicit label rather than imputing the mode, to avoid biasing platform-level engagement stats.
- **Missing `text_content`**: kept the row (engagement metrics are still valid data) but added a `has_text` boolean flag so text-only analyses (hashtags, sentiment) can filter cleanly instead of treating an empty string as content.
- **Text encoding fixes**: HTML-unescaped entities, corrected Latin-1/UTF-8 mojibake, stripped stray HTML tags, collapsed embedded newlines from multi-line quoted CSV fields into single-line text.

## 3. Post-Cleaning Dataset

- **12,000** unique posts (from 12,360 raw rows; 360 duplicates removed)
- **1,500** users, joined on `user_id`

## 4. Key Findings

- **Platform split** is fairly even across Facebook, YouTube, Twitter, Reddit, Instagram (~2,000 each); ~1,780 posts (15%) had platform unrecoverable and are labeled `Unknown`.
- **Engagement is comparable across platforms** — average likes range only from ~2,438 (Twitter) to ~2,529 (Facebook), suggesting no single platform dominates engagement in this dataset. See `chart_engagement_by_platform.png`.
- **~14% of posts have no text content** (1,711 of 12,000) — worth flagging as a data-quality ceiling for any NLP/sentiment work in later rounds. Interestingly, engagement is *not* lower for these posts (see `chart_engagement_by_has_text.png`), so missing text isn't a proxy for low-quality/low-engagement content.
- **Top hashtags** cluster around generic marketing/review language (`#Reviews`, `#Fitness`, `#BestValue`, `#Eco`, `#SpecialOffer`) with fairly even frequency (~715–750 each), suggesting these were drawn from a fixed tag pool rather than organic user tagging. Breaking this down by platform shows different platforms favor slightly different top tags — see `extended_eda.json` → `top_3_hashtags_per_platform`.
- **Geographic spread** is global and fairly uniform (~400–460 posts per top city), no single market dominates.
- **Engagement metrics are statistically independent of each other and of audience size.** The correlation matrix (`chart_correlation_matrix.png`) shows `likes`, `shares`, `comments`, and `follower_count` all correlate at ~±0.02 — essentially zero. In real social data, higher-follower accounts and higher-like posts typically show *some* positive correlation with shares/comments; the complete absence of it here indicates the engagement numbers were most likely randomly generated for the challenge rather than simulating organic behavior. This is a structural property of the dataset worth stating explicitly rather than something to "fix."
- **Posting volume over time** (`chart_posts_over_time.png`) shows no strong seasonal trend, consistent with synthetically generated timestamps rather than a real platform's usage pattern.

## 5. Files Produced

- `Social_Engine_Round1.ipynb` — full notebook: cleaning + EDA in one reproducible document
- `Social_Engine_Posts_Cleaned.csv` / `.json` — cleaned posts
- `Social_Engine_Users_Cleaned.csv` — cleaned users
- `eda_summary.json` — base cleaning summary stats
- `extended_eda.json` — deeper EDA stats (correlations, cross-cuts by platform/text presence)
- `chart_engagement_distributions.png`, `chart_engagement_by_platform.png`, `chart_engagement_by_has_text.png`, `chart_correlation_matrix.png`, `chart_posts_over_time.png`, `chart_followers_vs_likes.png` — supporting visuals
- `clean_data.py` — standalone cleaning script
- `extended_eda.py` — standalone deeper-EDA script
- `README.md` — setup and reproduction instructions
- `requirements.txt` — pinned dependencies
