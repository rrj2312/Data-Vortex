"""
Social Engine — Round 1, Phase 1: Data Recovery & Restoration
================================================================
Restores the corrupted Dataset 1 (Social_Engine_Posts_Corrupted.csv)
and merges it with Social_Engine_Users.csv for downstream EDA.

Every transformation below is intentionally kept explicit and
commented so the reasoning can be lifted straight into the
submission's "assumptions" documentation.
"""

import pandas as pd
import numpy as np
import re
import html
import json
from datetime import datetime, timezone

pd.set_option("display.max_columns", None)

RAW_POSTS_PATH = "Social_Engine_Posts_Corrupted.csv"
RAW_USERS_PATH = "Social_Engine_Users.csv"

# ---------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------
posts = pd.read_csv(RAW_POSTS_PATH, dtype=str, keep_default_na=False)
users = pd.read_csv(RAW_USERS_PATH, dtype=str, keep_default_na=False)

print(f"Raw posts: {posts.shape}, Raw users: {users.shape}")

# ---------------------------------------------------------------
# 2. STANDARDISE NULL REPRESENTATIONS
# ---------------------------------------------------------------
# The corruption injected several stand-ins for "missing":
# true empty string, the literal text "NULL", and whitespace-only
# cells. We collapse all of these to a real NaN so pandas' own
# missing-value tooling (isna, fillna, dropna) works uniformly.
NULL_TOKENS = {"", "NULL", "null", "NaN", "nan", "None"}

def normalize_nulls(df):
    return df.apply(lambda col: col.map(lambda v: np.nan if str(v).strip() in NULL_TOKENS else v))

posts = normalize_nulls(posts)
users = normalize_nulls(users)

# ---------------------------------------------------------------
# 3. DEDUPLICATE POSTS
# ---------------------------------------------------------------
# 360 duplicate post_ids were found. Assumption: a post_id is a
# unique key, so a repeat is a duplicated recovery-log entry
# (not two distinct posts). We keep the FIRST occurrence, since in
# the raw file earlier rows are less likely to have been touched
# by the corruption process a second time. This is a documented
# assumption — a defensible alternative is "keep last" if later
# entries are treated as corrections.
dupes_before = posts["post_id"].duplicated().sum()
posts = posts.drop_duplicates(subset="post_id", keep="first").reset_index(drop=True)
print(f"Dropped {dupes_before} duplicate post_id rows.")

# ---------------------------------------------------------------
# 4. FIX TEXT ENCODING / MARKUP CORRUPTION
# ---------------------------------------------------------------
# Three distinct text corruptions were detected in text_content:
#   a) HTML entities, e.g. "&amp;"          -> unescape
#   b) Mojibake from a UTF-8/Latin-1 mismatch, e.g. "Ã©" -> "é"
#   c) Stray leftover HTML tags, e.g. "<div>", "<br>"     -> strip
#   d) Embedded newlines inside a single logical post (multi-line
#      quoted CSV fields) -> collapse to single spaces.
def fix_mojibake(text):
    if pd.isna(text):
        return text
    try:
        # Round-trips text that was decoded as UTF-8 but originally
        # encoded as Latin-1 (the classic "Ã©" pattern for "é").
        return text.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

def clean_text(text):
    if pd.isna(text):
        return text
    t = html.unescape(text)          # &amp; -> &
    t = fix_mojibake(t)              # Ã© -> é
    t = re.sub(r"<[^>]+>", "", t)    # strip stray HTML tags
    t = re.sub(r"\s+", " ", t).strip()  # collapse embedded newlines/whitespace
    return t

posts["text_content"] = posts["text_content"].apply(clean_text)

# ---------------------------------------------------------------
# 5. NORMALISE TIMESTAMPS
# ---------------------------------------------------------------
# Three formats coexist in one column:
#   - Unix epoch seconds, e.g. "1731041269"
#   - ISO 8601,            e.g. "2025-02-03T02:09:31"
#   - DD-MM-YYYY,          e.g. "28-05-2024"  (day precision only)
# All are converted to a single ISO 8601 UTC string. DD-MM-YYYY
# rows have no time component, so they're set to midnight UTC —
# documented as a precision loss, not a fabrication.
def normalize_timestamp(ts):
    if pd.isna(ts):
        return np.nan
    ts = ts.strip()
    # Unix epoch: all digits
    if re.fullmatch(r"\d{9,10}", ts):
        try:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, OSError):
            return np.nan
    # ISO 8601
    if "T" in ts:
        try:
            return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return np.nan
    # DD-MM-YYYY
    m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", ts)
    if m:
        d, mo, y = m.groups()
        try:
            return datetime(int(y), int(mo), int(d), tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return np.nan
    return np.nan  # unrecognized format -> flagged as missing

posts["timestamp_clean"] = posts["timestamp"].apply(normalize_timestamp)
bad_ts = posts["timestamp_clean"].isna().sum()
print(f"Timestamps that failed to parse (set to NaN): {bad_ts}")

# ---------------------------------------------------------------
# 6. FIX NUMERIC COLUMNS (likes, shares, comments)
# ---------------------------------------------------------------
for col in ["likes", "shares", "comments"]:
    posts[col] = pd.to_numeric(posts[col], errors="coerce")

# Negative likes are physically impossible. Assumption: this is a
# sign-flip corruption (a plausible injected fault), not a
# semantically different metric — so we take the absolute value
# rather than dropping the row or imputing, preserving the most
# information. Document the alternative (treat as missing) if a
# stricter policy is preferred.
neg_likes = (posts["likes"] < 0).sum()
posts["likes"] = posts["likes"].abs()
print(f"Corrected {neg_likes} negative 'likes' values via absolute value.")

# ---------------------------------------------------------------
# 7. FIX MISSING PLATFORM
# ---------------------------------------------------------------
# No reliable way to infer the true platform from other fields,
# so missing platform is labeled explicitly rather than dropped
# or silently imputed to the mode (which would bias platform-level
# EDA).
posts["platform"] = posts["platform"].fillna("Unknown")

# ---------------------------------------------------------------
# 8. HANDLE MISSING text_content
# ---------------------------------------------------------------
# Rows with no text are kept (they still carry valid engagement
# metrics/platform/user), but flagged with a boolean so text-based
# analyses (hashtag extraction, sentiment, etc.) can filter them
# out cleanly instead of treating empty string as real content.
posts["has_text"] = posts["text_content"].notna()

# ---------------------------------------------------------------
# 9. EXTRACT STRUCTURED SIGNALS FOR EDA
# ---------------------------------------------------------------
def extract_hashtags(text):
    if pd.isna(text):
        return []
    return re.findall(r"#(\w+)", text)

def extract_mentions(text):
    if pd.isna(text):
        return []
    return re.findall(r"@(\w+)", text)

posts["hashtags"] = posts["text_content"].apply(extract_hashtags)
posts["mentions"] = posts["text_content"].apply(extract_mentions)

# ---------------------------------------------------------------
# 10. FINAL SCHEMA + SAVE
# ---------------------------------------------------------------
final_cols = [
    "post_id", "user_id", "platform", "text_content", "has_text",
    "timestamp_clean", "likes", "shares", "comments",
    "hashtags", "mentions"
]
posts_clean = posts[final_cols].rename(columns={"timestamp_clean": "timestamp"})

posts_clean.to_csv("Social_Engine_Posts_Cleaned.csv", index=False)

# JSON export needs list columns serialized properly
posts_json = posts_clean.copy()
posts_json.to_json("Social_Engine_Posts_Cleaned.json", orient="records", indent=2)

users.to_csv("Social_Engine_Users_Cleaned.csv", index=False)

print("\nSaved: Social_Engine_Posts_Cleaned.csv / .json, Social_Engine_Users_Cleaned.csv")
print(posts_clean.head(3))

# ---------------------------------------------------------------
# 11. QUICK EDA SUMMARY (for the report)
# ---------------------------------------------------------------
summary = {}
summary["total_posts_after_cleaning"] = len(posts_clean)
summary["duplicate_rows_removed"] = int(dupes_before)
summary["negative_likes_corrected"] = int(neg_likes)
summary["platform_distribution"] = posts_clean["platform"].value_counts().to_dict()
summary["posts_missing_text"] = int((~posts_clean["has_text"]).sum())
summary["timestamps_unparseable"] = int(bad_ts)
summary["avg_likes"] = float(posts_clean["likes"].mean())
summary["avg_shares"] = float(posts_clean["shares"].mean())
summary["avg_comments"] = float(posts_clean["comments"].mean())

all_hashtags = [h for tags in posts_clean["hashtags"] for h in tags]
top_hashtags = pd.Series(all_hashtags).value_counts().head(10).to_dict()
summary["top_10_hashtags"] = top_hashtags

merged = posts_clean.merge(users, on="user_id", how="left")
summary["top_10_locations_by_post_volume"] = merged["location"].value_counts().head(10).to_dict()
summary["avg_likes_by_platform"] = merged.groupby("platform")["likes"].mean().round(1).to_dict()

with open("eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\n=== EDA SUMMARY ===")
print(json.dumps(summary, indent=2, default=str))
