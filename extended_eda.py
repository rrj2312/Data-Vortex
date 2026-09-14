"""
Social Engine — Round 1, Phase 1: Extended EDA & Visualizations
==================================================================
Builds on clean_social_engine_data.py's cleaned output to produce
deeper insight-discovery: distributions, correlations, cross-cuts
by platform/text-presence, and saved chart images for the report.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import ast

plt.rcParams["figure.dpi"] = 110

posts = pd.read_csv("Social_Engine_Posts_Cleaned.csv")
users = pd.read_csv("Social_Engine_Users_Cleaned.csv")

# hashtags/mentions were saved as stringified lists in CSV -> parse back
posts["hashtags"] = posts["hashtags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
posts["timestamp"] = pd.to_datetime(posts["timestamp"])
merged = posts.merge(users, on="user_id", how="left")

out = {}

# ---------------------------------------------------------------
# 1. ENGAGEMENT DISTRIBUTIONS (likes/shares/comments)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["likes", "shares", "comments"]):
    ax.hist(posts[col].dropna(), bins=40, color="#4C72B0", edgecolor="white")
    ax.set_title(f"{col.capitalize()} distribution")
    ax.set_xlabel(col)
plt.tight_layout()
plt.savefig("chart_engagement_distributions.png")
plt.close()

# ---------------------------------------------------------------
# 2. AVG ENGAGEMENT BY PLATFORM
# ---------------------------------------------------------------
by_platform = posts.groupby("platform")[["likes", "shares", "comments"]].mean().round(1)
fig, ax = plt.subplots(figsize=(8, 4.5))
by_platform.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_title("Average engagement by platform")
ax.set_ylabel("Average count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("chart_engagement_by_platform.png")
plt.close()
out["avg_engagement_by_platform"] = by_platform.to_dict()

# ---------------------------------------------------------------
# 3. DOES MISSING TEXT CORRELATE WITH ENGAGEMENT?
# ---------------------------------------------------------------
by_text_flag = posts.groupby("has_text")[["likes", "shares", "comments"]].mean().round(1)
out["engagement_by_has_text"] = by_text_flag.to_dict()
fig, ax = plt.subplots(figsize=(6, 4))
by_text_flag.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_title("Engagement: posts with vs. without recovered text")
ax.set_xticklabels(["No text", "Has text"], rotation=0)
plt.tight_layout()
plt.savefig("chart_engagement_by_has_text.png")
plt.close()

# ---------------------------------------------------------------
# 4. CORRELATION MATRIX (likes/shares/comments/follower_count)
# ---------------------------------------------------------------
users["follower_count"] = pd.to_numeric(users["follower_count"], errors="coerce")
corr_df = merged[["likes", "shares", "comments", "follower_count"]].corr()
out["correlation_matrix"] = corr_df.round(3).to_dict()

fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_df.columns)))
ax.set_yticks(range(len(corr_df.columns)))
ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
ax.set_yticklabels(corr_df.columns)
for i in range(len(corr_df.columns)):
    for j in range(len(corr_df.columns)):
        ax.text(j, i, f"{corr_df.iloc[i,j]:.2f}", ha="center", va="center", color="black")
fig.colorbar(im)
ax.set_title("Correlation matrix")
plt.tight_layout()
plt.savefig("chart_correlation_matrix.png")
plt.close()

# ---------------------------------------------------------------
# 5. TOP HASHTAGS BY PLATFORM (cross-cut)
# ---------------------------------------------------------------
exploded = posts.explode("hashtags").dropna(subset=["hashtags"])
top_by_platform = (
    exploded.groupby(["platform", "hashtags"]).size()
    .reset_index(name="count")
    .sort_values(["platform", "count"], ascending=[True, False])
    .groupby("platform").head(3)
)
out["top_3_hashtags_per_platform"] = (
    top_by_platform.groupby("platform")
    .apply(lambda d: list(zip(d["hashtags"], d["count"])))
    .to_dict()
)

# ---------------------------------------------------------------
# 6. POSTING VOLUME OVER TIME
# ---------------------------------------------------------------
by_month = posts.set_index("timestamp").resample("ME").size()
fig, ax = plt.subplots(figsize=(9, 4))
by_month.plot(ax=ax, color="#4C72B0", marker="o", markersize=3)
ax.set_title("Post volume over time")
ax.set_ylabel("Number of posts")
plt.tight_layout()
plt.savefig("chart_posts_over_time.png")
plt.close()

# ---------------------------------------------------------------
# 7. FOLLOWER COUNT vs. AVG LIKES (does audience size predict engagement?)
# ---------------------------------------------------------------
user_engagement = merged.groupby("user_id").agg(
    avg_likes=("likes", "mean"),
    follower_count=("follower_count", "first")
).dropna()
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(user_engagement["follower_count"], user_engagement["avg_likes"], alpha=0.3, s=12, color="#4C72B0")
ax.set_xlabel("Follower count")
ax.set_ylabel("Average likes per post")
ax.set_title("Follower count vs. average likes")
plt.tight_layout()
plt.savefig("chart_followers_vs_likes.png")
plt.close()
out["follower_likes_correlation"] = round(
    user_engagement["follower_count"].corr(user_engagement["avg_likes"]), 3
)

with open("extended_eda.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

print("Saved 6 charts + extended_eda.json")
print(json.dumps(out, indent=2, default=str)[:1500])
