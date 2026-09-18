"""
Data Vortex A'26 - Round 2: Rebuilding the Social Engine (Semantic Layer)
=========================================================================
NLP pipeline for 3-class sentiment classification on Dataset 2
(Labeled_Social_NLP_Training_Data.csv).

Everything in the submitted reports is produced by running this file.
No metric is hardcoded.

Usage:  python nlp_pipeline.py
"""

import os, re, json, html, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, roc_auc_score)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA = "/mnt/user-data/uploads/Labeled_Social_NLP_Training_Data.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = {}


def section(t):
    print("\n" + "=" * 70); print(t); print("=" * 70)


# ============================================================
# 1. LOAD + DATA AUDIT
# ============================================================
section("1. DATA AUDIT")
df = pd.read_csv(DATA)
print(f"Rows: {len(df)}   Columns: {list(df.columns)}")
print("\nSentiment distribution:\n", df.sentiment_label.value_counts().to_string())
print("\nTopic distribution:\n", df.topic_category.value_counts().to_string())
print("\nNulls:\n", df.isna().sum().to_string())

n_dup = df.post_text.duplicated().sum()
dup_groups = df[df.post_text.duplicated(keep=False)].groupby("post_text").sentiment_label.nunique()
n_conflict = int((dup_groups > 1).sum())
print(f"\nDuplicate post_text rows: {n_dup}")
print(f"Duplicate text groups: {len(dup_groups)}  |  groups with conflicting labels: {n_conflict}")

RESULTS["audit"] = {
    "n_rows": int(len(df)),
    "sentiment_distribution": df.sentiment_label.value_counts().to_dict(),
    "topic_distribution": df.topic_category.value_counts().to_dict(),
    "n_missing": int(df.isna().sum().sum()),
    "n_duplicate_text_rows": int(n_dup),
    "n_duplicate_groups": int(len(dup_groups)),
    "n_duplicate_groups_with_conflicting_labels": n_conflict,
    "char_len_mean": float(df.post_text.str.len().mean()),
    "char_len_min": int(df.post_text.str.len().min()),
    "char_len_max": int(df.post_text.str.len().max()),
}

# Is topic_category independent of sentiment? (chi-square)
from scipy.stats import chi2_contingency
ct = pd.crosstab(df.sentiment_label, df.topic_category)
chi2, p, dof, _ = chi2_contingency(ct)
print(f"\nChi-square (sentiment x topic): chi2={chi2:.2f}, p={p:.4f}")
RESULTS["audit"]["sentiment_topic_chi2"] = {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


# ============================================================
# 2. PREPROCESSING
# ============================================================
section("2. PREPROCESSING")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")


def clean_text(s, strip_user=True, keep_hashtag_word=True, unescape=True):
    """Deterministic, reversible-by-flag cleaning. Each step is ablated below."""
    if unescape:
        s = html.unescape(html.unescape(s))          # double-encoded entities seen in Dataset 1
    s = s.replace('"""', ' ').replace('""', ' ')      # CSV over-quoting artifact
    s = URL_RE.sub(" URLTOKEN ", s)
    if strip_user:
        s = USER_RE.sub(" USERTOKEN ", s)
    if keep_hashtag_word:
        s = HASH_RE.sub(r" \1 ", s)                   # #Barcelona -> Barcelona (keeps semantics)
    s = WS_RE.sub(" ", s).strip()
    return s


df["clean_text"] = df.post_text.apply(clean_text)

# --- Deduplicate BEFORE splitting (leakage control) ---
before = len(df)
df_model = df.drop_duplicates(subset="post_text", keep="first").reset_index(drop=True)
print(f"Deduplicated: {before} -> {len(df_model)} rows ({before - len(df_model)} exact duplicates removed)")
print("Post-dedup sentiment balance:\n", df_model.sentiment_label.value_counts().to_string())
RESULTS["preprocessing"] = {
    "rows_before_dedup": before,
    "rows_after_dedup": int(len(df_model)),
    "rows_removed": int(before - len(df_model)),
    "post_dedup_distribution": df_model.sentiment_label.value_counts().to_dict(),
}

X = df_model.clean_text.values
y = df_model.sentiment_label.values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain: {len(X_tr)}   Test: {len(X_te)}  (stratified 80/20)")
RESULTS["split"] = {"train": int(len(X_tr)), "test": int(len(X_te)), "stratified": True,
                    "test_size": 0.2, "random_state": RANDOM_STATE}


# ============================================================
# 3. MODEL SELECTION - 5-fold CV on TRAIN only
# ============================================================
section("3. MODEL SELECTION (5-fold stratified CV on training set)")

word_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                             strip_accents="unicode", lowercase=True)
char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                             sublinear_tf=True, lowercase=True)


def make_pipe(clf, feats="word"):
    if feats == "word":
        vec = ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                        strip_accents="unicode"))
    elif feats == "char":
        vec = ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                        sublinear_tf=True))
    else:  # union
        vec = ("tfidf", FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  strip_accents="unicode")),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                  sublinear_tf=True))]))
    return Pipeline([vec, ("clf", clf)])


candidates = {
    "Majority baseline":        Pipeline([("tfidf", TfidfVectorizer()), ("clf", DummyClassifier(strategy="most_frequent"))]),
    "MultinomialNB (word)":     make_pipe(MultinomialNB(alpha=0.5), "word"),
    "ComplementNB (word)":      make_pipe(ComplementNB(alpha=0.5), "word"),
    "LogisticRegression (word)": make_pipe(LogisticRegression(max_iter=2000, C=5, random_state=RANDOM_STATE), "word"),
    "LinearSVC (word)":         make_pipe(LinearSVC(C=0.5, random_state=RANDOM_STATE), "word"),
    "LinearSVC (char)":         make_pipe(LinearSVC(C=0.5, random_state=RANDOM_STATE), "char"),
    "LinearSVC (word+char)":    make_pipe(LinearSVC(C=0.5, random_state=RANDOM_STATE), "union"),
    "LogisticRegression (word+char)": make_pipe(LogisticRegression(max_iter=2000, C=5, random_state=RANDOM_STATE), "union"),
    "SGD hinge (word+char)":    make_pipe(SGDClassifier(loss="hinge", alpha=1e-5, max_iter=3000,
                                                        random_state=RANDOM_STATE), "union"),
    "RandomForest (word)":      make_pipe(RandomForestClassifier(n_estimators=150, n_jobs=1,
                                                                 random_state=RANDOM_STATE), "word"),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
    acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_rows.append({"model": name, "cv_f1_macro": scores.mean(), "cv_f1_std": scores.std(),
                    "cv_accuracy": acc.mean()})
    print(f"{name:34s} F1-macro {scores.mean():.4f} (+/-{scores.std():.4f})   Acc {acc.mean():.4f}")

cv_df = pd.DataFrame(cv_rows).sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)
cv_df.to_csv(f"{OUT}/model_selection_cv.csv", index=False)
RESULTS["model_selection"] = cv_df.to_dict(orient="records")
best_name = cv_df.iloc[0]["model"]
print(f"\nBest by CV F1-macro: {best_name}")


# ============================================================
# 4. PREPROCESSING ABLATION (justifies each cleaning step)
# ============================================================
section("4. PREPROCESSING ABLATION (LinearSVC word+char, 5-fold CV F1-macro)")
raw_sub = df_model.post_text.values
ablations = {
    "Raw text (no cleaning)":        [clean_text(t, unescape=False, strip_user=False, keep_hashtag_word=False) if False else t for t in raw_sub],
    "+ HTML unescape + quote fix":   [clean_text(t, strip_user=False, keep_hashtag_word=False) for t in raw_sub],
    "+ @user -> USERTOKEN":          [clean_text(t, strip_user=True, keep_hashtag_word=False) for t in raw_sub],
    "+ hashtag word split (FULL)":   [clean_text(t, strip_user=True, keep_hashtag_word=True) for t in raw_sub],
}
abl_rows = []
for name, texts in ablations.items():
    Xa = np.array(texts)
    Xa_tr, _, ya_tr, _ = train_test_split(Xa, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    s = cross_val_score(make_pipe(LinearSVC(C=0.5, random_state=RANDOM_STATE), "union"),
                        Xa_tr, ya_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
    abl_rows.append({"variant": name, "cv_f1_macro": s.mean()})
    print(f"{name:34s} {s.mean():.4f}")
pd.DataFrame(abl_rows).to_csv(f"{OUT}/preprocessing_ablation.csv", index=False)
RESULTS["preprocessing_ablation"] = abl_rows


# ============================================================
# 5. HYPERPARAMETER TUNING on the winning family
# ============================================================
section("5. HYPERPARAMETER TUNING (GridSearchCV)")
grid_pipe = Pipeline([
    ("tfidf", FeatureUnion([
        ("w", TfidfVectorizer(sublinear_tf=True, strip_accents="unicode")),
        ("c", TfidfVectorizer(analyzer="char_wb", sublinear_tf=True))])),
    ("clf", LinearSVC(random_state=RANDOM_STATE))
])
param_grid = {
    "tfidf__w__ngram_range": [(1, 2)],
    "tfidf__w__min_df": [2],
    "tfidf__c__ngram_range": [(3, 5)],
    "tfidf__c__min_df": [3],
    "clf__C": [0.1, 0.25, 0.5, 1.0],
}
gs = GridSearchCV(grid_pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
gs.fit(X_tr, y_tr)
print("Best params:", gs.best_params_)
print(f"Best CV F1-macro: {gs.best_score_:.4f}")
RESULTS["tuning"] = {"best_params": {k: str(v) for k, v in gs.best_params_.items()},
                     "best_cv_f1_macro": float(gs.best_score_)}
best_model = gs.best_estimator_


# ============================================================
# 6. FINAL EVALUATION on held-out test set
# ============================================================
section("6. FINAL EVALUATION (held-out test set, never used in training/tuning)")
y_pred = best_model.predict(X_te)
labels = ["Negative", "Neutral", "Positive"]

acc = accuracy_score(y_te, y_pred)
f1m = f1_score(y_te, y_pred, average="macro")
f1w = f1_score(y_te, y_pred, average="weighted")
p, r, f1c, sup = precision_recall_fscore_support(y_te, y_pred, labels=labels)
print(f"Accuracy:          {acc:.4f}")
print(f"Macro F1:          {f1m:.4f}")
print(f"Weighted F1:       {f1w:.4f}")
print("\n" + classification_report(y_te, y_pred, digits=4))

# ROC-AUC (one-vs-rest) via decision_function
dec = best_model.decision_function(X_te)
y_bin = pd.get_dummies(pd.Series(y_te))[labels].values
try:
    auc = roc_auc_score(y_bin, dec, average="macro", multi_class="ovr")
except Exception:
    auc = float("nan")
print(f"Macro ROC-AUC (OvR): {auc:.4f}")

cm = confusion_matrix(y_te, y_pred, labels=labels)
print("\nConfusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

RESULTS["final_evaluation"] = {
    "accuracy": float(acc), "macro_f1": float(f1m), "weighted_f1": float(f1w),
    "macro_roc_auc_ovr": float(auc),
    "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]),
                        "f1": float(f1c[i]), "support": int(sup[i])}
                  for i, lab in enumerate(labels)},
    "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    "random_baseline_accuracy": 1 / 3,
}

# Confusion matrix figure (counts + row-normalised)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
axes[0].set_title("Confusion matrix (counts)"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
cmn = cm / cm.sum(axis=1, keepdims=True)
sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=False)
axes[1].set_title("Confusion matrix (row-normalised = per-class recall)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrix.png", dpi=160); plt.close()

# Model comparison figure
fig, ax = plt.subplots(figsize=(9, 5))
plot_df = cv_df[cv_df.model != "Majority baseline"].sort_values("cv_f1_macro")
ax.barh(plot_df.model, plot_df.cv_f1_macro, color="#4C72B0")
ax.axvline(1/3, ls="--", color="grey", label="random baseline (0.333)")
ax.set_xlabel("5-fold CV F1-macro"); ax.set_title("Model selection"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/model_comparison.png", dpi=160); plt.close()


# ============================================================
# 7. ERROR ANALYSIS
# ============================================================
section("7. ERROR ANALYSIS")
err = pd.DataFrame({"text": X_te, "true": y_te, "pred": y_pred})
err["correct"] = err.true == err.pred
err["margin"] = np.max(dec, axis=1) - np.sort(dec, axis=1)[:, -2]
err["n_chars"] = err.text.str.len()
err["n_words"] = err.text.str.split().str.len()

# 7a. confusion pairs
pairs = (err[~err.correct].groupby(["true", "pred"]).size()
         .sort_values(ascending=False).reset_index(name="count"))
pairs["pct_of_errors"] = (pairs["count"] / (~err.correct).sum() * 100).round(1)
print("\nMost frequent confusion pairs:")
print(pairs.to_string(index=False))
RESULTS["error_analysis"] = {"confusion_pairs": pairs.to_dict(orient="records"),
                             "n_errors": int((~err.correct).sum()),
                             "n_test": int(len(err))}

# 7b. error rate by length quartile
err["len_q"] = pd.qcut(err.n_words, 4, labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"])
by_len = err.groupby("len_q", observed=True).agg(error_rate=("correct", lambda s: 1 - s.mean()),
                                                 n=("correct", "size")).reset_index()
print("\nError rate by text-length quartile:")
print(by_len.to_string(index=False))
RESULTS["error_analysis"]["error_rate_by_length"] = by_len.astype({"len_q": str}).to_dict(orient="records")

# 7c. confidence (margin) of correct vs incorrect
m_ok = err[err.correct].margin.mean(); m_bad = err[~err.correct].margin.mean()
print(f"\nMean decision margin  correct: {m_ok:.3f}   incorrect: {m_bad:.3f}")
RESULTS["error_analysis"]["mean_margin_correct"] = float(m_ok)
RESULTS["error_analysis"]["mean_margin_incorrect"] = float(m_bad)

# 7d. low-margin band = where the model is genuinely uncertain
low = err.margin < err.margin.quantile(0.25)
print(f"Accuracy in lowest-margin quartile: {err[low].correct.mean():.4f}")
print(f"Accuracy in highest-margin quartile: {err[err.margin > err.margin.quantile(0.75)].correct.mean():.4f}")
RESULTS["error_analysis"]["acc_low_margin_quartile"] = float(err[low].correct.mean())
RESULTS["error_analysis"]["acc_high_margin_quartile"] = float(
    err[err.margin > err.margin.quantile(0.75)].correct.mean())

# 7e. representative misclassified examples (most confident mistakes)
worst = err[~err.correct].sort_values("margin", ascending=False).head(12)
worst[["true", "pred", "margin", "text"]].to_csv(f"{OUT}/misclassified_examples.csv", index=False)
print("\nMost confident mistakes (top 6):")
for _, row in worst.head(6).iterrows():
    print(f"  [true={row.true} -> pred={row.pred} | margin={row.margin:.2f}] {row.text[:110]}")
RESULTS["error_analysis"]["confident_mistakes"] = worst.head(8)[["true", "pred", "margin", "text"]].to_dict(orient="records")

# 7f. most informative features per class
vec = best_model.named_steps["tfidf"]
clf = best_model.named_steps["clf"]
feat_names = np.array(vec.get_feature_names_out())
top_feats = {}
for i, lab in enumerate(clf.classes_):
    coefs = clf.coef_[i]
    top = feat_names[np.argsort(coefs)[-15:]][::-1]
    top_feats[lab] = [t for t in top.tolist()]
    print(f"\nTop features -> {lab}: {', '.join(top_feats[lab][:12])}")
RESULTS["error_analysis"]["top_features"] = top_feats


# ============================================================
# 8. SECONDARY TASK: topic_category (is the second label learnable?)
# ============================================================
section("8. SECONDARY TASK - topic_category classification")
yt = df_model.topic_category.values
Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(X, yt, test_size=0.20, stratify=yt, random_state=RANDOM_STATE)
topic_pipe = make_pipe(LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE), "union")
topic_pipe.fit(Xt_tr, yt_tr)
yt_pred = topic_pipe.predict(Xt_te)
t_acc = accuracy_score(yt_te, yt_pred)
t_f1 = f1_score(yt_te, yt_pred, average="macro")
maj = pd.Series(yt_te).value_counts(normalize=True).iloc[0]
print(f"Topic accuracy: {t_acc:.4f}   macro-F1: {t_f1:.4f}   majority-class baseline: {maj:.4f}")
print(classification_report(yt_te, yt_pred, digits=3, zero_division=0))
RESULTS["secondary_topic_task"] = {"accuracy": float(t_acc), "macro_f1": float(t_f1),
                                   "majority_baseline": float(maj),
                                   "beats_majority": bool(t_acc > maj)}


# ============================================================
# 9. PERSIST ARTIFACTS
# ============================================================
section("9. SAVING ARTIFACTS")
joblib.dump(best_model, f"{OUT}/sentiment_model.pkl")
joblib.dump(topic_pipe, f"{OUT}/topic_model.pkl")
with open(f"{OUT}/evaluation_metrics.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
pd.DataFrame([RESULTS["final_evaluation"]["per_class"][l] | {"class": l} for l in labels]).to_csv(
    f"{OUT}/per_class_metrics.csv", index=False)
print(f"Saved model + metrics to {OUT}")


# ============================================================
# 10. INFERENCE DEMO (the rebuilt semantic layer in use)
# ============================================================
section("10. INFERENCE DEMO")
demo = ["absolutely loved this update, best release in years",
        "the app crashes every single time I try to log in, furious",
        "Reuters reports the summit will be held in Geneva next month"]
for t, pr in zip(demo, best_model.predict([clean_text(x) for x in demo])):
    print(f"  {pr:9s} <- {t}")

print("\nDONE.")
