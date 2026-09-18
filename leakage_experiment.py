"""Leakage experiment: same tuned model, split BEFORE vs AFTER deduplication."""
import json, html, re
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.insert(0, "/home/claude/r2")
from nlp_pipeline_clean import clean_text  # shared cleaning fn

RS = 42
df = pd.read_csv("/mnt/user-data/uploads/Labeled_Social_NLP_Training_Data.csv")
df["clean_text"] = df.post_text.apply(clean_text)


def tuned():
    return Pipeline([("tfidf", FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True))])),
        ("clf", LinearSVC(C=0.1, random_state=RS))])


out = {}

# A) naive: split the full 9,000 rows (duplicates can straddle the split)
X, y = df.clean_text.values, df.sentiment_label.values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
n_leak = int(pd.Series(Xte).isin(set(Xtr)).sum())
m = tuned().fit(Xtr, ytr); p = m.predict(Xte)
out["naive_with_duplicates"] = {"test_n": len(Xte), "leaked_test_rows": n_leak,
                                "leaked_pct": round(100 * n_leak / len(Xte), 2),
                                "accuracy": float(accuracy_score(yte, p)),
                                "macro_f1": float(f1_score(yte, p, average="macro"))}
# accuracy on the leaked subset vs the rest
mask = pd.Series(Xte).isin(set(Xtr)).values
out["naive_with_duplicates"]["accuracy_on_leaked_rows"] = float(accuracy_score(yte[mask], p[mask]))
out["naive_with_duplicates"]["accuracy_on_unseen_rows"] = float(accuracy_score(yte[~mask], p[~mask]))

# B) correct: deduplicate first
dd = df.drop_duplicates(subset="post_text", keep="first")
X2, y2 = dd.clean_text.values, dd.sentiment_label.values
Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.2, stratify=y2, random_state=RS)
m2 = tuned().fit(Xtr2, ytr2); p2 = m2.predict(Xte2)
out["deduplicated"] = {"test_n": len(Xte2), "leaked_test_rows": 0,
                       "accuracy": float(accuracy_score(yte2, p2)),
                       "macro_f1": float(f1_score(yte2, p2, average="macro"))}

out["inflation_accuracy_points"] = round(
    100 * (out["naive_with_duplicates"]["accuracy"] - out["deduplicated"]["accuracy"]), 2)

print(json.dumps(out, indent=2))
json.dump(out, open("/home/claude/r2/outputs/leakage_experiment.json", "w"), indent=2)
