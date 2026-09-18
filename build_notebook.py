"""Builds Social_Engine_Round2.ipynb — the required NLP model notebook."""
import json, re

SRC = open("/home/claude/r2/nlp_pipeline.py").read()


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def grab(start_marker, end_marker):
    a = SRC.index(start_marker)
    b = SRC.index(end_marker)
    body = SRC[a:b]
    # strip the banner comment blocks
    body = re.sub(r"# ={10,}\n# .*\n# ={10,}\n", "", body)
    return body.strip()


cells = [
    md("# Data Vortex A'26 — Round 2\n"
       "## Rebuilding the Social Engine: Semantic Comprehension Layer\n\n"
       "Three-class sentiment classification over Dataset 2 "
       "(`Labeled_Social_NLP_Training_Data.csv`).\n\n"
       "This notebook is the executable record of the submission. Every number in the Technical "
       "Report and the Evaluation Metrics Report is produced by running these cells — nothing is "
       "hardcoded.\n\n"
       "**Pipeline:** audit → preprocess & deduplicate → stratified split → model selection (5-fold CV) "
       "→ preprocessing ablation → hyperparameter tuning → single held-out evaluation → error "
       "analysis → secondary topic task."),
    md("## 0. Imports and configuration"),
    code(grab("import os, re, json, html, warnings", 'DATA = "/mnt/user-data/uploads')
         + '\n\nDATA = "Labeled_Social_NLP_Training_Data.csv"   # path to Dataset 2\n'
           'OUT = "outputs"\nos.makedirs(OUT, exist_ok=True)\nRESULTS = {}\n\n'
           'def section(t):\n    print("\\n" + "=" * 70); print(t); print("=" * 70)'),
    md("## 1. Data audit\n\n"
       "Before any modelling: how balanced are the labels, are there missing values, and — the "
       "question that shaped this entire pipeline — are there duplicate texts?"),
    code(grab('df = pd.read_csv(DATA)', '# ============================================================\n# 2. PREPROCESSING')),
    md("**Key finding:** 1,100 rows repeat a `post_text` seen elsewhere, all label-consistent. "
       "If the train/test split happens before these are removed, the same sentence appears on both "
       "sides and the model is partly graded on memorisation. Section 5b quantifies this."),
    md("## 2. Preprocessing\n\n"
       "Deterministic normalisation, each step flagged so it can be ablated. Casing, punctuation and "
       "negation words are deliberately **kept** — they carry sentiment in short text. No stopword "
       "removal, no stemming; bigrams are used instead so `not good` survives as a feature."),
    code(grab("URL_RE = re.compile", "# --- Deduplicate BEFORE splitting")),
    md("### Deduplicate *before* splitting — the leakage control"),
    code(grab("before = len(df)", "# ============================================================\n# 3. MODEL SELECTION")),
    md("## 3. Model selection — 5-fold stratified CV on the training split only\n\n"
       "Ten candidates across three families (probabilistic, linear-margin, ensemble) under identical "
       "CV. The test set is not touched until Section 6."),
    code(grab("word_tfidf = TfidfVectorizer", "# ============================================================\n# 4. PREPROCESSING ABLATION")),
    md("Linear models beat the ensemble (sparse high-dimensional TF-IDF favours a linear boundary over "
       "axis-aligned splits), and **character n-grams outperform word n-grams alone** — social text is "
       "noisy, and `char_wb` 3–5 grams capture sub-word cues (`lov`, `n't`, `:)`) that survive "
       "misspellings and hashtags. The final model unions both feature spaces."),
    md("## 4. Preprocessing ablation\n\n"
       "Each cleaning step measured rather than assumed helpful."),
    code(grab("raw_sub = df_model.post_text.values", "# ============================================================\n# 5. HYPERPARAMETER TUNING")),
    md("Honest reading: the cleaning steps are **performance-neutral** — the spread is inside "
       "fold-to-fold noise, because the character n-grams are already robust to surface noise. "
       "The cleaning is retained for interpretability and production robustness, not for an accuracy "
       "claim."),
    md("## 5. Hyperparameter tuning"),
    code(grab("grid_pipe = Pipeline", "# ============================================================\n# 6. FINAL EVALUATION")),
    md("## 5b. Leakage experiment\n\n"
       "Same tuned configuration, split **before** vs **after** deduplication — measuring the cost of "
       "getting this wrong."),
    code('''from sklearn.pipeline import Pipeline as _P

def tuned():
    return Pipeline([("tfidf", FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True))])),
        ("clf", LinearSVC(C=0.1, random_state=RANDOM_STATE))])

# A) naive: split the full file, duplicates can straddle the split
Xn, yn = df.clean_text.values, df.sentiment_label.values
Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(Xn, yn, test_size=0.2, stratify=yn, random_state=RANDOM_STATE)
mask = pd.Series(Xn_te).isin(set(Xn_tr)).values
mn = tuned().fit(Xn_tr, yn_tr); pn = mn.predict(Xn_te)
print(f"NAIVE  acc={accuracy_score(yn_te, pn):.4f}  leaked rows={mask.sum()} ({100*mask.mean():.1f}%)")
print(f"       acc on leaked rows  = {accuracy_score(yn_te[mask], pn[mask]):.4f}")
print(f"       acc on unseen rows  = {accuracy_score(yn_te[~mask], pn[~mask]):.4f}")
print(f"DEDUP  acc={accuracy_score(y_te, best_model.predict(X_te)):.4f}   <- reported figure")'''),
    md("~17% of the naive test set was memorised, scoring ~93% on those rows against ~60% on unseen "
       "rows. The blended number overstates real performance by ~4.3 accuracy points. **All reported "
       "metrics use the deduplicated protocol.**"),
    md("## 6. Final evaluation — held-out test set, used exactly once"),
    code(grab("y_pred = best_model.predict(X_te)", "# Confusion matrix figure")),
    md("## 7. Confusion matrix"),
    code(grab("fig, axes = plt.subplots(1, 2", "# ============================================================\n# 7. ERROR ANALYSIS")),
    md("The off-diagonal mass is not uniform: the Negative↔Positive cells are the two smallest, while "
       "every cell involving Neutral is large. The model rarely confuses polar opposites — its errors "
       "are polarity-*strength* errors, not polarity-*direction* errors."),
    md("## 8. Error analysis\n\n"
       "Four decompositions: which confusions dominate, whether confidence is informative, whether "
       "length explains errors, and what the confident mistakes look like."),
    code(grab("err = pd.DataFrame({", "# ============================================================\n# 8. SECONDARY TASK")),
    md("**What the errors mean.** ~80% of errors involve Neutral. Neutral is the residual category — "
       "its lexical signature is the *absence* of strong cues, which is the hardest thing for a "
       "bag-of-features model to represent. The confident mistakes cluster into three patterns: "
       "charged vocabulary inside neutral reporting; sentiment implied by world knowledge rather than "
       "stated; and genuine annotation ambiguity that caps achievable accuracy for any model.\n\n"
       "**Calibration is useful.** Accuracy rises from ~46% to ~84% between the least and most "
       "confident quartiles, so thresholding on margin lets the Social Engine auto-label the confident "
       "majority and route the uncertain quartile to human review."),
    md("## 9. Secondary task — topic_category\n\n"
       "The second label is heavily skewed (86% one class), so accuracy alone would mislead."),
    code(grab("yt = df_model.topic_category.values", "# ============================================================\n# 9. PERSIST ARTIFACTS")),
    md("High accuracy but materially lower macro-F1: the two rare classes (Account_Security, "
       "Feature_Feedback) are under-recovered. Reporting accuracy alone here would overstate the "
       "model."),
    md("## 10. Save artifacts"),
    code(grab("joblib.dump(best_model", "# ============================================================\n# 10. INFERENCE DEMO")),
    md("## 11. Inference demo — the rebuilt semantic layer in use"),
    code(grab('demo = ["absolutely loved', 'print("\\nDONE.")')),
    md("---\n### Limitations\n\n"
       "- **No pretrained contextual model.** A fine-tuned transformer would typically add 10–20 "
       "points on short social text by resolving exactly the error patterns above; the execution "
       "environment had no access to pretrained weights, so every model here is trained from scratch "
       "on 6,320 rows. Read the reported figures as a *from-scratch* ceiling.\n"
       "- **Annotation ceiling.** Several confident mistakes are arguably mislabelled; true human "
       "agreement on this dataset is unmeasured.\n"
       "- **Single test set.** With 1,580 test rows the 95% CI on accuracy is roughly ±2.4 points — "
       "smaller model differences are not real differences."),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
    if c["cell_type"] == "code":
        c.setdefault("outputs", [])

path = "/mnt/user-data/outputs/Social_Engine_Round2.ipynb"
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path, len(cells), "cells")
