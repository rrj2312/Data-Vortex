# Data Vortex A'26 — Round 2: Rebuilding the Social Engine (Semantic Layer)

Three-class sentiment classification over Dataset 2 (`Labeled_Social_NLP_Training_Data.csv`),
plus a secondary `topic_category` classifier.

## Headline result

| Metric | Value | Baseline |
|---|---|---|
| Accuracy | **0.6158** | 0.3333 random / 0.3478 majority |
| Macro F1 | **0.6169** | 0.1720 majority |
| Macro ROC-AUC (OvR) | 0.7901 | 0.5000 random |

Final model: `TF-IDF (word 1–2 gram, min_df=2 + char_wb 3–5 gram, min_df=3) → LinearSVC(C=0.1)`,
selected by 5-fold stratified CV over 10 candidates and tuned by GridSearchCV.

## The number that matters most

The dataset contains **1,100 exact duplicate texts**. Splitting before removing them puts the same
sentence in train and test:

| Split strategy | Leaked test rows | Accuracy |
|---|---|---|
| Naive (split before dedup) | 313 / 1,800 (17.4%) | 0.6589 |
| Correct (dedup before split) | 0 | **0.6158** |

On the memorised rows the naive model scores 0.9297; on genuinely unseen rows, 0.6019. The blended
figure overstates real performance by **4.31 accuracy points**. Every metric reported in this
submission uses the deduplicated protocol.

## Files

```
nlp_pipeline.py            full pipeline: audit → preprocess → CV → ablation → tuning → eval → error analysis
leakage_experiment.py      reproduces the leakage table above
build_reports.py           regenerates both submission PDFs from the metrics JSON
build_notebook.py          regenerates the submission notebook
Social_Engine_Round2.ipynb required notebook (same pipeline, annotated)
requirements.txt           pinned dependencies
outputs/
  sentiment_model.pkl        trained sentiment classifier (joblib)
  topic_model.pkl            trained topic classifier (joblib)
  evaluation_metrics.json    every metric in the reports, machine-readable
  leakage_experiment.json    leakage measurement
  model_selection_cv.csv     10-model CV comparison
  preprocessing_ablation.csv cleaning-step ablation
  per_class_metrics.csv       precision/recall/F1 per class
  misclassified_examples.csv  highest-confidence mistakes, for error analysis
  confusion_matrix.png        counts + row-normalised
  model_comparison.png        CV macro-F1 across candidates
```

## Reproduce

```bash
pip install -r requirements.txt
python nlp_pipeline.py          # writes outputs/*, ~10 min single-core
python leakage_experiment.py
python build_reports.py
```

No metric in either PDF is hardcoded — both are generated from `outputs/evaluation_metrics.json`
and `outputs/leakage_experiment.json` at build time.

## Inference

```python
import joblib
from nlp_pipeline_clean import clean_text
model = joblib.load("outputs/sentiment_model.pkl")
model.predict([clean_text("absolutely loved this update")])   # -> ['Positive']
```

## Known limitations

- **No pretrained contextual model.** A fine-tuned transformer would typically add 10–20 points on
  short social text; the execution environment had no access to pretrained weights. All models here
  are trained from scratch on the 6,320 supplied training rows.
- **Annotation ceiling.** Inspection of confident mistakes shows several arguably-mislabelled
  Neutral/Negative boundary cases. True human-agreement ceiling is unmeasured.
- **Single held-out test set.** 1,580 test rows → ~±2.4 point 95% CI on accuracy. Model differences
  smaller than that are not treated as real.
