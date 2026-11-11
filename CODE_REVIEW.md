# Code Review — Telecom Customer Churn Prediction

Reviewer-facing deep-dive on the correctness findings in this repo (Telco churn;
two CSV halves joined positionally; `TotalCharges` blank-string coercion;
GradientBoosting; imbalanced, decision threshold 0.4; FastAPI `/predict`) and the
fixes that landed. Each finding gives **Where**, **Root cause**, **Fix**
(before/after from the actual commit), and the **regression test** that locks it.
Commits referenced: `fd9b2f5`, `5b7d933`, `756a1f6`.

The ML core (split-before-fit, train-only transforms, held-out metrics,
train/serve coercion parity) was audited and found correct — see "Verified
correct, not changed" at the end.

---

## 1. Model artifact loaded twice during evaluation

**Where:** `src/churn/components/model_evaluation.py` — `evaluate()` and
`_log_to_mlflow()` (commit `fd9b2f5`).

**Root cause:** `evaluate()` deserialized the serving pipeline to score the test
set, then `_log_to_mlflow()` called `load_object(...)` again to log the same
artifact. Two disk reads of the same joblib per run.

**Fix:** load once, pass the object into `_log_to_mlflow(model)`.

```python
# before
        self._log_to_mlflow(metrics)
        return metrics

    def _log_to_mlflow(self, metrics: dict) -> None:
        ...
            mlflow.log_metrics(metrics)
            try:
                model = load_object(self.config.model_file)   # second disk read
                mlflow.sklearn.log_model(model, name="model")

# after
        self._log_to_mlflow(metrics, model)
        return metrics

    def _log_to_mlflow(self, metrics: dict, model) -> None:
        ...
            # MLflow metrics must be numeric; drop any None (e.g. undefined AUC).
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            try:
                mlflow.sklearn.log_model(model, name="model")
```

**Regression test:** the end-to-end smoke test runs evaluation and asserts
`metrics.json` is produced and sane. Pure de-duplication; metric values unchanged.

---

## 2. Single-class test split crashed ROC-AUC

**Where:** `src/churn/components/model_evaluation.py` — `evaluate()`
(commit `fd9b2f5`).

**Root cause:** `roc_auc_score(y, proba)` raises when `y` has a single class.
Churn is imbalanced; a small or degenerate test split could trip this and crash
the whole stage.

**Fix:** guard on `y.nunique() < 2`, record `roc_auc: null`, and filter `None`
before `mlflow.log_metrics`.

```python
# before
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
        }

# after
            "f1": float(f1_score(y, preds, zero_division=0)),
        }
        # ROC-AUC needs both classes present; record null (not a crash) if the
        # test labels are single-class.
        if y.nunique() < 2:
            logger.warning("Only one class in y_true; ROC-AUC is undefined.")
            metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
```

**Regression test:** the end-to-end test asserts `metrics["roc_auc"] > 0.75` on
the real two-class data (Telco is well-separated), locking the happy path; the
guard is the defensive branch for degenerate splits.

---

## 3. O(n²) duplicate-column check in the ingestion join guard

**Where:** `src/churn/components/data_ingestion.py` — `_check_join()`
(commit `5b7d933`).

**Root cause:** the two raw halves are joined **positionally** (file 2 has no
`customerID`, so there is no join key — correctness rests on row order). The
guard that detects an overlapping/misaligned join scanned for duplicate column
names with `list(combined.columns).count(c)` inside a set comprehension — a
`.count()` per column, i.e. O(n²) in the column count. Correct, but needlessly
quadratic.

**Fix:** a single `collections.Counter` pass. The raised error is byte-for-byte
unchanged.

```python
# before
        dupes = sorted({c for c in combined.columns if list(combined.columns).count(c) > 1})

# after
        dupes = sorted(c for c, n in Counter(combined.columns).items() if n > 1)
```

**Regression test:** `tests/test_data_ingestion.py` —
`test_check_join_rejects_duplicate_columns` (and the well-formed /
missing-column cases) exercise `_check_join` directly with synthetic pandas
frames, no sklearn or disk needed.

---

## 4. Positive-class probability hard-coded to column index 1

**Where:** `src/churn/pipeline/prediction.py` — `PredictionPipeline`
(commit `756a1f6`).

**Root cause:** `predict()` read the churn probability as
`model.predict_proba(df)[:, 1]`. That index-1 assumption is only valid because
the target happens to encode `Yes → 1` and sklearn orders `predict_proba` columns
by `model.classes_ = [0, 1]`. If the model were ever retrained with a different
label encoding or class order, column 1 would silently become the *wrong* class,
mislabeling every served probability — a quiet correctness failure with no error.

**Fix:** resolve the positive-class column from `model.classes_` once in
`__init__` (index of the known positive label `1`), and index `predict_proba`
with it. Current Telco behavior is identical (the lookup resolves to 1); it is
now robust to a reordered/retargeted model.

```python
# before
        proba = float(self.model.predict_proba(df)[:, 1][0])

# after  (in __init__)
    POSITIVE_LABEL = 1  # Yes -> 1 (churn), set in transformation
    ...
        self._pos_idx = list(self.model.classes_).index(self.POSITIVE_LABEL)
    # in predict()
        # Look up the positive-class column instead of assuming index 1, so a
        # reordered/retargeted model still reads the right probability.
        proba = float(self.model.predict_proba(df)[0, self._pos_idx])
```

**Trade-off / assumption:** the lookup still assumes the encoded positive label
is the integer `1`. That is the contract established in
`data_transformation` (`Yes → 1`); if that encoding ever changes, `POSITIVE_LABEL`
is the single constant to update, and `.index()` would raise clearly if `1` were
absent rather than returning a wrong column.

**Regression test:** `tests/test_prediction.py` covers the missing-feature guard
(below); the positive-class lookup is exercised by the end-to-end test, which
scores a known-churny payload and asserts a valid probability in `[0, 1]`.

---

## 5. `predict()` raised a bare `KeyError` on a missing feature

**Where:** `src/churn/pipeline/prediction.py` — `predict()` (commit `756a1f6`).

**Root cause:** same shape as the loan repo — a dict missing a feature key hit a
`KeyError` inside the `ColumnTransformer`, surfacing as an opaque 500.

**Fix:** validate the expected columns up front (clear `ValueError` naming the
missing fields), reindex to the training feature order, then run the existing
`TotalCharges` numeric coercion.

```python
# before
        df = pd.DataFrame([features])
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

# after
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing required feature field(s): {missing}")
        # Reindex to the training feature order (extra keys ignored).
        df = pd.DataFrame([features]).reindex(columns=list(self.feature_columns))
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
```

**Regression test:** `tests/test_prediction.py` —
`test_missing_feature_raises_clear_error` / `test_missing_feature_names_the_field`,
bypassing `__init__` so no trained model is required:

```python
def _pipeline_with_columns(columns):
    p = PredictionPipeline.__new__(PredictionPipeline)
    p.feature_columns = tuple(columns)
    p.numeric_columns = ()
    p.threshold = 0.4
    p.model = None  # must not be reached when a field is missing
    return p
```

---

## Verified correct, not changed (the ML core)

- **Split before fit, fit on train only.** `transform()` splits first, then fits
  the `ColumnTransformer` on the training split only — **no leakage** from the
  median imputer, scaler, or one-hot encoder.
- **Stratified, reproducible split.** `stratify=df[target]` with a fixed
  `random_state` preserves the ~27% positive rate and is deterministic.
- **Train/serve coercion parity.** `TotalCharges` blank strings are coerced to
  numeric both in `transform()` and in `predict()` (now after the reindex), so the
  served feature matches what the model was trained on.
- **Held-out metrics, probabilities for AUC.** Evaluation reads `test.csv` and
  computes ROC-AUC on probabilities at the configured 0.4 threshold — the same
  threshold `/predict` uses, so served decisions match reported metrics.
- **Positional join is guarded.** The halves have no shared key; `_check_join`
  rejects duplicate columns and missing expected columns, hardening the
  same-row-order contract (finding #3 only made that guard faster).
