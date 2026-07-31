"""
================================================================================
 train_model.py  —  PRODUCTION ML TRAINING PIPELINE (10 explicit stages)
================================================================================

Trains a REAL machine-learning model on the 1000-sample synthetic dataset to
classify surface defects (good / scratch / crack / dent / corrosion) AND to
provide a defect-vs-good probability used by the decision-fusion layer.

The pipeline is deliberately staged and instrumented so every step is visible
and reproducible:

  STAGE 1  Load dataset manifest + label CSVs (train / test).
  STAGE 2  Extract the 51-D feature vector for every image (with disk caching).
  STAGE 3  Assemble feature matrices X_train / X_test and label vectors.
  STAGE 4  Build a preprocessing + model Pipeline (impute -> scale -> classifier).
  STAGE 5  Stratified k-fold cross-validation on the training set.
  STAGE 6  Randomised hyper-parameter search (RandomForest + GradientBoosting +
           a soft-voting ensemble) — pick the best by CV macro-F1.
  STAGE 7  Refit the best estimator on all training data.
  STAGE 8  Evaluate on the held-out TEST set (accuracy, macro-F1, per-class,
           confusion matrix, binary defect-detection metrics, ROC-AUC).
  STAGE 9  Calibrate probabilities (isotonic/sigmoid) for trustworthy fusion.
  STAGE 10 Persist model bundle (+ metadata, feature names, class list) and
           write metrics JSON + confusion-matrix & importance plots.

Artefacts
---------
    models/defect_model.joblib          the deployable model bundle
    models/training_metrics.json        full metrics report
    models/confusion_matrix.png
    models/feature_importance.png
    models/feature_cache.npz            cached feature matrices
    logs/training.log                   run log

Run
---
    python src/train_model.py                 # full pipeline
    python src/train_model.py --fast          # smaller search (quicker)
    python src/train_model.py --rebuild-cache # force feature re-extraction
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     RandomizedSearchCV)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score,
                             precision_score, recall_score)

from feature_engine import FeatureEngine, FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
LOGS = os.path.join(ROOT, "logs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

CLASSES = ["good", "scratch", "crack", "dent", "corrosion"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
def get_logger():
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(LOGS, "training.log"), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


LOG = get_logger()


def banner(txt):
    LOG.info("=" * 72)
    LOG.info(txt)
    LOG.info("=" * 72)


# =========================================================================== #
#  STAGE 1 — load dataset
# =========================================================================== #
def stage1_load_manifest():
    banner("STAGE 1/10 — Loading dataset manifest and label CSVs")
    manifest_path = os.path.join(DATA, "dataset_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "Dataset not found. Run: python src/dataset_generator.py --n 1000")
    manifest = json.load(open(manifest_path))
    train_df = pd.read_csv(os.path.join(DATA, "train", "labels.csv"))
    test_df = pd.read_csv(os.path.join(DATA, "test", "labels.csv"))
    LOG.info(f"train={len(train_df)}  test={len(test_df)}  "
             f"classes={manifest['classes']}")
    return manifest, train_df, test_df


# =========================================================================== #
#  STAGE 2 — feature extraction (with caching)
# =========================================================================== #
def _extract_matrix(df, engine, split_name):
    X = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        path = os.path.join(DATA, row.filepath)
        feats, _ = engine.extract(path)
        X[i] = engine.to_vector(feats)
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            LOG.info(f"  [{split_name}] {i + 1}/{len(df)} "
                     f"({rate:.1f} img/s)")
    return X


def stage2_features(train_df, test_df, rebuild=False):
    banner("STAGE 2/10 — Extracting 51-D feature vectors (with caching)")
    cache = os.path.join(MODELS, "feature_cache.npz")
    if os.path.isfile(cache) and not rebuild:
        d = np.load(cache, allow_pickle=True)
        if (len(d["X_train"]) == len(train_df) and
                len(d["X_test"]) == len(test_df)):
            LOG.info("Loaded cached feature matrices.")
            return d["X_train"], d["X_test"]
        LOG.info("Cache size mismatch — re-extracting.")

    engine = FeatureEngine()
    X_train = _extract_matrix(train_df, engine, "train")
    X_test = _extract_matrix(test_df, engine, "test")
    np.savez_compressed(cache, X_train=X_train, X_test=X_test)
    LOG.info(f"Cached features -> {cache}")
    return X_train, X_test


# =========================================================================== #
#  STAGE 3 — assemble matrices / labels
# =========================================================================== #
def stage3_assemble(train_df, test_df, X_train, X_test):
    banner("STAGE 3/10 — Assembling label vectors")
    y_train = train_df["class"].map(CLASS_TO_IDX).to_numpy()
    y_test = test_df["class"].map(CLASS_TO_IDX).to_numpy()
    LOG.info(f"X_train={X_train.shape}  X_test={X_test.shape}")
    LOG.info("class balance (train): " +
             ", ".join(f"{c}={int((y_train == i).sum())}"
                       for i, c in enumerate(CLASSES)))
    return y_train, y_test


# =========================================================================== #
#  STAGE 4 — build candidate pipelines
# =========================================================================== #
def _pipe(clf):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def stage4_build_candidates():
    banner("STAGE 4/10 — Building candidate model pipelines")
    rf = _pipe(RandomForestClassifier(random_state=42, n_jobs=-1,
                                      class_weight="balanced_subsample"))
    gb = _pipe(GradientBoostingClassifier(random_state=42))
    LOG.info("Candidates: RandomForest, GradientBoosting, (+ Voting ensemble)")
    return rf, gb


# =========================================================================== #
#  STAGE 5 — cross-validation baseline
# =========================================================================== #
def stage5_cv(rf, gb, X, y):
    banner("STAGE 5/10 — Stratified 5-fold cross-validation (baseline)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, est in (("RandomForest", rf), ("GradientBoosting", gb)):
        scores = cross_val_score(est, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
        LOG.info(f"  {name:18s} CV macro-F1 = "
                 f"{scores.mean():.4f} ± {scores.std():.4f}")


# =========================================================================== #
#  STAGE 6 — hyper-parameter search
# =========================================================================== #
def stage6_search(X, y, fast=False):
    banner("STAGE 6/10 — Randomised hyper-parameter search")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_iter = 8 if fast else 25

    rf_space = {
        "clf__n_estimators": [200, 300, 400, 600],
        "clf__max_depth": [None, 12, 18, 26],
        "clf__min_samples_leaf": [1, 2, 3],
        "clf__max_features": ["sqrt", "log2", 0.5],
    }
    rf_search = RandomizedSearchCV(
        _pipe(RandomForestClassifier(random_state=42, n_jobs=-1,
                                     class_weight="balanced_subsample")),
        rf_space, n_iter=n_iter, cv=skf, scoring="f1_macro",
        random_state=42, n_jobs=-1, verbose=0)
    LOG.info("  searching RandomForest …")
    rf_search.fit(X, y)
    LOG.info(f"  best RF macro-F1={rf_search.best_score_:.4f} "
             f"params={rf_search.best_params_}")

    gb_space = {
        "clf__n_estimators": [150, 250, 350],
        "clf__learning_rate": [0.03, 0.06, 0.1],
        "clf__max_depth": [2, 3, 4],
        "clf__subsample": [0.8, 1.0],
    }
    gb_search = RandomizedSearchCV(
        _pipe(GradientBoostingClassifier(random_state=42)),
        gb_space, n_iter=max(5, n_iter // 2), cv=skf, scoring="f1_macro",
        random_state=42, n_jobs=-1, verbose=0)
    LOG.info("  searching GradientBoosting …")
    gb_search.fit(X, y)
    LOG.info(f"  best GB macro-F1={gb_search.best_score_:.4f} "
             f"params={gb_search.best_params_}")

    # Build a soft-voting ensemble of the two tuned models
    ensemble = VotingClassifier(
        estimators=[("rf", rf_search.best_estimator_),
                    ("gb", gb_search.best_estimator_)],
        voting="soft", weights=[2, 1], n_jobs=-1)
    ens_scores = cross_val_score(ensemble, X, y, cv=skf,
                                 scoring="f1_macro", n_jobs=-1)
    LOG.info(f"  Voting ensemble CV macro-F1 = "
             f"{ens_scores.mean():.4f} ± {ens_scores.std():.4f}")

    candidates = {
        "RandomForest": (rf_search.best_estimator_, rf_search.best_score_),
        "GradientBoosting": (gb_search.best_estimator_, gb_search.best_score_),
        "VotingEnsemble": (ensemble, ens_scores.mean()),
    }
    best_name = max(candidates, key=lambda k: candidates[k][1])
    LOG.info(f"  -> selected: {best_name} "
             f"(CV macro-F1={candidates[best_name][1]:.4f})")
    return best_name, candidates[best_name][0], {
        k: round(v[1], 4) for k, v in candidates.items()}


# =========================================================================== #
#  STAGE 7 — refit
# =========================================================================== #
def stage7_refit(best_est, X, y):
    banner("STAGE 7/10 — Refitting best estimator on full training set")
    best_est.fit(X, y)
    LOG.info("  refit complete.")
    return best_est


# =========================================================================== #
#  STAGE 8 — test-set evaluation
# =========================================================================== #
def stage8_evaluate(model, X_test, y_test):
    banner("STAGE 8/10 — Held-out TEST-set evaluation")
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    LOG.info(f"  TEST accuracy = {acc:.4f}")
    LOG.info(f"  TEST macro-F1 = {macro_f1:.4f}")

    report = classification_report(y_test, y_pred, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # binary defect-vs-good metrics
    good_idx = CLASS_TO_IDX["good"]
    y_bin_true = (y_test != good_idx).astype(int)
    y_bin_pred = (y_pred != good_idx).astype(int)
    p_defect = 1.0 - proba[:, good_idx]
    bin_metrics = {
        "accuracy": round(accuracy_score(y_bin_true, y_bin_pred), 4),
        "precision": round(precision_score(y_bin_true, y_bin_pred, zero_division=0), 4),
        "recall": round(recall_score(y_bin_true, y_bin_pred, zero_division=0), 4),
        "f1": round(f1_score(y_bin_true, y_bin_pred, zero_division=0), 4),
    }
    try:
        bin_metrics["roc_auc"] = round(roc_auc_score(y_bin_true, p_defect), 4)
    except ValueError:
        bin_metrics["roc_auc"] = None

    LOG.info(f"  DEFECT-detection  acc={bin_metrics['accuracy']} "
             f"recall={bin_metrics['recall']} f1={bin_metrics['f1']} "
             f"auc={bin_metrics['roc_auc']}")
    for c in CLASSES:
        r = report[c]
        LOG.info(f"    {c:10s} P={r['precision']:.3f} R={r['recall']:.3f} "
                 f"F1={r['f1-score']:.3f} n={int(r['support'])}")

    metrics = {
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(macro_f1, 4),
        "per_class": {c: {k: round(report[c][k], 4)
                          for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASSES},
        "confusion_matrix": cm.tolist(),
        "binary_defect_detection": bin_metrics,
    }
    return metrics, cm


# =========================================================================== #
#  STAGE 9 — probability calibration
# =========================================================================== #
def stage9_calibrate(model, X_train, y_train):
    banner("STAGE 9/10 — Calibrating probabilities for trustworthy fusion")
    method = "isotonic" if len(y_train) >= 300 else "sigmoid"
    cal = CalibratedClassifierCV(model, method=method, cv=3)
    cal.fit(X_train, y_train)
    LOG.info(f"  calibrated with method='{method}'")
    return cal


# =========================================================================== #
#  STAGE 10 — persist + plots
# =========================================================================== #
def stage10_persist(model, calibrated, best_name, cv_scores, metrics, cm):
    banner("STAGE 10/10 — Persisting model bundle, metrics and plots")

    bundle = {
        "model": calibrated,          # calibrated estimator (predict_proba)
        "raw_model": model,           # uncalibrated (feature importance access)
        "classes": CLASSES,
        "feature_names": FEATURE_NAMES,
        "selected_model": best_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_pipeline": True,
        "version": "1.0.0",
    }
    model_path = os.path.join(MODELS, "defect_model.joblib")
    joblib.dump(bundle, model_path, compress=3)
    LOG.info(f"  saved model -> {model_path}")

    full_metrics = {
        "selected_model": best_name,
        "cv_scores": cv_scores,
        **metrics,
        "trained_at": bundle["trained_at"],
        "n_features": len(FEATURE_NAMES),
    }
    with open(os.path.join(MODELS, "training_metrics.json"), "w") as f:
        json.dump(full_metrics, f, indent=2)
    LOG.info("  saved metrics -> models/training_metrics.json")

    _plot_confusion(cm)
    _plot_importance(model)
    LOG.info("  saved plots -> models/confusion_matrix.png, feature_importance.png")
    return full_metrics


def _plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix (row-normalised)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS, "confusion_matrix.png"), dpi=130)
    plt.close(fig)


def _plot_importance(model):
    # dig the RandomForest out of a Pipeline / VotingClassifier
    clf = None
    est = model
    if hasattr(est, "named_estimators_"):
        est = est.named_estimators_.get("rf", None)
    if est is not None and hasattr(est, "named_steps"):
        clf = est.named_steps.get("clf")
    if clf is None or not hasattr(clf, "feature_importances_"):
        return
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh([FEATURE_NAMES[i] for i in order][::-1],
            imp[order][::-1], color="#2a7de1")
    ax.set_title("Top-20 Feature Importances (RandomForest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS, "feature_importance.png"), dpi=130)
    plt.close(fig)


# =========================================================================== #
#  ORCHESTRATION
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller search")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    manifest, train_df, test_df = stage1_load_manifest()
    X_train, X_test = stage2_features(train_df, test_df, rebuild=args.rebuild_cache)
    y_train, y_test = stage3_assemble(train_df, test_df, X_train, X_test)
    rf, gb = stage4_build_candidates()
    stage5_cv(rf, gb, X_train, y_train)
    best_name, best_est, cv_scores = stage6_search(X_train, y_train, fast=args.fast)
    model = stage7_refit(best_est, X_train, y_train)
    metrics, cm = stage8_evaluate(model, X_test, y_test)
    calibrated = stage9_calibrate(model, X_train, y_train)
    # re-evaluate the CALIBRATED model too (final deployed metrics)
    cal_pred = calibrated.predict(X_test)
    metrics["calibrated_test_accuracy"] = round(accuracy_score(y_test, cal_pred), 4)
    metrics["calibrated_test_macro_f1"] = round(
        f1_score(y_test, cal_pred, average="macro"), 4)
    LOG.info(f"  CALIBRATED test accuracy = {metrics['calibrated_test_accuracy']}")
    full = stage10_persist(model, calibrated, best_name, cv_scores, metrics, cm)

    banner("TRAINING COMPLETE")
    LOG.info(f"  selected model : {full['selected_model']}")
    LOG.info(f"  TEST accuracy  : {full['test_accuracy']}")
    LOG.info(f"  TEST macro-F1  : {full['test_macro_f1']}")
    LOG.info(f"  defect recall  : {full['binary_defect_detection']['recall']}")
    LOG.info(f"  total time     : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
