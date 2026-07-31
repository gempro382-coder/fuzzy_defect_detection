"""
================================================================================
 evaluate_system.py  —  END-TO-END EVALUATION OF THE FUSED SYSTEM
================================================================================

Evaluates, on the held-out TEST set, THREE decision paths side by side:

    1. ML-only            (trained model)
    2. Fuzzy-only         (knowledge-based rules)
    3. FUSED (ML + Fuzzy) (the deployed production decision)

Reports for each path:
    * defect-detection accuracy / precision / recall / F1
    * borderline-defect recall (the hard cases)
    * (for ML & fused) 5-class / type accuracy
and writes:
    results/system_evaluation.json
    results/system_comparison.png
    results/fused_confusion_matrix.png

This is the evidence that fusing the two systems is at least as good as — and
on borderline cases better than — either alone.
================================================================================
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_engine import FeatureEngine
from fusion_engine import FusionEngine, CLASSES, DEFECT_CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def _metrics(tp, tn, fp, fn):
    tot = tp + tn + fp + fn
    acc = (tp + tn) / tot if tot else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    test_df = pd.read_csv(os.path.join(DATA, "test", "labels.csv"))
    engine = FeatureEngine()
    fusion = FusionEngine()

    # counters for each path
    paths = ["ml", "fuzzy", "fused"]
    det = {p: dict(tp=0, tn=0, fp=0, fn=0) for p in paths}
    border = {p: dict(hit=0, tot=0) for p in paths}
    type_correct = {p: 0 for p in ["ml", "fused"]}
    type_total = 0
    # fused 5-class confusion
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    cidx = {c: i for i, c in enumerate(CLASSES)}
    n_review = 0
    n_disagree = 0

    for k in range(len(test_df)):
        row = test_df.iloc[k]
        path = os.path.join(DATA, row["filepath"])
        feats, _ = engine.extract(path)
        rep = fusion.decide(feats)

        true_defect = int(row["label_defective"])
        true_class = str(row["class"])
        band = str(row["severity_band"])

        # --- ML path ---
        ml_pos = rep["ml"]["p_defect"] >= 0.5
        # --- fuzzy path ---
        fz_pos = rep["fuzzy"]["p_defect"] >= 0.22
        # --- fused path ---
        fu_pos = rep["final"]["is_defect"]

        for p, pos in (("ml", ml_pos), ("fuzzy", fz_pos), ("fused", fu_pos)):
            if true_defect and pos:
                det[p]["tp"] += 1
            elif true_defect and not pos:
                det[p]["fn"] += 1
            elif (not true_defect) and pos:
                det[p]["fp"] += 1
            else:
                det[p]["tn"] += 1
            if true_defect and band == "borderline":
                border[p]["tot"] += 1
                if pos:
                    border[p]["hit"] += 1

        # --- type / class accuracy (fused & ml) on true defects ---
        if true_defect:
            type_total += 1
            if rep["ml"]["predicted_type"] == true_class:
                type_correct["ml"] += 1
            if rep["final"]["defect_type"] == true_class:
                type_correct["fused"] += 1

        # --- fused 5-class confusion (all samples) ---
        pred_cls = ("good" if not fu_pos else rep["final"]["defect_type"])
        if pred_cls not in cidx:
            pred_cls = "good"
        cm[cidx[true_class], cidx[pred_cls]] += 1

        if rep["final"]["needs_review"]:
            n_review += 1
        if not rep["final"]["agreement"]:
            n_disagree += 1

        if (k + 1) % 50 == 0:
            print(f"  evaluated {k + 1}/{len(test_df)} …")

    report = {
        "n_test": len(test_df),
        "detection": {p: _metrics(**det[p]) for p in paths},
        "borderline_recall": {
            p: round(border[p]["hit"] / border[p]["tot"], 4)
            if border[p]["tot"] else None for p in paths},
        "n_borderline": border["fused"]["tot"],
        "type_accuracy": {
            "ml": round(type_correct["ml"] / type_total, 4) if type_total else 0,
            "fused": round(type_correct["fused"] / type_total, 4) if type_total else 0,
        },
        "fused_confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist()},
        "review_flagged": n_review,
        "path_disagreements": n_disagree,
    }
    with open(os.path.join(RESULTS, "system_evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)

    _print(report)
    _plot_comparison(report)
    _plot_cm(cm)
    print(f"\nWrote results to {os.path.abspath(RESULTS)}")
    return report


def _print(r):
    print("\n" + "=" * 68)
    print(" FUSED SYSTEM EVALUATION  (held-out test set)")
    print("=" * 68)
    print(f" test samples: {r['n_test']}  |  borderline: {r['n_borderline']}")
    print("\n DEFECT-DETECTION (defect present?)")
    print(f"  {'metric':10s}{'ML':>10s}{'Fuzzy':>10s}{'FUSED':>10s}")
    for m in ("accuracy", "precision", "recall", "f1"):
        print(f"  {m:10s}"
              f"{r['detection']['ml'][m]:>10.3f}"
              f"{r['detection']['fuzzy'][m]:>10.3f}"
              f"{r['detection']['fused'][m]:>10.3f}")
    print("\n BORDERLINE recall:")
    print(f"   ML={r['borderline_recall']['ml']}  "
          f"Fuzzy={r['borderline_recall']['fuzzy']}  "
          f"FUSED={r['borderline_recall']['fused']}")
    print(f"\n TYPE accuracy:  ML={r['type_accuracy']['ml']}  "
          f"FUSED={r['type_accuracy']['fused']}")
    print(f" review-flagged: {r['review_flagged']}  "
          f"disagreements: {r['path_disagreements']}")
    print("=" * 68)


def _plot_comparison(r):
    metrics = ["accuracy", "precision", "recall", "f1"]
    ml = [r["detection"]["ml"][m] for m in metrics]
    fz = [r["detection"]["fuzzy"][m] for m in metrics]
    fu = [r["detection"]["fused"][m] for m in metrics]
    ml.append(r["borderline_recall"]["ml"] or 0)
    fz.append(r["borderline_recall"]["fuzzy"] or 0)
    fu.append(r["borderline_recall"]["fused"] or 0)
    labels = ["Accuracy", "Precision", "Recall", "F1", "Borderline\nRecall"]
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w, ml, w, label="ML only", color="#e07b39")
    ax.bar(x, fz, w, label="Fuzzy only", color="#8a63d2")
    ax.bar(x + w, fu, w, label="FUSED (deployed)", color="#2a7de1")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
    ax.set_title("Defect Detection: ML vs Fuzzy vs Fused")
    for i in range(len(labels)):
        for off, v in ((-w, ml[i]), (0, fz[i]), (w, fu[i])):
            ax.text(i + off, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "system_comparison.png"), dpi=130)
    plt.close(fig)


def _plot_cm(cm):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("FUSED System — Confusion Matrix")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fused_confusion_matrix.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
