"""
================================================================================
 inference_pipeline.py  —  REAL-TIME PRODUCTION INFERENCE PIPELINE
================================================================================

Runs the FULL production pipeline on a single uploaded image and produces a
complete, JSON-serialisable diagnostic report + visual artefacts for the web
dashboard. Every stage is explicit and timed so the UI can show a live,
step-by-step progress trace — proving the image is genuinely processed by BOTH
the Python image-processing/fuzzy engine AND the trained ML model, then fused.

Pipeline stages (each reported to the caller via a progress callback)
---------------------------------------------------------------------
  S1  Acquire & validate image
  S2  Preprocess  (grayscale → NL-means denoise → illumination fix → CLAHE)
  S3  Segment defect candidates (multi-channel saliency + morphology)
  S4  Extract 58-D feature vector (geometry · GLCM · LBP · Gabor · gradient ·
      frequency · morphology · region stats)
  S5  ML inference        (calibrated ensemble → class probabilities)
  S6  Fuzzy inference      (Mamdani FIS → severity + type)
  S7  Decision fusion      (weighted + safety rules → final verdict)
  S8  Render visual artefacts (overlay, stage panel, detection panel,
      fuzzy membership plot, probability charts)
  S9  Compile report

Public API
----------
    InferencePipeline().run(path_or_array, progress_cb=None) -> report dict
================================================================================
"""

from __future__ import annotations

import os
import io
import time
import base64
from datetime import datetime

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_engine import FeatureEngine
from fusion_engine import FusionEngine, load_model, CLASSES, DEFECT_CLASSES
import fuzzy_system as fz


# --------------------------------------------------------------------------- #
#  Encoding helpers
# --------------------------------------------------------------------------- #
def _fig_b64(fig, dpi=118):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _img_b64(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# =========================================================================== #
#  RENDERERS  (dark booth theme to match the modern UI)
# =========================================================================== #
plt.style.use("dark_background")
_BG = "#1E252B"
_FG = "#E4E7EB"
_MUTED = "#8E9AA3"
_GRID = "#2B3238"
_ACCENT = "#4FD8E8"
_GOOD = "#4ade80"
_WARN = "#fbbf24"
_BAD = "#f87171"


def _verdict_colour_bgr(decision):
    if decision.startswith("REJECT"):
        return (38, 38, 220)        # red (BGR)
    if "note" in decision:
        return (10, 145, 235)       # amber
    return (80, 175, 40)            # green


class Renderers:
    @staticmethod
    def overlay(artifacts, final):
        base = cv2.cvtColor(artifacts["enhanced"], cv2.COLOR_GRAY2BGR)
        mask = artifacts["mask"]
        colour = _verdict_colour_bgr(final["decision"])
        ov = base.copy()
        ov[mask > 0] = colour
        blend = cv2.addWeighted(base, 0.62, ov, 0.38, 0)
        
        # We no longer draw rectangles or the text banner here,
        # because the 3D stage UI handles reticles and metadata dynamically.
        return blend

    @staticmethod
    def stage_panel(artifacts):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), facecolor=_BG)
        items = [("grayscale", "1 · Grayscale"),
                 ("denoised", "2 · Denoised (NL-means)"),
                 ("illumination_corrected", "3 · Illumination corrected"),
                 ("enhanced", "4 · CLAHE enhanced")]
        for ax, (k, t) in zip(axes, items):
            ax.imshow(artifacts[k], cmap="gray", vmin=0, vmax=255)
            ax.set_title(t, color=_FG, fontsize=10)
            ax.axis("off")
        fig.suptitle("Preprocessing pipeline", color=_FG, fontsize=12)
        return _fig_b64(fig)

    @staticmethod
    def detection_panel(artifacts, overlay_bgr):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=_BG)
        axes[0].imshow(artifacts["saliency"], cmap="inferno")
        axes[0].set_title("Defect saliency", color=_FG, fontsize=10)
        axes[1].imshow(artifacts["mask"], cmap="gray")
        axes[1].set_title("Segmented mask", color=_FG, fontsize=10)
        axes[2].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Annotated overlay", color=_FG, fontsize=10)
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Defect localisation", color=_FG, fontsize=12)
        return _fig_b64(fig)

    @staticmethod
    def ml_proba_bar(proba_map):
        fig, ax = plt.subplots(figsize=(6.4, 3.0), facecolor=_BG)
        cls = list(proba_map.keys())
        vals = [proba_map[c] for c in cls]
        best = int(np.argmax(vals))
        colours = [_ACCENT if i == best else "#c7d2e8" for i in range(len(cls))]
        bars = ax.bar(cls, vals, color=colours)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=8, color=_FG)
        ax.set_ylim(0, 1.08)
        ax.set_title("ML model — class probabilities", color=_FG, fontsize=11)
        ax.tick_params(colors=_MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return _fig_b64(fig)

    @staticmethod
    def fusion_bar(p_ml, p_fuzzy, p_fused, threshold=0.45):
        fig, ax = plt.subplots(figsize=(6.4, 3.0), facecolor=_BG)
        labels = ["ML\nP(defect)", "Fuzzy\nP(defect)", "FUSED\nP(defect)"]
        vals = [p_ml, p_fuzzy, p_fused]
        colours = ["#e07b39", "#8a63d2", _ACCENT]
        bars = ax.bar(labels, vals, color=colours)
        ax.axhline(threshold, color=_BAD, ls="--", lw=1.5,
                   label=f"decision threshold = {threshold}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=9, color=_FG)
        ax.set_ylim(0, 1.08)
        ax.set_title("Decision fusion", color=_FG, fontsize=11)
        ax.tick_params(colors=_MUTED)
        ax.legend(fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return _fig_b64(fig)

    @staticmethod
    def membership(fuzzy_inputs, severity):
        size, contrast, roughness, sev = fz._SEV_VARS
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), facecolor=_BG)
        axes = axes.ravel()
        cfg = [(size, "Defect size", fuzzy_inputs.get("size", 0), _ACCENT),
               (contrast, "Defect contrast", fuzzy_inputs.get("contrast", 0), _GOOD),
               (roughness, "Surface roughness", fuzzy_inputs.get("roughness", 0), _WARN)]
        for ax, (var, label, crisp, col) in zip(axes[:3], cfg):
            for term in var.terms:
                ax.plot(var.universe, var[term].mf, lw=1.7, label=term)
            ax.axvline(crisp, color=col, ls="--", lw=2, label=f"input={crisp:.2f}")
            ax.set_title(label, color=_FG, fontsize=10)
            ax.set_ylim(-0.03, 1.08)
            ax.legend(fontsize=7)
            ax.grid(color=_GRID)
        for term in sev.terms:
            axes[3].plot(sev.universe, sev[term].mf, lw=1.7, label=term)
        axes[3].axvline(severity, color=_BAD, ls="--", lw=2.4,
                        label=f"severity={severity:.0f}")
        axes[3].set_title("Output: severity (0-100)", color=_FG, fontsize=10)
        axes[3].set_ylim(-0.03, 1.08)
        axes[3].legend(fontsize=7)
        axes[3].grid(color=_GRID)
        fig.suptitle("Fuzzy inference — membership functions & activation",
                     color=_FG, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return _fig_b64(fig)


# =========================================================================== #
#  EXPLANATION BUILDER
# =========================================================================== #
def build_explanation(report):
    ml = report["ml"]
    fu = report["fuzzy"]
    fn = report["final"]
    fus = report["fusion"]
    lines = []
    lines.append(
        f"The trained ML ensemble analysed the 58-dimensional feature vector "
        f"and estimated a <b>{ml['p_defect']*100:.0f}%</b> probability that a "
        f"defect is present, with the most likely class being "
        f"<b>{ml['predicted_class']}</b>.")
    lines.append(
        f"Independently, the fuzzy inference system evaluated size, contrast and "
        f"roughness and produced a severity of "
        f"<b>{fu['severity_score']:.0f}/100</b> "
        f"(<b>{fu['severity_label']}</b>), suggesting type "
        f"<b>{fu['defect_type']}</b>.")
    agree = "agree" if fn["agreement"] else "<b>disagree</b>"
    lines.append(
        f"The two paths {agree}. Weighted fusion "
        f"(ML {fus['weights']['ml']:.0%} / fuzzy {fus['weights']['fuzzy']:.0%})"
        f"{' plus a safety boost' if fus['safety_boost_applied'] else ''} "
        f"gives a fused defect probability of "
        f"<b>{fus['p_fused']:.2f}</b>.")
    lines.append(
        f"<b>Final verdict: {fn['decision']}</b> — type "
        f"<b>{fn['defect_type']}</b>, severity <b>{fn['severity']:.0f}/100</b>, "
        f"confidence <b>{fn['confidence']:.2f}</b>"
        f"{'. Flagged for human review.' if fn['needs_review'] else '.'}")
    return lines


# =========================================================================== #
#  PIPELINE
# =========================================================================== #
class InferencePipeline:
    def __init__(self):
        self.engine = FeatureEngine()
        self.fusion = FusionEngine()

    def run(self, path_or_array, progress_cb=None):
        def rep(p, msg, stage=None):
            if progress_cb:
                progress_cb(p, msg, stage)

        timings = {}
        t_all = time.time()

        # S1 acquire
        rep(5, "Acquiring & validating image…", "acquire")
        t = time.time()
        if isinstance(path_or_array, str):
            original = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        else:
            original = np.asarray(path_or_array)
            if original.ndim == 2:
                original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        if original is None:
            raise ValueError("Unable to read the image.")
        timings["acquire"] = round(time.time() - t, 3)

        # S2-S4 preprocess + segment + features (engine does all three)
        rep(20, "Preprocessing (denoise · illumination · CLAHE)…", "preprocess")
        t = time.time()
        features, artifacts = self.engine.extract(original)
        timings["feature_extraction"] = round(time.time() - t, 3)
        rep(42, "Segmenting defects & extracting 58 features…", "features")

        # S5-S7 ML + fuzzy + fusion
        rep(58, "Running ML model inference…", "ml")
        t = time.time()
        decision = self.fusion.decide(features)
        timings["decision"] = round(time.time() - t, 3)
        rep(70, "Running fuzzy inference & decision fusion…", "fusion")

        final = decision["final"]

        # Extract normalized bounding boxes for 3D stage reticles (max 3 largest)
        mask = artifacts["mask"]
        h_img, w_img = mask.shape
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = [c for c in cnts if cv2.contourArea(c) >= 18]
        valid_cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)[:3]
        
        bboxes = []
        for c in valid_cnts:
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append({
                "x": x / w_img, "y": y / h_img,
                "w": w / w_img, "h": h / h_img
            })
        final["bboxes"] = bboxes

        # S8 render
        rep(82, "Rendering visual diagnostics…", "render")
        t = time.time()
        overlay_bgr = Renderers.overlay(artifacts, final)
        images = {
            "original": _img_b64(original),
            "overlay": _img_b64(overlay_bgr),
            "stage_panel": Renderers.stage_panel(artifacts),
            "detection_panel": Renderers.detection_panel(artifacts, overlay_bgr),
        }
        figures = {
            "ml_proba": Renderers.ml_proba_bar(decision["ml"]["proba"]),
            "fusion_bar": Renderers.fusion_bar(
                decision["fusion"]["p_ml_defect"],
                decision["fusion"]["p_fuzzy_defect"],
                decision["fusion"]["p_fused"],
                self.fusion.cfg.DETECT_THRESHOLD),
            "membership": Renderers.membership(
                decision["fuzzy"]["fuzzy_inputs"],
                decision["fuzzy"]["severity_score"]),
        }
        timings["render"] = round(time.time() - t, 3)

        # S9 compile
        rep(95, "Compiling report…", "compile")
        explanation = build_explanation(decision)
        verdict_class = ("reject" if final["decision"].startswith("REJECT")
                         else "warn" if "note" in final["decision"]
                         else "accept")
        timings["total"] = round(time.time() - t_all, 3)
        rep(100, "Done.", "done")

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "verdict_class": verdict_class,
            "final": final,
            "ml": decision["ml"],
            "fuzzy": decision["fuzzy"],
            "fusion": decision["fusion"],
            "features": {k: round(float(v), 4) for k, v in features.items()},
            "images": images,
            "figures": figures,
            "explanation": explanation,
            "timings": timings,
        }


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import glob, json
    here = os.path.dirname(os.path.abspath(__file__))
    pipe = InferencePipeline()
    for cls in CLASSES:
        ps = glob.glob(os.path.join(here, "..", "data", "test", cls, "*.png"))
        if not ps:
            continue
        r = pipe.run(ps[0])
        f = r["final"]
        print(f"[{cls:9s}] {f['decision']:18s} type={f['defect_type']:9s} "
              f"sev={f['severity']:5.1f} conf={f['confidence']} "
              f"total={r['timings']['total']}s")
