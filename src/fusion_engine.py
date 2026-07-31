"""
================================================================================
 fusion_engine.py  —  DECISION-LEVEL FUSION OF ML MODEL + FUZZY SYSTEM
================================================================================

This is where the two independent decision paths are combined into ONE final,
production verdict:

    PATH A  (data-driven)  : the trained ML model  -> class probabilities,
                             P(defect), predicted type.
    PATH B  (knowledge)    : the fuzzy inference system -> severity 0-100,
                             fuzzy defect type, linguistic verdict.

Both look at the SAME 58-D feature vector, but reason completely differently
(learned statistics vs expert linguistic rules), so their errors are largely
independent — fusing them is more robust than either alone.

Fusion strategy (weighted confidence fusion, as specified)
----------------------------------------------------------
1.  DETECTION (defect present?)
      p_ml       = model P(defect)                      ∈ [0,1]
      p_fuzzy    = fuzzy severity / 100                 ∈ [0,1]
      p_fused    = w_ml * p_ml + w_fuzzy * p_fuzzy
    Final "defective" if p_fused ≥ DETECT_THRESHOLD.
    A safety-first OR-boost: if EITHER path is highly confident (> HI), the
    fused probability is raised, so a strong single detector is never ignored.

2.  TYPE (which defect?)
      Combine the ML class-probability vector with the fuzzy type-score vector
      (normalised), weighted, and take the arg-max over the 4 defect types.

3.  SEVERITY & GRADE
      Fuzzy severity is the primary grade; the ML defect-probability modulates
      it slightly upward when the model is very confident.

4.  CONFIDENCE & AGREEMENT
      Report a fused confidence and whether the two paths AGREE, so borderline
      disagreements can be surfaced for human review.

Public API
----------
    FusionEngine(model_bundle).decide(features) -> dict  (full fused report)
================================================================================
"""

from __future__ import annotations

import os
import numpy as np
import joblib

from feature_engine import FeatureEngine, FEATURE_NAMES
import fuzzy_system as fz


CLASSES = ["good", "scratch", "crack", "dent", "corrosion"]
DEFECT_CLASSES = ["scratch", "crack", "dent", "corrosion"]


# --------------------------------------------------------------------------- #
#  Tunable fusion parameters
# --------------------------------------------------------------------------- #
class FusionConfig:
    # The ML model is the stronger detector on this data, so it carries the
    # larger weight; the fuzzy path acts as a corroborating / safety signal
    # that can *raise* confidence but rarely veto a confident ML detection.
    W_ML_DETECT = 0.75          # weight of ML in detection
    W_FUZZY_DETECT = 0.25       # weight of fuzzy in detection
    DETECT_THRESHOLD = 0.45     # fused P(defect) cut-off (slightly sensitive)
    HI_CONF = 0.70              # "highly confident" single-path threshold
    HI_BOOST = 0.28             # boost applied when one path is highly confident

    W_ML_TYPE = 0.72            # weight of ML in type decision
    W_FUZZY_TYPE = 0.28         # weight of fuzzy in type decision

    # Disjunctive safety rule: if EITHER path alone crosses its solo threshold,
    # the part is flagged defective (a missed defect is costlier than a re-check).
    SOLO_ML = 0.55              # ML alone is enough above this
    SOLO_FUZZY = 0.42          # fuzzy alone is enough above this
    ML_CLEAN_VETO = 0.15       # a solo-fuzzy trigger is ignored if ML < this

    # severity grade bands (0-100)
    ACCEPT_MAX = 22
    NOTE_MAX = 45


# --------------------------------------------------------------------------- #
#  Model loader (singleton-ish)
# --------------------------------------------------------------------------- #
_MODEL_BUNDLE = None


def load_model(path: str | None = None):
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None and path is None:
        return _MODEL_BUNDLE
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "models", "defect_model.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Train it: python src/train_model.py")
    _MODEL_BUNDLE = joblib.load(path)
    return _MODEL_BUNDLE


# =========================================================================== #
#  FUSION ENGINE
# =========================================================================== #
class FusionEngine:
    def __init__(self, model_bundle=None, cfg: FusionConfig = FusionConfig()):
        self.bundle = model_bundle or load_model()
        self.model = self.bundle["model"]
        self.classes = self.bundle["classes"]
        self.feature_names = self.bundle["feature_names"]
        self.cfg = cfg
        self.engine = FeatureEngine()

    # ---- ML path ---- #
    def _ml_predict(self, features: dict) -> dict:
        vec = np.array([features.get(n, 0.0) for n in self.feature_names],
                       dtype=np.float32).reshape(1, -1)
        proba = self.model.predict_proba(vec)[0]
        proba_map = {c: float(proba[i]) for i, c in enumerate(self.classes)}
        good_p = proba_map.get("good", 0.0)
        p_defect = float(1.0 - good_p)
        # most likely defect type (exclude good)
        defect_probs = {c: proba_map[c] for c in DEFECT_CLASSES if c in proba_map}
        pred_class = max(proba_map, key=proba_map.get)
        pred_type = (max(defect_probs, key=defect_probs.get)
                     if defect_probs else "scratch")
        return {
            "proba": proba_map,
            "p_defect": round(p_defect, 4),
            "predicted_class": pred_class,
            "predicted_type": pred_type,
            "defect_type_probs": {k: round(v, 4) for k, v in defect_probs.items()},
        }

    # ---- fuzzy path ---- #
    @staticmethod
    def _fuzzy_predict(features: dict) -> dict:
        # fuzzy_system.assess expects its own feature keys; map from our engine
        mapped = {
            "area_ratio": features.get("geo_area_ratio", 0.0),
            "elongation": features.get("geo_elongation", 0.0),
            "solidity": features.get("geo_solidity", 0.0),
            "num_regions": int(round(features.get("geo_num_regions", 0.0))),
            "entropy": features.get("tex_entropy", 0.0) * 8.0,   # de-normalise
            "norm_intensity_contrast": features.get("reg_intensity_contrast", 0.0),
            "local_roughness": features.get("reg_local_roughness", 0.0),
            "norm_contrast": features.get("glcm_contrast", 0.0),
            "energy": features.get("glcm_energy", 0.0),
        }
        diag = fz.assess(mapped)
        # normalise fuzzy type scores into a probability-like vector
        ts = diag["type_scores"]
        tot = sum(ts.values()) + 1e-9
        norm_scores = {k: v / tot for k, v in ts.items()}
        diag["type_scores_norm"] = norm_scores
        diag["p_defect"] = round(diag["severity_score"] / 100.0, 4)
        return diag

    # ---- weighted detection fusion ---- #
    def _fuse_detection(self, p_ml: float, p_fuzzy: float) -> dict:
        """
        Weighted linear fusion with a SAFETY-FIRST boost: in quality control a
        missed defect (false-negative) is far costlier than a false alarm, so
        when *either* independent path is highly confident that a defect is
        present, the fused probability is boosted toward detection. This makes
        the fused detector at least as sensitive as its stronger component.
        """
        c = self.cfg
        fused = c.W_ML_DETECT * p_ml + c.W_FUZZY_DETECT * p_fuzzy
        boosted = False
        strongest = max(p_ml, p_fuzzy)
        if strongest >= c.HI_CONF and fused < strongest:
            # pull the fused value toward the confident single-path estimate
            fused = min(1.0, fused + c.HI_BOOST * (strongest - fused))
            fused = max(fused, min(strongest, fused + c.HI_BOOST))
            boosted = True

        # Weighted-threshold decision …
        is_defect = fused >= c.DETECT_THRESHOLD
        # … OR the disjunctive safety rule (either path solo-confident).
        # A solo-fuzzy trigger is vetoed only when the ML model is *strongly*
        # confident the surface is clean (guards against fuzzy over-reacting to
        # brushed-metal texture), keeping precision high without losing recall.
        solo_ml = p_ml >= c.SOLO_ML
        solo_fuzzy = (p_fuzzy >= c.SOLO_FUZZY) and (p_ml >= c.ML_CLEAN_VETO)
        if (solo_ml or solo_fuzzy) and not is_defect:
            is_defect = True
            boosted = True
            fused = max(fused, c.DETECT_THRESHOLD)   # reflect the decision
        return {"p_fused": round(float(fused), 4), "is_defect": bool(is_defect),
                "safety_boost_applied": boosted}

    # ---- weighted type fusion ---- #
    def _fuse_type(self, ml_type_probs: dict, fuzzy_type_norm: dict) -> dict:
        c = self.cfg
        combined = {}
        for t in DEFECT_CLASSES:
            combined[t] = (c.W_ML_TYPE * ml_type_probs.get(t, 0.0) +
                           c.W_FUZZY_TYPE * fuzzy_type_norm.get(t, 0.0))
        best = max(combined, key=combined.get)
        return {"type": best,
                "scores": {k: round(v, 4) for k, v in combined.items()}}

    # ---- final grade / verdict ---- #
    def _grade(self, fuzzy_severity: float, p_ml_defect: float,
               p_fused: float, is_defect: bool) -> dict:
        """
        The final severity blends the fuzzy severity (fine-grained physical
        grade) with the fused defect probability, so an ML-confident detection
        that the fuzzy path under-scored is still graded as a real defect.
        """
        c = self.cfg
        sev = float(fuzzy_severity)

        if is_defect:
            # floor the severity by the fused confidence: a confident detection
            # cannot be graded below the 'minor' band.
            prob_severity = 100.0 * p_fused
            sev = max(sev, 0.45 * sev + 0.55 * prob_severity)
            # extra push when the model is highly confident
            if p_ml_defect > 0.85:
                sev = min(100.0, sev + 10.0 * (p_ml_defect - 0.85) / 0.15)
            sev = max(sev, c.ACCEPT_MAX + 3)     # ensure it clears ACCEPT
        sev = round(float(min(100.0, sev)), 2)

        if not is_defect or sev < c.ACCEPT_MAX:
            grade, decision = "Good (no significant defect)", "ACCEPT"
        elif sev < c.NOTE_MAX:
            grade, decision = "Minor defect", "ACCEPT (with note)"
        elif sev < 68:
            grade, decision = "Moderate defect", "REJECT"
        else:
            grade, decision = "Critical defect", "REJECT"
        return {"final_severity": sev, "grade": grade, "decision": decision}

    # ---- public entry point ---- #
    def decide(self, features: dict) -> dict:
        ml = self._ml_predict(features)
        fuzzy = self._fuzzy_predict(features)

        det = self._fuse_detection(ml["p_defect"], fuzzy["p_defect"])

        if det["is_defect"]:
            typ = self._fuse_type(ml["defect_type_probs"],
                                  fuzzy["type_scores_norm"])
            final_type = typ["type"]
        else:
            typ = {"type": "none", "scores": {}}
            final_type = "none"

        grade = self._grade(fuzzy["severity_score"], ml["p_defect"],
                            det["p_fused"], det["is_defect"])
        if not det["is_defect"]:
            final_type = "none"

        # agreement: do both paths call it defective / clean the same way?
        ml_says_defect = ml["p_defect"] >= 0.5
        fuzzy_says_defect = fuzzy["p_defect"] >= (self.cfg.ACCEPT_MAX / 100.0)
        agreement = (ml_says_defect == fuzzy_says_defect)

        # fused confidence: distance of fused prob from the decision boundary
        margin = abs(det["p_fused"] - self.cfg.DETECT_THRESHOLD)
        confidence = round(min(1.0, 0.5 + margin), 4)
        if not agreement:
            confidence = round(confidence * 0.8, 4)   # penalise disagreement

        return {
            "final": {
                "decision": grade["decision"],
                "defect_type": final_type,
                "severity": grade["final_severity"],
                "grade": grade["grade"],
                "is_defect": det["is_defect"],
                "confidence": confidence,
                "p_defect_fused": det["p_fused"],
                "agreement": agreement,
                "needs_review": (not agreement) or (0.4 <= det["p_fused"] <= 0.6),
            },
            "ml": ml,
            "fuzzy": {
                "severity_score": fuzzy["severity_score"],
                "severity_label": fuzzy["severity_label"],
                "defect_type": fuzzy["defect_type"],
                "decision": fuzzy["decision"],
                "type_scores": fuzzy["type_scores"],
                "fuzzy_inputs": fuzzy["fuzzy_inputs"],
                "p_defect": fuzzy["p_defect"],
            },
            "fusion": {
                "p_ml_defect": ml["p_defect"],
                "p_fuzzy_defect": fuzzy["p_defect"],
                "p_fused": det["p_fused"],
                "weights": {"ml": self.cfg.W_ML_DETECT,
                            "fuzzy": self.cfg.W_FUZZY_DETECT},
                "type_scores": typ["scores"],
                "safety_boost_applied": det["safety_boost_applied"],
            },
        }

    # convenience: run straight from an image path/array
    def decide_from_image(self, path_or_array):
        feats, artifacts = self.engine.extract(path_or_array)
        report = self.decide(feats)
        report["features"] = feats
        return report, artifacts


# =========================================================================== #
#  Self-test
# =========================================================================== #
if __name__ == "__main__":
    import glob, json
    here = os.path.dirname(os.path.abspath(__file__))
    fe = FusionEngine()
    for cls in CLASSES:
        paths = glob.glob(os.path.join(here, "..", "data", "test", cls, "*.png"))
        if not paths:
            continue
        rep, _ = fe.decide_from_image(paths[0])
        f = rep["final"]
        print(f"[{cls:9s}] -> decision={f['decision']:18s} type={f['defect_type']:9s} "
              f"sev={f['severity']:5.1f} conf={f['confidence']} "
              f"agree={f['agreement']} | ml_p={rep['ml']['p_defect']:.2f} "
              f"fz_p={rep['fuzzy']['p_defect']:.2f}")
