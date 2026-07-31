"""
fuzzy_system.py
---------------
The Fuzzy Inference System (FIS) at the core of this project.

Two cooperating Mamdani fuzzy systems are implemented with scikit-fuzzy:

1.  SEVERITY FIS
    Inputs : defect_size      (normalised defect area)
             contrast         (defect-to-background intensity contrast)
             roughness        (texture entropy / GLCM contrast)
    Output : severity         (0-100  ->  good / minor / moderate / critical)

    This is where fuzzy logic shines: borderline defects with ambiguous
    boundaries are handled gracefully through overlapping membership
    functions and linguistic rules, instead of a hard binary threshold.

2.  TYPE classifier (fuzzy scoring)
    Uses fuzzy membership reasoning over shape descriptors
    (elongation, solidity, roughness, contrast, region count) to decide the
    most likely defect TYPE: scratch / crack / dent / corrosion.

The module exposes a single high-level function `assess(features)` returning a
complete diagnosis dictionary.

Author: Final Year Project
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# =========================================================================== #
#  1.  SEVERITY  Fuzzy Inference System
# =========================================================================== #
def _build_severity_fis():
    # ---- Antecedents (inputs) ----
    # defect_size : normalised area ratio, scaled to 0-1 (we map area_ratio*100)
    size = ctrl.Antecedent(np.arange(0, 1.001, 0.001), "size")
    contrast = ctrl.Antecedent(np.arange(0, 1.001, 0.001), "contrast")
    roughness = ctrl.Antecedent(np.arange(0, 1.001, 0.001), "roughness")

    # ---- Consequent (output) ----
    severity = ctrl.Consequent(np.arange(0, 101, 1), "severity")

    # ---- Membership functions (overlapping triangles/trapezoids) ----
    # Calibrated from observed feature statistics of good vs defective surfaces.
    size["small"] = fuzz.trapmf(size.universe, [0, 0, 0.08, 0.25])
    size["medium"] = fuzz.trimf(size.universe, [0.15, 0.40, 0.65])
    size["large"] = fuzz.trapmf(size.universe, [0.50, 0.75, 1.0, 1.0])

    # contrast = defect-to-surface intensity contrast (good < ~0.15)
    contrast["low"] = fuzz.trapmf(contrast.universe, [0, 0, 0.10, 0.22])
    contrast["medium"] = fuzz.trimf(contrast.universe, [0.15, 0.32, 0.55])
    contrast["high"] = fuzz.trapmf(contrast.universe, [0.40, 0.65, 1.0, 1.0])

    # roughness = surface-anomaly indicator (good maps near 0 after flooring)
    roughness["smooth"] = fuzz.trapmf(roughness.universe, [0, 0, 0.10, 0.22])
    roughness["medium"] = fuzz.trimf(roughness.universe, [0.16, 0.36, 0.58])
    roughness["rough"] = fuzz.trapmf(roughness.universe, [0.45, 0.68, 1.0, 1.0])

    severity["good"] = fuzz.trapmf(severity.universe, [0, 0, 8, 22])
    severity["minor"] = fuzz.trimf(severity.universe, [15, 33, 50])
    severity["moderate"] = fuzz.trimf(severity.universe, [40, 58, 76])
    severity["critical"] = fuzz.trapmf(severity.universe, [68, 85, 100, 100])

    # ---- Fuzzy rule base (linguistic IF-THEN rules) ----
    rules = [
        # Clearly good surface: smooth texture AND low contrast => GOOD.
        ctrl.Rule(size["small"] & contrast["low"] & roughness["smooth"],
                  severity["good"]),
        ctrl.Rule(size["medium"] & contrast["low"] & roughness["smooth"],
                  severity["good"]),

        # --- Size-independent anomaly rules ---------------------------------
        # Thin defects (scratches, hairline cracks) occupy little AREA but
        # have strong roughness/contrast. These rules ensure such defects are
        # caught regardless of size, which a size-driven binary method misses.
        ctrl.Rule(roughness["rough"], severity["moderate"]),
        ctrl.Rule(contrast["high"], severity["moderate"]),
        ctrl.Rule(roughness["medium"] & contrast["medium"], severity["moderate"]),

        # Minor defects: small but visible, or slightly rough
        ctrl.Rule(size["small"] & contrast["medium"] & roughness["smooth"],
                  severity["minor"]),
        ctrl.Rule(size["small"] & contrast["low"] & roughness["medium"],
                  severity["minor"]),
        ctrl.Rule(roughness["medium"] & contrast["low"], severity["minor"]),
        ctrl.Rule(size["medium"] & contrast["low"] & roughness["medium"],
                  severity["minor"]),

        # Moderate defects
        ctrl.Rule(size["medium"] & contrast["medium"], severity["moderate"]),
        ctrl.Rule(size["small"] & contrast["high"] & roughness["smooth"],
                  severity["moderate"]),
        ctrl.Rule(size["medium"] & roughness["rough"], severity["moderate"]),
        ctrl.Rule(size["large"] & contrast["low"] & roughness["smooth"],
                  severity["moderate"]),

        # Critical defects
        ctrl.Rule(size["large"] & contrast["high"], severity["critical"]),
        ctrl.Rule(size["large"] & roughness["rough"], severity["critical"]),
        ctrl.Rule(size["medium"] & contrast["high"] & roughness["rough"],
                  severity["critical"]),
        ctrl.Rule(size["large"] & contrast["medium"] & roughness["medium"],
                  severity["critical"]),

        # High contrast OR very rough always raises concern (safety bias)
        ctrl.Rule(contrast["high"] & roughness["rough"], severity["critical"]),
    ]

    system = ctrl.ControlSystem(rules)
    return system, (size, contrast, roughness, severity)


_SEV_SYSTEM, _SEV_VARS = _build_severity_fis()


def evaluate_severity(size_val, contrast_val, roughness_val):
    """
    Run the severity FIS. Inputs are clamped to [0,1].
    Returns (severity_score 0-100, severity_label).
    """
    sim = ctrl.ControlSystemSimulation(_SEV_SYSTEM)
    sim.input["size"] = float(np.clip(size_val, 0, 1))
    sim.input["contrast"] = float(np.clip(contrast_val, 0, 1))
    sim.input["roughness"] = float(np.clip(roughness_val, 0, 1))
    sim.compute()
    score = float(sim.output.get("severity", 0.0))

    if score < 22:
        label = "Good (no significant defect)"
    elif score < 45:
        label = "Minor defect"
    elif score < 68:
        label = "Moderate defect"
    else:
        label = "Critical defect"
    return round(score, 2), label


# =========================================================================== #
#  2.  DEFECT TYPE  fuzzy classifier
# =========================================================================== #
def _tri(x, a, b, c):
    """Single-point triangular membership evaluation."""
    return float(fuzz.interp_membership(np.arange(0, 1.001, 0.001),
                                        fuzz.trimf(np.arange(0, 1.001, 0.001),
                                                   [a, b, c]), np.clip(x, 0, 1)))


def _trap(x, a, b, c, d):
    return float(fuzz.interp_membership(np.arange(0, 1.001, 0.001),
                                        fuzz.trapmf(np.arange(0, 1.001, 0.001),
                                                    [a, b, c, d]), np.clip(x, 0, 1)))


def classify_type(features):
    """
    Fuzzy scoring of defect type from shape & texture descriptors.

    Reasoning (linguistic):
      SCRATCH   : very elongated, high solidity, high intensity contrast, 1 region
      CRACK     : elongated, LOW solidity (jagged), dark/low contrast, branching
      DENT      : low elongation (round), medium solidity, medium contrast
      CORROSION : rough texture (high entropy / GLCM contrast), low solidity,
                  many small regions

    Returns (best_type, scores_dict).
    """
    elong = features.get("elongation", 0.0)
    solidity = features.get("solidity", 0.0)
    icontrast = features.get("norm_intensity_contrast", 0.0)
    n_reg = features.get("num_regions", 0)
    entropy = features.get("entropy", 3.6)
    # normalise region count and roughness
    n_reg_norm = min(1.0, n_reg / 20.0)
    # roughness proxy (floor-subtracted so normal texture reads ~0)
    local_rough = max(0.0, (features.get("local_roughness", 0.0) - 0.20) / 0.80)
    roughness = float(np.clip(max(local_rough, (entropy - 3.95) / 0.75), 0, 1))

    # SCRATCH : very elongated linear mark, compact (high solidity),
    #           usually bright (medium-high contrast), few regions.
    scratch = (
        _trap(elong, 0.78, 0.93, 1.0, 1.0) * 1.3 +
        _trap(solidity, 0.75, 0.88, 1.0, 1.0) * 1.0 +
        _trap(icontrast, 0.10, 0.25, 1.0, 1.0) * 0.7 +
        _trap(n_reg_norm, 0.0, 0.0, 0.25, 0.55) * 0.6
    )

    # CRACK : jagged / branching dark hairline -> LOW solidity is the defining
    #         cue, combined with LOW intensity contrast (it is dark, not bright)
    #         and elongated-or-fragmented geometry.
    crack = (
        _trap(solidity, 0.0, 0.0, 0.50, 0.68) * 1.5 +    # LOW solidity dominant
        _trap(icontrast, 0.0, 0.0, 0.10, 0.22) * 1.0 +   # dark / low contrast
        _trap(elong, 0.5, 0.75, 1.0, 1.0) * 0.5 +
        _trap(n_reg_norm, 0.3, 0.7, 1.0, 1.0) * 0.5
    )

    # DENT : rounded deformation -> LOW elongation dominant, plus clear
    #        intensity contrast from its shaded centre + bright rim. This
    #        contrast cue separates dents from cracks (which are dark/flat).
    dent = (
        _trap(elong, 0.0, 0.0, 0.25, 0.55) * 1.4 +       # LOW elong dominant
        _trap(icontrast, 0.10, 0.20, 1.0, 1.0) * 1.1 +   # visible shading
        _tri(roughness, 0.0, 0.25, 0.55) * 0.4
    )

    # CORROSION : rough pitted patch -> high roughness/entropy dominant,
    #             very LOW intensity contrast, many small regions.
    corrosion = (
        _trap(roughness, 0.22, 0.42, 1.0, 1.0) * 1.5 +   # roughness dominant
        _trap(icontrast, 0.0, 0.0, 0.08, 0.18) * 1.0 +   # very low contrast
        _trap(n_reg_norm, 0.2, 0.5, 1.0, 1.0) * 0.5
    )

    scores = {
        "scratch": round(scratch, 4),
        "crack": round(crack, 4),
        "dent": round(dent, 4),
        "corrosion": round(corrosion, 4),
    }
    best = max(scores, key=scores.get)
    return best, scores


# =========================================================================== #
#  3.  High-level assessment
# =========================================================================== #
def assess(features):
    """
    Complete fuzzy diagnosis of a component from its extracted features.

    Returns a dictionary with severity score/label, defect type, type scores,
    a pass/fail decision, and the fuzzy inputs used (for transparency).
    """
    # Map raw features to the severity FIS input scale [0,1].
    #
    # SIZE     : segmented defect area (scaled). Thin defects occupy little
    #            area, so this is a supporting -- not dominant -- cue.
    # CONTRAST : how strongly the defect stands out in intensity from the
    #            surrounding surface (bright scratches, dent shading).
    # ROUGHNESS: the primary surface-anomaly indicator. It fuses local
    #            intensity roughness inside the candidate region with the
    #            global texture entropy excess and the shape irregularity
    #            (low solidity = jagged cracks). Defect-free brushed metal
    #            sits well below the 'rough' membership.
    # Good brushed-metal surfaces yield small spurious saliency blobs
    # (area_ratio ~0.02-0.04). Subtract that texture floor before scaling so
    # only genuinely larger defect regions register as 'medium/large'.
    area_excess = max(0.0, features.get("area_ratio", 0.0) - 0.010)
    size_val = min(1.0, area_excess * 18.0)

    contrast_val = float(np.clip(features.get("norm_intensity_contrast", 0.0),
                                 0, 1))

    # Entropy of defect-free brushed metal reaches ~3.8; only treat the
    # EXCESS above that natural texture level as an anomaly signal.
    entropy = features.get("entropy", 0.0)
    entropy_excess = (entropy - 3.95) / 0.75         # good (<=3.8) -> <= 0
    solidity = features.get("solidity", 1.0)
    n_reg = features.get("num_regions", 0)
    # Jaggedness: jagged/branching cracks have low solidity AND fragment into
    # several regions. Good compact blobs do not, so we require fragmentation
    # (n_reg) before treating low solidity as a genuine crack anomaly.
    jaggedness = (max(0.0, (0.62 - solidity) / 0.62)
                  if (n_reg >= 8 and solidity < 0.55) else 0.0)

    # local_roughness of good metal is ~0.14-0.19; map with a soft floor at
    # 0.16 and full scale by ~0.85 so borderline defects (0.30+) read clearly
    # above the 'smooth' band while good stays low.
    local_rough = np.clip((features.get("local_roughness", 0.0) - 0.16) / 0.55,
                          0, 1)

    roughness_val = max(local_rough,
                        entropy_excess,
                        0.9 * jaggedness)
    roughness_val = float(np.clip(roughness_val, 0, 1))

    # If literally no defect pixels were segmented, short-circuit to "Good"
    no_defect = (features.get("num_regions", 0) == 0
                 and features.get("area_ratio", 0.0) == 0.0)

    score, sev_label = evaluate_severity(size_val, contrast_val, roughness_val)
    dtype, type_scores = classify_type(features)

    # Decision: a clean surface or very low severity => PASS
    if no_defect or score < 22:
        decision = "ACCEPT"
        dtype = "none"
        sev_label = "Good (no significant defect)"
    elif score < 45:
        decision = "ACCEPT (with note)"
    else:
        decision = "REJECT"

    return {
        "severity_score": score,
        "severity_label": sev_label,
        "defect_type": dtype,
        "type_scores": type_scores,
        "decision": decision,
        "fuzzy_inputs": {
            "size": round(size_val, 4),
            "contrast": round(contrast_val, 4),
            "roughness": round(roughness_val, 4),
        },
    }


if __name__ == "__main__":
    import json
    # Quick self-test with representative feature vectors
    demo = {
        "GOOD": {"area_ratio": 0.0, "norm_intensity_contrast": 0.04,
                 "entropy": 3.09, "norm_contrast": 0.001, "num_regions": 0,
                 "elongation": 0, "solidity": 0, "energy": 0.75},
        "SCRATCH": {"area_ratio": 0.0057, "norm_intensity_contrast": 1.0,
                    "entropy": 4.0, "norm_contrast": 0.005, "num_regions": 1,
                    "elongation": 0.99, "solidity": 0.71, "energy": 0.70},
        "CRACK": {"area_ratio": 0.006, "norm_intensity_contrast": 0.07,
                  "entropy": 3.92, "norm_contrast": 0.002, "num_regions": 3,
                  "elongation": 0.63, "solidity": 0.36, "energy": 0.66},
        "CORROSION": {"area_ratio": 0.0001, "norm_intensity_contrast": 0.20,
                      "entropy": 3.5, "norm_contrast": 0.001, "num_regions": 1,
                      "elongation": 0.56, "solidity": 0.87, "energy": 0.79},
    }
    for name, feats in demo.items():
        print(f"\n=== {name} ===")
        print(json.dumps(assess(feats), indent=2))
