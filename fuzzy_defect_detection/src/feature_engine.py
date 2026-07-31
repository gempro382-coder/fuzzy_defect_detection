"""
================================================================================
 feature_engine.py  —  PRODUCTION FEATURE EXTRACTION ENGINE
================================================================================

A single, authoritative feature-extraction engine used IDENTICALLY by:
    * the ML training pipeline (train_model.py), and
    * the real-time inference pipeline (inference_pipeline.py / web app).

This guarantees train/serve consistency (no feature skew).

It produces a rich, fixed-order feature vector (~40 features) spanning:

  A. Global intensity statistics        (mean, std, skew, kurtosis, percentiles)
  B. Segmentation-based geometry        (area, elongation, solidity, extent,
                                         eccentricity, region count, perimeter…)
  C. GLCM Haralick texture              (contrast, homogeneity, energy,
                                         correlation, dissimilarity, ASM, entropy)
  D. Local Binary Pattern histogram     (rotation-invariant micro-texture)
  E. Gradient / edge statistics         (Sobel magnitude, edge density)
  F. Frequency-domain energy            (high-frequency ratio via FFT)
  G. Morphological line responses       (top-hat / black-hat -> scratches/cracks)
  H. Local roughness map statistics     (corrosion / pitting)

The engine also returns intermediate ARTIFACTS (preprocessing stages, saliency,
mask, overlay-ready data) for visualisation and for the fuzzy system.

Public API
----------
    FeatureEngine().extract(path_or_array) -> (features: dict, artifacts: dict)
    FEATURE_NAMES  (ordered list -> the model's expected column order)
    features_to_vector(features) -> np.ndarray  (in FEATURE_NAMES order)
================================================================================
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import stats as sstats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy


# =========================================================================== #
#  PREPROCESSOR
# =========================================================================== #
class Preprocessor:
    """Noise reduction, illumination correction and contrast enhancement."""

    def __init__(self, nlm_h: int = 4, illum_sigma: int = 31,
                 clahe_clip: float = 2.0, clahe_grid: int = 8):
        self.nlm_h = nlm_h
        self.illum_sigma = illum_sigma
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                     tileGridSize=(clahe_grid, clahe_grid))

    @staticmethod
    def to_gray(x) -> np.ndarray:
        if isinstance(x, str):
            img = cv2.imread(x, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {x}")
        else:
            img = np.asarray(x)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.uint8)

    def denoise(self, g: np.ndarray) -> np.ndarray:
        g = cv2.GaussianBlur(g, (3, 3), 0)
        return cv2.fastNlMeansDenoising(g, None, h=self.nlm_h,
                                        templateWindowSize=7, searchWindowSize=21)

    def correct_illumination(self, g: np.ndarray) -> np.ndarray:
        bg = cv2.GaussianBlur(g, (0, 0), sigmaX=self.illum_sigma)
        corrected = cv2.subtract(g, bg)
        corrected = cv2.add(corrected, int(np.mean(bg)))
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def run(self, x) -> dict:
        gray = self.to_gray(x)
        den = self.denoise(gray)
        illum = self.correct_illumination(den)
        enhanced = self.clahe.apply(illum)
        return {"grayscale": gray, "denoised": den,
                "illumination_corrected": illum, "enhanced": enhanced}


# =========================================================================== #
#  SEGMENTER  (multi-channel defect saliency)
# =========================================================================== #
class Segmenter:
    def __init__(self, k_sigma: float = 2.6, min_abs: int = 8):
        self.k_sigma = k_sigma
        self.min_abs = min_abs

    @staticmethod
    def _local_std(gray: np.ndarray, win: int = 15) -> np.ndarray:
        g = gray.astype(np.float32)
        mean = cv2.boxFilter(g, -1, (win, win), normalize=True)
        sq = cv2.boxFilter(g * g, -1, (win, win), normalize=True)
        return np.sqrt(np.clip(sq - mean * mean, 0, None))

    def saliency(self, enhanced: np.ndarray):
        g = enhanced.astype(np.float32)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        wth = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, se).astype(np.float32)
        bth = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, se).astype(np.float32)
        line = np.maximum(wth, bth)

        tex = self._local_std(enhanced, 15)
        tex_anom = np.clip(tex - cv2.GaussianBlur(tex, (0, 0), 21), 0, None)

        smooth = cv2.GaussianBlur(enhanced, (0, 0), 9)
        residual = np.abs(g - smooth.astype(np.float32))

        def n(x):
            return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

        sal = np.maximum.reduce([n(line), n(tex_anom) * 0.9, n(residual) * 0.7])
        sal = cv2.GaussianBlur(sal, (3, 3), 0)
        return sal

    def segment(self, enhanced: np.ndarray):
        sal = self.saliency(enhanced)
        thr = max(self.min_abs, sal.mean() + self.k_sigma * sal.std())
        mask = (sal >= thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        sal_vis = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mask, sal, sal_vis


# =========================================================================== #
#  FEATURE COMPUTERS
# =========================================================================== #
def _safe(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


class FeatureComputers:
    """Each method returns an ordered dict of named features."""

    # ---- A. global intensity statistics ---- #
    @staticmethod
    def intensity(gray: np.ndarray) -> dict:
        x = gray.astype(np.float32).ravel()
        p = np.percentile(x, [5, 25, 50, 75, 95])
        return {
            "int_mean": _safe(x.mean() / 255.0),
            "int_std": _safe(x.std() / 128.0),
            "int_skew": _safe(sstats.skew(x)),
            "int_kurtosis": _safe(sstats.kurtosis(x)),
            "int_p5": _safe(p[0] / 255.0),
            "int_p50": _safe(p[2] / 255.0),
            "int_p95": _safe(p[4] / 255.0),
            "int_iqr": _safe((p[3] - p[1]) / 255.0),
            "int_range": _safe((x.max() - x.min()) / 255.0),
        }

    # ---- B. geometry from segmentation ---- #
    @staticmethod
    def geometry(mask: np.ndarray) -> dict:
        h, w = mask.shape
        img_area = float(h * w)
        out = {
            "geo_area_ratio": 0.0, "geo_elongation": 0.0, "geo_solidity": 0.0,
            "geo_extent": 0.0, "geo_eccentricity": 0.0, "geo_aspect": 0.0,
            "geo_num_regions": 0.0, "geo_perimeter_ratio": 0.0,
            "geo_compactness": 0.0, "geo_orientation_disp": 0.0,
            "geo_fill_biggest": 0.0,
        }
        n_lab, labels, statc, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_lab <= 1:
            return out
        areas = statc[1:, cv2.CC_STAT_AREA]
        valid = np.where(areas >= 15)[0]
        if len(valid) == 0:
            return out

        out["geo_num_regions"] = float(len(valid))
        out["geo_area_ratio"] = float(areas[valid].sum() / img_area)
        biggest = valid[np.argmax(areas[valid])] + 1
        rmask = (labels == biggest).astype(np.uint8)

        cnts, _ = cv2.findContours(rmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        out["geo_solidity"] = _safe(area / hull_area) if hull_area > 0 else 0.0

        x, y, bw, bh = cv2.boundingRect(cnt)
        out["geo_extent"] = _safe(area / (bw * bh)) if bw * bh > 0 else 0.0
        long_s, short_s = max(bw, bh), max(1, min(bw, bh))
        out["geo_aspect"] = _safe(min(1.0, (long_s / short_s) / 20.0))
        out["geo_perimeter_ratio"] = _safe(perim / (2 * (bw + bh))) if (bw + bh) else 0.0
        out["geo_compactness"] = _safe((4 * np.pi * area) / (perim ** 2)) if perim > 0 else 0.0
        out["geo_fill_biggest"] = _safe(area / (areas[valid].sum() + 1e-6))

        if len(cnt) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
            major, minor = max(MA, ma), min(MA, ma)
            if major > 0:
                out["geo_elongation"] = _safe(1.0 - minor / major)
                out["geo_eccentricity"] = _safe(np.sqrt(max(0.0, 1 - (minor / major) ** 2)))

        # dispersion of region orientations (branching cracks -> high)
        orientations = []
        for lbl in valid:
            m = (labels == (lbl + 1)).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs and len(cs[0]) >= 5:
                (_, _), (_, _), ang = cv2.fitEllipse(cs[0])
                orientations.append(ang)
        if len(orientations) >= 2:
            out["geo_orientation_disp"] = _safe(np.std(orientations) / 90.0)
        return out

    # ---- C. GLCM Haralick texture ---- #
    @staticmethod
    def glcm(enhanced: np.ndarray) -> dict:
        levels = 32
        q = np.clip((enhanced.astype(np.float32) / 256.0 * levels), 0, levels - 1).astype(np.uint8)
        g = graycomatrix(q, distances=[1, 2, 4],
                         angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=levels, symmetric=True, normed=True)
        props = {}
        for p in ("contrast", "dissimilarity", "homogeneity", "energy",
                  "correlation", "ASM"):
            props[f"glcm_{p}"] = _safe(graycoprops(g, p).mean())
        props["glcm_contrast"] = _safe(props["glcm_contrast"] / 50.0)  # normalise
        props["glcm_dissimilarity"] = _safe(props["glcm_dissimilarity"] / 8.0)
        props["tex_entropy"] = _safe(shannon_entropy(enhanced) / 8.0)
        return props

    # ---- D. Local Binary Pattern (multi-scale, rotation invariant) ---- #
    @staticmethod
    def lbp(enhanced: np.ndarray, P: int = 8, R: int = 1) -> dict:
        codes = local_binary_pattern(enhanced, P, R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins), density=True)
        out = {f"lbp_{i}": _safe(hist[i]) for i in range(n_bins)}
        # coarse-scale LBP (R=3) — captures larger corrosion pits / roughness
        codes3 = local_binary_pattern(enhanced, P, 3, method="uniform")
        hist3, _ = np.histogram(codes3, bins=n_bins, range=(0, n_bins), density=True)
        # summarise the coarse histogram to a few robust scalars
        out["lbp3_uniform_frac"] = _safe(hist3[:-1].sum())
        out["lbp3_nonuniform"] = _safe(hist3[-1])
        out["lbp3_entropy"] = _safe(sstats.entropy(hist3 + 1e-9) / np.log(n_bins))
        return out

    # ---- D2. Gabor filter-bank energy (multi-orientation / frequency) ---- #
    @staticmethod
    def gabor(enhanced: np.ndarray) -> dict:
        """
        Gabor responses are highly discriminative for oriented texture:
        smooth metal -> low energy; corrosion/pitting -> high, isotropic energy;
        scratches/cracks -> strong response at their orientation.
        """
        img = enhanced.astype(np.float32) / 255.0
        energies = []
        thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        freqs = [0.15, 0.30]
        for f in freqs:
            for th in thetas:
                ksize = 15
                lam = 1.0 / f
                kern = cv2.getGaborKernel((ksize, ksize), sigma=3.0, theta=th,
                                          lambd=lam, gamma=0.5, psi=0)
                resp = cv2.filter2D(img, cv2.CV_32F, kern)
                energies.append(float(np.sqrt((resp ** 2).mean())))
        energies = np.array(energies)
        return {
            "gabor_mean": _safe(energies.mean()),
            "gabor_max": _safe(energies.max()),
            "gabor_std": _safe(energies.std()),
            "gabor_aniso": _safe(energies.std() / (energies.mean() + 1e-6)),
        }

    # ---- E. gradient / edges ---- #
    @staticmethod
    def gradient(enhanced: np.ndarray) -> dict:
        gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edges = cv2.Canny(enhanced, 50, 150)
        return {
            "grad_mean": _safe(mag.mean() / 255.0),
            "grad_std": _safe(mag.std() / 255.0),
            "grad_max": _safe(mag.max() / 1024.0),
            "edge_density": _safe((edges > 0).mean()),
        }

    # ---- F. frequency-domain energy ---- #
    @staticmethod
    def frequency(enhanced: np.ndarray) -> dict:
        f = np.fft.fftshift(np.fft.fft2(enhanced.astype(np.float32)))
        mag = np.abs(f)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        rmax = dist.max()
        total = mag.sum() + 1e-9
        high = mag[dist > 0.30 * rmax].sum()
        mid = mag[(dist > 0.10 * rmax) & (dist <= 0.30 * rmax)].sum()
        return {
            "freq_high_ratio": _safe(high / total),
            "freq_mid_ratio": _safe(mid / total),
        }

    # ---- G. morphological line responses ---- #
    @staticmethod
    def morphology(enhanced: np.ndarray, mask: np.ndarray) -> dict:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        wth = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, se).astype(np.float32)
        bth = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, se).astype(np.float32)
        return {
            "morph_tophat_mean": _safe(wth.mean() / 40.0),
            "morph_tophat_p99": _safe(np.percentile(wth, 99) / 60.0),
            "morph_blackhat_mean": _safe(bth.mean() / 40.0),
            "morph_blackhat_p99": _safe(np.percentile(bth, 99) / 60.0),
        }

    # ---- H. defect-region roughness / contrast ---- #
    @staticmethod
    def region_stats(enhanced: np.ndarray, mask: np.ndarray, saliency: np.ndarray) -> dict:
        out = {"reg_local_roughness": 0.0, "reg_intensity_contrast": 0.0,
               "reg_saliency_mean": 0.0, "reg_saliency_p95": 0.0}
        m = mask > 0
        if m.sum() > 0:
            dp = enhanced[m]
            bg = enhanced[~m]
            out["reg_local_roughness"] = _safe(min(1.0, dp.std() / 45.0))
            if bg.size:
                out["reg_intensity_contrast"] = _safe(min(1.0, abs(dp.mean() - bg.mean()) / 60.0))
            out["reg_saliency_mean"] = _safe(saliency[m].mean() / 255.0)
            out["reg_saliency_p95"] = _safe(np.percentile(saliency[m], 95) / 255.0)
        return out


# =========================================================================== #
#  ORDERED FEATURE NAME LIST  (defines the model's column order)
# =========================================================================== #
def _build_feature_names() -> list[str]:
    names = []
    names += ["int_mean", "int_std", "int_skew", "int_kurtosis", "int_p5",
              "int_p50", "int_p95", "int_iqr", "int_range"]
    names += ["geo_area_ratio", "geo_elongation", "geo_solidity", "geo_extent",
              "geo_eccentricity", "geo_aspect", "geo_num_regions",
              "geo_perimeter_ratio", "geo_compactness", "geo_orientation_disp",
              "geo_fill_biggest"]
    names += ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
              "glcm_energy", "glcm_correlation", "glcm_ASM", "tex_entropy"]
    names += [f"lbp_{i}" for i in range(10)]          # P=8 -> 10 bins
    names += ["lbp3_uniform_frac", "lbp3_nonuniform", "lbp3_entropy"]
    names += ["gabor_mean", "gabor_max", "gabor_std", "gabor_aniso"]
    names += ["grad_mean", "grad_std", "grad_max", "edge_density"]
    names += ["freq_high_ratio", "freq_mid_ratio"]
    names += ["morph_tophat_mean", "morph_tophat_p99",
              "morph_blackhat_mean", "morph_blackhat_p99"]
    names += ["reg_local_roughness", "reg_intensity_contrast",
              "reg_saliency_mean", "reg_saliency_p95"]
    return names


FEATURE_NAMES: list[str] = _build_feature_names()


# =========================================================================== #
#  MAIN ENGINE
# =========================================================================== #
class FeatureEngine:
    def __init__(self):
        self.pre = Preprocessor()
        self.seg = Segmenter()
        self.fc = FeatureComputers()

    def extract(self, path_or_array):
        stages = self.pre.run(path_or_array)
        gray = stages["grayscale"]
        enhanced = stages["enhanced"]
        mask, saliency, sal_vis = self.seg.segment(enhanced)

        feats = {}
        feats.update(self.fc.intensity(gray))
        feats.update(self.fc.geometry(mask))
        feats.update(self.fc.glcm(enhanced))
        feats.update(self.fc.lbp(enhanced))
        feats.update(self.fc.gabor(enhanced))
        feats.update(self.fc.gradient(enhanced))
        feats.update(self.fc.frequency(enhanced))
        feats.update(self.fc.morphology(enhanced, mask))
        feats.update(self.fc.region_stats(enhanced, mask, saliency))

        # ensure every declared feature exists
        for name in FEATURE_NAMES:
            feats.setdefault(name, 0.0)

        artifacts = {**stages, "mask": mask, "saliency": sal_vis}
        return feats, artifacts

    @staticmethod
    def to_vector(feats: dict) -> np.ndarray:
        return np.array([_safe(feats.get(n, 0.0)) for n in FEATURE_NAMES],
                        dtype=np.float32)


def features_to_vector(feats: dict) -> np.ndarray:
    return FeatureEngine.to_vector(feats)


# =========================================================================== #
#  Self-test
# =========================================================================== #
if __name__ == "__main__":
    import os, glob, json
    here = os.path.dirname(os.path.abspath(__file__))
    eng = FeatureEngine()
    print(f"Total features declared: {len(FEATURE_NAMES)}")
    samples = glob.glob(os.path.join(here, "..", "data", "train", "*", "*.png"))[:3]
    for s in samples:
        f, art = eng.extract(s)
        vec = eng.to_vector(f)
        print(f"\n{os.path.basename(s)}  vector shape={vec.shape} "
              f"nan={np.isnan(vec).any()}")
        print("  sample feats:", json.dumps(
            {k: round(f[k], 3) for k in list(FEATURE_NAMES)[:6]}))
