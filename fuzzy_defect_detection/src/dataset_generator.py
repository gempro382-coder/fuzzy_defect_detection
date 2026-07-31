"""
================================================================================
 dataset_generator.py  —  PRODUCTION-GRADE SYNTHETIC DATASET GENERATOR
================================================================================

Generates a large, richly-varied, ground-truth-labelled dataset of synthetic
manufactured-component surface images for TRAINING and TESTING the machine
learning model, and for benchmarking the fuzzy inference system.

Why synthetic?
--------------
A defensible, reproducible benchmark requires perfectly-known ground truth.
This generator produces photorealistic-*style* brushed-metal surfaces with
physically-motivated defect signatures whose class AND severity are known
exactly, so measured accuracy is trustworthy.

Design goals
------------
1.  Realism      — brushed-metal texture, vignetting, directional lighting,
                   camera sensor noise, JPEG-like softening, dust specks.
2.  Variety      — 5 classes (good, scratch, crack, dent, corrosion), each with
                   randomised geometry, orientation, position, intensity and a
                   continuous severity in [0,1].
3.  Difficulty   — a controllable fraction of *borderline* (subtle) defects to
                   stress-test both the ML model and the fuzzy system.
4.  Balance      — near-equal class counts with a stratified train/test split.
5.  Reproducible — deterministic given a seed.

Outputs
-------
    data/train/<class>/*.png          training images (per-class folders)
    data/test/<class>/*.png           held-out test images
    data/train/labels.csv             filename,class,severity,split,...
    data/test/labels.csv
    data/dataset_manifest.json        summary statistics of the generated set

Run
---
    python src/dataset_generator.py --n 1000 --img-size 256 --seed 42
================================================================================
"""

from __future__ import annotations

import os
import csv
import json
import time
import argparse
from dataclasses import dataclass, asdict, field
from typing import Callable

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
CLASSES = ["good", "scratch", "crack", "dent", "corrosion"]
DEFECT_CLASSES = ["scratch", "crack", "dent", "corrosion"]


@dataclass
class GenConfig:
    """All tunable knobs for dataset generation."""
    n_samples: int = 1000
    img_size: int = 256
    seed: int = 42
    test_fraction: float = 0.20
    # class distribution weights (good is a bit larger as a single class)
    class_weights: dict = field(default_factory=lambda: {
        "good": 0.24, "scratch": 0.19, "crack": 0.19,
        "dent": 0.19, "corrosion": 0.19,
    })
    # severity sampling
    min_severity: float = 0.18
    max_severity: float = 1.0
    borderline_fraction: float = 0.20   # share of defects that are subtle
    # surface appearance
    base_gray_mean: tuple = (120, 175)
    brushed_strength: tuple = (4, 12)
    vignette_strength: tuple = (0.12, 0.35)
    sensor_noise_sigma: tuple = (1.2, 5.5)
    add_dust_prob: float = 0.5
    multi_defect_prob: float = 0.18       # secondary, lighter defect


# =========================================================================== #
#  SECTION 1 — SURFACE (BACKGROUND) SYNTHESIS
# =========================================================================== #
class SurfaceFactory:
    """Creates realistic defect-free brushed-metal backgrounds."""

    def __init__(self, cfg: GenConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def _brushed_metal(self, size: int) -> np.ndarray:
        mean = self.rng.uniform(*self.cfg.base_gray_mean)
        base = self.rng.normal(mean, 6, (size, size)).astype(np.float32)

        # directional brushed streaks (random dominant orientation)
        if self.rng.random() < 0.7:
            streaks = self.rng.normal(0, self.rng.uniform(*self.cfg.brushed_strength),
                                      (size, 1)).astype(np.float32)
            base += streaks                      # horizontal streaks
        else:
            streaks = self.rng.normal(0, self.rng.uniform(*self.cfg.brushed_strength),
                                      (1, size)).astype(np.float32)
            base += streaks                      # vertical streaks

        # fine grain
        grain = self.rng.normal(0, 2.5, (size, size)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), 0.7)
        base += grain
        return base

    def _apply_illumination(self, img: np.ndarray) -> np.ndarray:
        size = img.shape[0]
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

        # radial vignette from a random centre
        cx, cy = self.rng.uniform(0.25, 0.75, 2) * size
        radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        vig = 1.0 - self.rng.uniform(*self.cfg.vignette_strength) * (radial / radial.max())

        # linear directional gradient (simulated angled lighting)
        ang = self.rng.uniform(0, 2 * np.pi)
        grad = (np.cos(ang) * (xx / size) + np.sin(ang) * (yy / size))
        grad = 1.0 + self.rng.uniform(-0.12, 0.12) * (grad - grad.mean())

        return img * vig * grad

    def _sensor_effects(self, img: np.ndarray) -> np.ndarray:
        # gaussian sensor noise
        sigma = self.rng.uniform(*self.cfg.sensor_noise_sigma)
        img = img + self.rng.normal(0, sigma, img.shape)

        # occasional dust specks (tiny bright/dark dots -> nuisance, not defect)
        if self.rng.random() < self.cfg.add_dust_prob:
            n = self.rng.integers(3, 20)
            for _ in range(n):
                x, y = self.rng.integers(0, img.shape[1]), self.rng.integers(0, img.shape[0])
                val = self.rng.choice([self.rng.uniform(30, 70),
                                       self.rng.uniform(190, 240)])
                cv2.circle(img, (int(x), int(y)), int(self.rng.integers(0, 2)),
                           float(val), -1)

        # mild blur to emulate optics / mild JPEG softening
        if self.rng.random() < 0.4:
            img = cv2.GaussianBlur(img, (0, 0), self.rng.uniform(0.4, 0.9))
        return img

    def make(self, size: int) -> np.ndarray:
        img = self._brushed_metal(size)
        img = self._apply_illumination(img)
        return np.clip(img, 0, 255).astype(np.float32)


# =========================================================================== #
#  SECTION 2 — DEFECT PAINTERS (one per defect class)
# =========================================================================== #
class DefectPainter:
    """Paints physically-motivated defect signatures onto a surface."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    # ---- scratch: thin bright reflective line ---------------------------- #
    def scratch(self, img: np.ndarray, severity: float) -> np.ndarray:
        h, w = img.shape
        x1, y1 = self.rng.integers(0, w), self.rng.integers(0, h)
        angle = self.rng.uniform(0, np.pi)
        length = int(self.rng.uniform(0.25, 0.9) * w * (0.5 + severity))
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
        thickness = max(1, int(round(1 + severity * 2.5)))
        base_val = self.rng.uniform(190, 255)
        color = float(np.clip(base_val * (0.75 + 0.25 * severity), 0, 255))
        overlay = img.copy()
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        # slight glow around a strong scratch
        if severity > 0.5:
            glow = cv2.GaussianBlur(overlay, (0, 0), 1.2)
            overlay = np.maximum(overlay, glow * 0.5 + img * 0.5)
        return overlay

    # ---- crack: jagged branching dark line ------------------------------ #
    def crack(self, img: np.ndarray, severity: float) -> np.ndarray:
        h, w = img.shape
        overlay = img.copy()

        def draw_branch(x, y, angle, steps, thickness):
            pts = [(x, y)]
            for _ in range(steps):
                angle += self.rng.uniform(-0.7, 0.7)
                step = self.rng.uniform(3, 9)
                x = int(np.clip(x + step * np.cos(angle), 0, w - 1))
                y = int(np.clip(y + step * np.sin(angle), 0, h - 1))
                pts.append((x, y))
            color = float(np.clip(45 - severity * 35, 0, 255))
            cv2.polylines(overlay, [np.array(pts, np.int32)], False,
                          color, thickness, cv2.LINE_AA)
            return pts

        x, y = self.rng.integers(w // 4, 3 * w // 4), self.rng.integers(h // 4, 3 * h // 4)
        main_angle = self.rng.uniform(0, 2 * np.pi)
        steps = int(15 + severity * 45)
        thickness = max(1, int(round(1 + severity * 1.8)))
        main = draw_branch(x, y, main_angle, steps, thickness)

        # side branches (more for severe cracks)
        n_branch = int(self.rng.integers(0, 1 + int(severity * 3)))
        for _ in range(n_branch):
            bx, by = main[self.rng.integers(0, len(main))]
            draw_branch(bx, by, main_angle + self.rng.uniform(-1.2, 1.2),
                        int(steps * 0.5), max(1, thickness - 1))
        return overlay

    # ---- dent: circular shading + bright rim ---------------------------- #
    def dent(self, img: np.ndarray, severity: float) -> np.ndarray:
        h, w = img.shape
        m = 40
        cx, cy = self.rng.integers(m, w - m), self.rng.integers(m, h - m)
        radius = int(self.rng.uniform(15, 45) * (0.7 + severity))
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        core = np.exp(-((dist ** 2) / (2.0 * radius ** 2)))
        rim = np.exp(-(((dist - radius) ** 2) / (2.0 * (radius * 0.22) ** 2)))
        overlay = img.copy()
        overlay -= core * (70 * severity)
        overlay += rim * (65 * severity)
        return overlay

    # ---- corrosion: rough pitted patch ---------------------------------- #
    def corrosion(self, img: np.ndarray, severity: float) -> np.ndarray:
        """
        Corrosion always produces a visibly ROUGH, pitted micro-texture even at
        low severity (real oxidation is never perfectly smooth). A minimum
        texture floor guarantees separability from clean brushed metal while
        severity scales the intensity and pit density.
        """
        h, w = img.shape
        m = 40
        cx, cy = self.rng.integers(m, w - m), self.rng.integers(m, h - m)
        radius = int(self.rng.uniform(24, 58) * (0.7 + severity))
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = (dist <= radius).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), 5)

        # high-frequency rough texture with a guaranteed minimum amplitude
        amp = 28 + 40 * severity            # floor of 28 even for borderline
        noise = self.rng.normal(0, amp, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 0.7)
        # add a second, finer noise octave for realistic granularity
        fine = self.rng.normal(0, amp * 0.6, (h, w)).astype(np.float32)
        overlay = img + (noise + fine) * mask - (10 + 14 * severity) * mask

        # discrete pits: dense even at low severity
        n_pits = int(28 + 45 * severity)
        for _ in range(n_pits):
            a = self.rng.uniform(0, 2 * np.pi)
            r = self.rng.uniform(0, radius * 0.92)
            px, py = int(cx + r * np.cos(a)), int(cy + r * np.sin(a))
            pr = int(self.rng.integers(1, 3))
            val = float(self.rng.uniform(25, 95))
            cv2.circle(overlay, (px, py), pr, val, -1)
        return overlay

    def paint(self, cls: str, img: np.ndarray, severity: float) -> np.ndarray:
        fn: Callable = getattr(self, cls)
        return fn(img, severity)


# =========================================================================== #
#  SECTION 3 — DATASET BUILDER
# =========================================================================== #
class DatasetBuilder:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.surface = SurfaceFactory(cfg, self.rng)
        self.painter = DefectPainter(self.rng)

    # ---- severity sampling with a controllable borderline share -------- #
    def _sample_severity(self) -> tuple[float, str]:
        if self.rng.random() < self.cfg.borderline_fraction:
            sev = float(self.rng.uniform(self.cfg.min_severity, 0.40))
        else:
            sev = float(self.rng.uniform(0.40, self.cfg.max_severity))
        band = ("borderline" if sev < 0.40 else
                "moderate" if sev < 0.72 else "severe")
        return round(sev, 3), band

    def _plan_classes(self) -> list[str]:
        """Return a shuffled list of class labels honouring the weights."""
        weights = np.array([self.cfg.class_weights[c] for c in CLASSES], dtype=float)
        weights = weights / weights.sum()
        counts = np.round(weights * self.cfg.n_samples).astype(int)
        # fix rounding drift
        while counts.sum() < self.cfg.n_samples:
            counts[self.rng.integers(0, len(CLASSES))] += 1
        while counts.sum() > self.cfg.n_samples:
            i = self.rng.integers(0, len(CLASSES))
            if counts[i] > 0:
                counts[i] -= 1
        plan = []
        for c, n in zip(CLASSES, counts):
            plan.extend([c] * int(n))
        self.rng.shuffle(plan)
        return plan

    def _make_one(self, cls: str) -> tuple[np.ndarray, dict]:
        size = self.cfg.img_size
        img = self.surface.make(size)
        meta = {"class": cls, "severity": 0.0, "severity_band": "none",
                "secondary": ""}

        if cls != "good":
            sev, band = self._sample_severity()
            img = self.painter.paint(cls, img, sev)
            meta.update(severity=sev, severity_band=band)

            # optional lighter secondary defect of another class
            if self.rng.random() < self.cfg.multi_defect_prob:
                other = self.rng.choice([c for c in DEFECT_CLASSES if c != cls])
                img = self.painter.paint(other, img, sev * 0.35)
                meta["secondary"] = str(other)

        img = self.surface._sensor_effects(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, meta

    def build(self, out_root: str):
        t0 = time.time()
        plan = self._plan_classes()

        # prepare directories
        for split in ("train", "test"):
            for c in CLASSES:
                os.makedirs(os.path.join(out_root, split, c), exist_ok=True)

        rows = {"train": [], "test": []}
        per_class_split = {c: {"train": 0, "test": 0} for c in CLASSES}
        idx = 0
        for cls in plan:
            img, meta = self._make_one(cls)
            # stratified split: assign test by per-class running fraction
            n_seen = per_class_split[cls]["train"] + per_class_split[cls]["test"]
            want_test = int(round((n_seen + 1) * self.cfg.test_fraction)) > \
                per_class_split[cls]["test"]
            split = "test" if want_test else "train"
            per_class_split[cls][split] += 1

            fname = f"{cls}_{idx:05d}_{meta['severity_band']}.png"
            rel = os.path.join(split, cls, fname)
            cv2.imwrite(os.path.join(out_root, rel), img)
            rows[split].append({
                "filepath": rel.replace("\\", "/"),
                "filename": fname,
                "class": cls,
                "label_defective": int(cls != "good"),
                "severity": meta["severity"],
                "severity_band": meta["severity_band"],
                "secondary": meta["secondary"],
                "split": split,
            })
            idx += 1
            if idx % 100 == 0:
                print(f"  generated {idx}/{self.cfg.n_samples} …")

        # write per-split label CSVs
        for split in ("train", "test"):
            csv_path = os.path.join(out_root, split, "labels.csv")
            self._write_csv(csv_path, rows[split])

        manifest = self._manifest(rows, per_class_split, time.time() - t0)
        with open(os.path.join(out_root, "dataset_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nDataset complete in {manifest['elapsed_sec']}s")
        print(json.dumps(manifest["counts"], indent=2))
        return manifest

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def _manifest(self, rows, per_class_split, elapsed):
        counts = {
            "total": self.cfg.n_samples,
            "train": len(rows["train"]),
            "test": len(rows["test"]),
            "per_class": per_class_split,
        }
        return {
            "config": asdict(self.cfg),
            "classes": CLASSES,
            "counts": counts,
            "elapsed_sec": round(elapsed, 2),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


# =========================================================================== #
#  CLI
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description="Synthetic defect dataset generator")
    ap.add_argument("--n", type=int, default=1000, help="total samples")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-fraction", type=float, default=0.20)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_root = args.out or os.path.join(here, "..", "data")

    cfg = GenConfig(n_samples=args.n, img_size=args.img_size, seed=args.seed,
                    test_fraction=args.test_fraction)
    print("=" * 70)
    print(" SYNTHETIC DEFECT DATASET GENERATOR")
    print("=" * 70)
    print(f" samples={cfg.n_samples}  img_size={cfg.img_size}  seed={cfg.seed}")
    print(f" output={os.path.abspath(out_root)}")
    print("-" * 70)
    DatasetBuilder(cfg).build(out_root)


if __name__ == "__main__":
    main()
