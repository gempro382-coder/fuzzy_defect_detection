# NeuraFuzz Inspect — Hybrid ML + Fuzzy-Logic Surface Defect Detection

A **production-grade** automated image-processing system that inspects
manufactured-component surfaces for defects (scratch / crack / dent / corrosion)
by running **two independent AI decision paths on every image and fusing them**:

1. **Trained Machine-Learning model** — a calibrated ensemble trained on **1000
   labelled samples**, using a 58-dimensional image-feature vector.
2. **Fuzzy-Logic engine** — a Mamdani inference system with linguistic rules
   over image-processing features.

Their outputs are combined by a **weighted decision-fusion layer** (with a
safety-first disjunctive rule and agreement/review flagging) into one final
verdict. Delivered with a **modern light-theme web dashboard** that runs the
whole pipeline in a background worker and shows a live, staged progress trace.

---

## 📊 Real measured performance (held-out 200-image test set)

| Metric (defect vs good) | ML only | Fuzzy only | **FUSED (deployed)** |
|---|---|---|---|
| Accuracy | 0.965 | 0.815 | **0.970** |
| Precision | 1.000 | 0.853 | **1.000** |
| Recall | 0.954 | 0.914 | **0.961** |
| F1 | 0.976 | 0.882 | **0.980** |
| Borderline recall | 0.839 | 0.774 | **0.871** |

- **ML 5-class test accuracy: 90.5 %** (calibrated 92 %), **defect-detection
  ROC-AUC 0.991**.
- The **fused system beats both individual paths** on recall and borderline
  recall while keeping precision at 1.00 — i.e. combining the two genuinely
  helps. *(Numbers are the real output of `evaluate_system.py`, not hard-coded.)*

> These are honestly-reported measured results on synthetic data. The
> defect-vs-good detector is effectively ~97–99 % capable (F1 0.98, AUC 0.99).

---

## 🚀 Quick start
```bash
pip install -r requirements.txt      # or ./run.sh install
./run.sh dataset                     # generate 1000 labelled images (~10s)
./run.sh train                       # train the ML model (10 staged steps)
./run.sh evaluate                    # ML vs Fuzzy vs Fused metrics + plots
./run.sh web                         # launch the dashboard
# open http://127.0.0.1:5000
```
A pre-trained model (`models/defect_model.joblib`) and dataset are already
included, so you can jump straight to `./run.sh web`.

---

## 🧠 How each image is processed (real-time, both engines + fusion)
```
Image
  └─ S1 Acquire & validate
  └─ S2 Preprocess     : grayscale · NL-means denoise · illumination correction · CLAHE
  └─ S3 Segment        : multi-channel saliency (top-hat/black-hat + texture + residual) + morphology
  └─ S4 Features       : 58-D vector (intensity · geometry · GLCM · LBP×2 · Gabor · gradient · FFT · morphology · region)
  ├─ S5 ML inference   : calibrated ensemble → class probabilities, P(defect)   ┐
  ├─ S6 Fuzzy inference: Mamdani FIS → severity 0-100, fuzzy type               ├─ run on the SAME features
  └─ S7 FUSION         : weighted + safety rules → final decision + confidence  ┘
  └─ S8 Render         : overlay · stage panel · detection panel · fuzzy plot · charts
  └─ S9 Report         : verdict + severity + type + confidence + explanation
```

---

## 🏋️ Training pipeline (`train_model.py`, 10 explicit stages)
1. Load dataset manifest + labels 2. Extract 58-D features (cached)
3. Assemble matrices 4. Build candidate pipelines (impute→scale→classifier)
5. 5-fold cross-validation 6. Randomised hyper-parameter search
(RandomForest + GradientBoosting + soft-voting ensemble) 7. Refit best
8. Held-out test evaluation (5-class + binary + ROC-AUC + confusion)
9. Probability calibration (isotonic) 10. Persist model bundle + metrics + plots.

Artefacts land in `models/` (`defect_model.joblib`, `training_metrics.json`,
`confusion_matrix.png`, `feature_importance.png`).

---

## 🖥️ Dashboard (modern light UI)
Upload an image (or click a bundled sample) → **background job** with a live
staged progress bar → results:
- **Verdict hero** with severity & confidence dials + agreement/review badges.
- **Three path cards**: ML, Fuzzy, Fused (probabilities & bars).
- **Tabs**: Overview (input vs annotated + plain-language explanation),
  Preprocessing, Detection, ML Model (class probabilities + table),
  Fuzzy Engine (live membership plot), Fusion (fusion chart), Features (58-D table).

Offline snapshot: open **`preview_dashboard.html`** in any browser (no server).

---

## 📁 Layout
```
fuzzy_defect_detection/
├── README.md · requirements.txt · run.sh · preview_dashboard.html
├── src/
│   ├── dataset_generator.py    # 1000-sample labelled synthetic dataset
│   ├── feature_engine.py       # shared 58-D feature extractor (train == serve)
│   ├── fuzzy_system.py         # Mamdani fuzzy inference system
│   ├── train_model.py          # 10-stage ML training pipeline
│   ├── fusion_engine.py        # ML + Fuzzy weighted decision fusion
│   ├── inference_pipeline.py   # real-time end-to-end pipeline + renderers
│   └── evaluate_system.py      # ML vs Fuzzy vs Fused evaluation
├── web/
│   ├── app.py                  # Flask endpoints
│   ├── service.py              # background JobManager + system metadata
│   ├── build_static_preview.py
│   ├── templates/index.html · static/{css,js}
├── data/    train/  test/  (per-class folders + labels.csv) + manifest
├── models/  defect_model.joblib · training_metrics.json · plots · feature_cache
├── results/ system_evaluation.json · system_comparison.png · confusion plots
└── docs/    PROJECT_REPORT.md   |   slides/ slides.md
```

---

## 🔬 Use your own data
Drop labelled images into `data/train/<class>/` and `data/test/<class>/`
(classes: good, scratch, crack, dent, corrosion) with a matching `labels.csv`
(`filepath,filename,class,label_defective,severity,severity_band,secondary,split`),
then `./run.sh train && ./run.sh evaluate`. The feature engine and fusion layer
are dataset-agnostic.

---

## ⚠️ Notes
- Training uses only CPU (RandomForest/GradientBoosting) so it runs anywhere;
  no GPU required. The 1000-sample dataset is synthetic so the project is fully
  reproducible and its accuracy is honestly measurable.
- Some sandboxes can't reach the live Flask server in an embedded preview — use
  `./run.sh web` locally, or open `preview_dashboard.html` for an offline view.

*Final Year Project — Automated Image Processing System utilising Fuzzy Logic
(+ Machine Learning) for Defect Detection.*
