# Presentation Slides
### Automated Image Processing System using Fuzzy Logic for Defect Detection
*(Render with Marp / Reveal.js, or read as an outline. Each `---` is a new slide.)*

---

## Slide 1 — Title
**Automated Image Processing System utilising Fuzzy Logic for Defect Detection**
A software-based prototype for industrial quality control
Final Year Project

---

## Slide 2 — The Problem
- Manual inspection: slow, subjective, inconsistent.
- Conventional AOI (binary thresholding): **one hard decision boundary**.
- Real defects (hairline cracks, faint scratches) have **no sharp edge** and
  change with **lighting**.
- ➜ Binary methods **miss borderline defects** or **over-flag** texture.

---

## Slide 3 — The Idea: Fuzzy Logic
- Human inspectors reason in *grades*: "a fairly long, fairly faint scratch."
- Fuzzy logic formalises this with **membership degrees** and **IF–THEN rules**.
- More nuanced than thresholding; more **interpretable & data-frugal** than CNNs.

---

## Slide 4 — Objectives
1. Robust preprocessing (noise + uneven lighting).
2. Extract geometric + textural features.
3. Mamdani FIS to **grade severity** from linguistic rules.
4. Fuzzy classifier for **defect type**.
5. **Beat conventional thresholding** on borderline defects.
6. Interactive web prototype.

---

## Slide 5 — System Architecture
`Image → Preprocess → Feature Extraction → Fuzzy Inference → Verdict`
- Preprocess: NL-means denoise · illumination correction · CLAHE
- Features: multi-channel saliency · shape · GLCM texture
- Fuzzy: severity FIS + type classifier → ACCEPT / NOTE / REJECT (+ severity, type)

---

## Slide 6 — Detection (saliency)
- Statistical threshold (mean + k·σ), **not** Otsu → quiet on clean surfaces.
- Three fused detectors:
  - white top-hat → bright scratches
  - black-hat → dark cracks/pits
  - local-texture anomaly → corrosion
- *(show `results/overlay_*.png` and the Detection tab)*

---

## Slide 7 — The Fuzzy Engine
- **Inputs:** size, contrast, roughness ∈ [0,1]
- **Output:** severity ∈ [0,100] → good / minor / moderate / critical
- Overlapping membership functions + **16 linguistic rules**
- Centroid defuzzification
- *(show the membership-function figure from the Fuzzy Engine tab)*

---

## Slide 8 — Example Rules
- IF contrast **low** AND roughness **smooth** → severity **good**
- IF roughness **rough** → severity **moderate** *(catches thin defects)*
- IF size **large** AND contrast **high** → severity **critical**

---

## Slide 9 — Results: Detection
| Metric | **Fuzzy** | Baseline |
|---|---|---|
| Accuracy | **0.81** | 0.44 |
| Recall | **0.79** | 0.25 |
| F1 | **0.86** | 0.40 |

*(show `results/comparison.png`)*

---

## Slide 10 — Results: The Decisive Test
**Borderline-defect recall**
- Fuzzy: **0.50**
- Baseline: **0.00**
➜ The binary method misses *every* borderline defect; fuzzy catches half.

---

## Slide 11 — Results: Type Classification
- Accuracy **0.67**; scratches & dents ≈ perfect.
- *(show `results/confusion_matrix.png`)*

---

## Slide 12 — Web Prototype
- Drag-and-drop upload → **background worker** → live progress.
- Dashboard: verdict gauge, preprocessing stages, detection overlay,
  **live fuzzy plots**, feature radar, baseline comparison,
  plain-language **explanation** of each decision.

---

## Slide 13 — Why Fuzzy Wins
- Graded memberships express the **ambiguous middle ground**.
- Every verdict is **explainable** (audit-friendly).
- Works with **little data**, no GPU.

---

## Slide 14 — Limitations & Future Work
- Validate on real datasets (NEU / MVTec-AD).
- Auto-tune membership functions (GA / ANFIS).
- Multi-defect localisation; live camera / in-line trigger.

---

## Slide 15 — Conclusion
A complete, working, **interpretable** prototype that fuses image processing
with fuzzy logic and **outperforms conventional thresholding** — especially on
the borderline defects that motivated the project. All objectives met.

---

## Slide 16 — Demo
`./run.sh web` → http://127.0.0.1:5000
Thank you — questions?
