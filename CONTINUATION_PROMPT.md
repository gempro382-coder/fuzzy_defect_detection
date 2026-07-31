# NeuraFuzz Inspect — Frontend Rebuild Continuation Prompt

You are continuing a frontend rebuild for **NeuraFuzz Inspect**, a hybrid ML + fuzzy-logic defect detection system. The user asked for "the world's best UI" with a **full 3D cinematic intro** showing engineers moving into a workshop, examining tiles for cracks and corrosion, followed by the main inspection interface.

---

## What Has Been Done So Far

### 1. **WebGL Engine (`web/static/js/nf-gl.js`)** — ✅ COMPLETE
   - **1745 lines**, dependency-free WebGL1 micro-engine
   - **Mat4/Vec3 maths**: column-major, perspective, lookAt, normal matrices
   - **Geometry builders**: `plane`, `quad`, `box`, `cylinder`, `sphere`, `torus`
   - **Surface shader with 4 modes**:
     - `0` plain (walls, figures, bench)
     - `1` procedural ceramic specimen — **cracks, corrosion, scratches revealed only by grazing light** (the core NDT principle)
     - `2` mapped photo texture with overlay cross-fade + scan bar
     - `3` cast concrete floor with control joints
   - **Raking-light physics**: spotlight returns `graze` param; defects gate on `lift = graze * reveal`
   - **Volumetrics**: light cones, floor pools, point-sprite dust
   - **Post chain**: bright-pass → 4-pass separable Gaussian blur → filmic composite (ACES shoulder, vignette, chromatic aberration, film grain, flash, scanlines)
   - **`Node` scene graph**: articulated rigs for engineer figures
   - **`Renderer.project(worldPoint)`**: projects 3D coords to CSS pixels for HTML reticles
   - **Graceful degradation**: returns `null` if WebGL unavailable; post disables if framebuffer incomplete

### 2. **Cinematic Script (`web/static/js/nf-intro.js`)** — ✅ COMPLETE (just written)
   - **655 lines**, real-time 4-shot sequence (16.4 seconds total)
   - **Shot A (0–4.3s)**: Roller shutter opens on Bay 04; two articulated engineer figures walk in under daylight wedge
   - **Shot B (4.3–8.1s)**: They arrive at the bench; overhead sodium lamp strikes on; coupons look fine under flat light
   - **Shot C (8.1–12.65s)**: One engineer lifts the handheld raking lamp and sweeps it across the hero coupon at **6° incidence** — crack network + oxide bloom + scoring **appear out of nowhere**. Three HTML reticles lock on with labels: "CRACK · branching · 0.42 mm", "OXIDE · Fe₂O₃ bloom · 9 %", "SCORING · directional · 3 µm"
   - **Shot D (12.65–16.4s)**: Pull back; everything falls away; title card resolves: "◈ NeuraFuzz Inspect"
   - **Camera path**: 8 keyframes, eased with `Ease.inOutSine` + handheld wobble
   - **Captions**: 4 lower-third slates with code + line, e.g. "BAY 04 · 06:30 · SHIFT 1 / Lot 8821 comes in for surface inspection."
   - **Colour palette** (from real NDT instruments):
     - `concrete: #3A3E42`, `wallPaint: #2B3238`, `steelDark: #20272D`
     - `sodium: #FFB03A` (overhead work-lamp)
     - `penetrant: #C8F135` (handheld raking lamp — fluorescent dye penetrant colour)
     - `phosphor: #4FD8E8`, `ceramic: #C3C0B4`, `hiVis: #D8E840`, `hardHat: #E8E2D6`
   - **Engineer rigs**: box/cylinder limbs on joint hierarchy; `poseEngineer()` drives walk cycle + head/lean/arm overrides
   - **Lamp rig**: free node in world space; `aimRot(dir)` solves Euler angles so local -Y points down beam axis
   - **Volumetrics**: daylight wedge, sodium cone + floor pool, rake beam (narrow cone at 6°), airborne dust sprites
   - **Reticles**: 3 HTML overlays projected via `Renderer.project()`, animated lock-on with `span()` easing
   - **Controls**: Esc/Enter/Space to skip; progress bar; `sessionStorage` + reduced-motion check
   - **API**: `NFIntro.play({ canvas, skipButton, bar, slate, slateCode, slateLine, title, reticleLayer, onDone })` returns controller or `null` if unavailable

### 3. **Existing Files** (read, not yet modified)
   - **`web/templates/index.html`** (195 lines) — current light SaaS theme markup. DOM IDs the new controller must bind to: `dropzone`, `fileInput`, `previewWrap`, `previewImg`, `analyzeBtn`, `clearBtn`, `progressPanel`, `bar`, `progressMsg`, `stagetrack`, `emptyState`, `errorState`, `errorMsg`, `results`, verdict/dial/pathcard/tab elements, figure `<img>` IDs for the 3D stage
   - **`web/static/css/style.css`** (176 lines) — **to be replaced**. Current palette: `--bg:#f4f6fb`, `--brand:#2563eb`, Inter font. This is the generic light template being replaced by the inspection-booth identity.
   - **`web/static/js/app.js`** (208 lines) — **to be rewritten**. Current controller: upload/drag-drop, sample tray, `/upload` → `/status/<id>` polling (400 ms) → `/result/<id>`, 7-stage progress track (`acquire`, `preprocess`, `features`, `ml`, `fusion`, `render`, `compile`), dial arcs (circumference 327, r=52), tab switching, verdict rendering. `})();` at the end is a marker for `build_static_preview.py`.
   - **`src/fusion_engine.py`** (read for data contract) — `DETECT_THRESHOLD = 0.45`, `W_ML_DETECT = 0.75`, `W_FUZZY_DETECT = 0.25`, `ACCEPT_MAX = 22`, `NOTE_MAX = 45`, moderate/critical split at **68**. JSON keys: `final{decision, defect_type, severity, grade, is_defect, confidence, p_defect_fused, agreement, needs_review}`, `fuzzy{severity_score, severity_label, defect_type, decision, type_scores, fuzzy_inputs, p_defect}`, `fusion{p_ml_defect, p_fuzzy_defect, p_fused, weights{ml,fuzzy}, type_scores, safety_boost_applied}`.
   - **`src/inference_pipeline.py`** (348 lines) — matplotlib figures use **light** palette (`_BG="#ffffff"`, `_FG="#1f2933"`, `_ACCENT="#2563eb"`) that will clash with dark booth UI → task #6. Progress fires at 5/20/42/58/70/82/95/100.
   - **`web/app.py`** — `TEST_DIR = ROOT/data/test` **does not exist**; only `data/custom_samples/{good,scratch,crack,dent,corrosion}/` exists → task #6.
   - **`web/build_static_preview.py`** — inlines CSS/JS by regex, defaults to `"../data/test/dent/*severe*.png"` (will fail), assumes single-file JS → task #6.

---

## Design Identity (from `frontend-design` skill)

**SUBJECT**: Raking-light NDT (non-destructive testing) inspection booth. Surface defects (cracks, corrosion, scoring) only appear when light skims a surface at a low angle — flat overhead light hides them. The interface must embody this reveal.

**PALETTE** (4 named values from the real material world, NOT AI defaults):
- **Graphite** `#1E252B` (booth walls, dark ground)
- **Sodium** `#FFB03A` (overhead work-lamp, warm accent)
- **Penetrant** `#C8F135` (fluorescent dye, used in liquid penetrant testing)
- **Phosphor** `#4FD8E8` (magnetic particle glow, cold accent)

Rejected: cream `#F4F1EA` + terracotta `#D97757` (AI default #1), black + acid-green (AI default #2), broadsheet hairlines (AI default #3).

**TYPOGRAPHY**:
- **Display**: Archivo Expanded (800 weight, uppercase, wide tracking) — technical drawing lettering
- **Body**: IBM Plex Sans (400/600) — engineered humanist
- **Data**: IBM Plex Mono (500) — precision readouts

Rejected: high-contrast serif, single monospace stack.

**LAYOUT**: The interface is a **pipeline ladder** — 9 stages vertically numbered S1–S9 (Acquire → Preprocess → Features → Detect → ML → Fuzzy → Fuse → Grade → Report). The numbering is justified: the Python pipeline genuinely **is** an ordered sequence, and the stage names come from `inference_pipeline.py` callbacks.

**SIGNATURE ELEMENT**: The **verdict slab** — a large raised panel with a slight 3D tilt (CSS `transform: perspective(1200px) rotateX(1.5deg)`), metallic sheen via a linear gradient overlay, and a **raking-light reflection** that sweeps across it when the job completes. The slab embodies the subject: a surface that only reveals itself under oblique light.

**3D SPECIMEN STAGE**: A live WebGL canvas showing the uploaded image mapped onto a tilted specimen plate on a bench, lit by a user-controlled raking lamp. The lamp angle is interactive (drag to orbit); a **scan sweep** travels across the plate bound to job progress; on result the plate cross-fades to the annotated overlay and tints to the verdict colour (green/amber/red).

---

## Remaining Tasks

### **TASK #3: Build the live 3D specimen stage (`nf-stage.js`)**
   **STATUS**: Not started
   **DESCRIPTION**: The WebGL canvas that appears in the results section after the intro. Shows:
   - Inspection bench (reuse geometry from intro)
   - Tilted specimen plate (30–40° from horizontal) textured with the uploaded/sample image
   - **Orbit controls**: drag to rotate camera around the specimen
   - **User-controlled rake lamp**: slider or drag to adjust incident angle (2°–20°)
   - **Scan sweep**: a bright bar that travels left→right across the plate, bound to `progress` (0–100). At progress=100 the sweep vanishes and the plate cross-fades to the annotated overlay image.
   - **Verdict tinting**: once the result arrives, the plate edges glow green (accept), amber (note), or red (reject)
   - **Fallback**: if WebGL unavailable, show a `<canvas>` with a 2D-drawn static plate + the image, or fall back to a plain `<img>`
   - **API**: `NFStage.create({ canvas, onOrbit })` returns controller with `.setImage(url)`, `.setProgress(p)`, `.setOverlay(url, verdict)`, `.setLampAngle(deg)`, `.resize()`

### **TASK #4: Rewrite markup + design system (`index.html` + `style.css`)**
   **STATUS**: Not started
   **DESCRIPTION**:
   - **Keep all existing DOM IDs** from the current `index.html` — the new controller depends on them
   - **New markup** for:
     - **Intro layer**: `<div id="introLayer">` with `<canvas id="introCanvas">`, `<div id="introReticles">`, `<div id="introSlate">`, `<div id="introTitle">`, `<button id="introSkip">SKIP</button>`, `<div id="introBar">`
     - **S1–S9 ladder** (replaces `#stagetrack`): a vertical rail with 9 numbered markers, each lighting up as its stage completes
     - **Verdict slab** (replaces/enhances `#verdictPanel`): large raised panel with perspective tilt, metallic sheen, raking-light sweep animation on result
     - **3D stage canvas** in results: `<canvas id="stageCanvas">` + `<div id="stageFallback">` (for the image if WebGL unavailable)
     - **Severity dial ticks** at 22/45/68 (the real grade-band constants from `FusionConfig`)
     - **Path meters** with a threshold marker at 0.45 (the real `DETECT_THRESHOLD`)
   - **CSS**:
     - Dark booth palette: `--graphite: #1E252B`, `--sodium: #FFB03A`, `--penetrant: #C8F135`, `--phosphor: #4FD8E8`
     - Archivo Expanded 800 for `<h1>`, `.verdict-tag`, stage numbers
     - IBM Plex Sans 400/600 for body, IBM Plex Mono 500 for data
     - **Raking-light sheen** on panels: `background: linear-gradient(135deg, var(--graphite) 0%, lighten(--graphite, 8%) 50%, var(--graphite) 100%)` with a `:before` pseudo-element `linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent)` animated via `@keyframes rake-sweep { from { transform: translateX(-100%); } to { transform: translateX(200%); } }`
     - **Canvas dials**: hand-drawn with `<canvas>` instead of SVG, so the arc can be drawn with a custom gradient
     - **Responsive**: breakpoints at 1000px (stack grid), 760px (stack dual cards), 680px (stack g2)
     - **Keyboard focus**: all interactive elements have visible `:focus-visible` rings
     - **Reduced motion**: `@media (prefers-reduced-motion: reduce)` disables intro, sweep animations, transitions > 0.2s
   - **Fonts**: use `@import url('https://fonts.googleapis.com/css2?family=Archivo+Expanded:wght@800&family=IBM+Plex+Mono:wght@500&family=IBM+Plex+Sans:wght@400;600&display=swap');` at the top of `style.css`

### **TASK #5: Rewrite the front-end controller (`app.js`)**
   **STATUS**: Not started
   **DESCRIPTION**:
   - **Intro sequence**:
     - On page load, check `sessionStorage.getItem('nf-intro-seen')` and `window.matchMedia('(prefers-reduced-motion: reduce)').matches`
     - If not seen and motion OK, show `#introLayer`, call `NFIntro.play({ ... })`, hide `#introLayer` and show interface on `onDone`
     - If seen or motion reduced, skip straight to interface
     - Set `sessionStorage.setItem('nf-intro-seen', '1')` after first play
   - **Upload/drag-drop**: same as current `app.js`, but also call `stage.setImage(dataURL)` once preview loads
   - **Sample tray**: fetch `/samples`, render coupon cards with thumbnails, call `stage.setImage(url)` + `startJob()` on click
   - **Job lifecycle**:
     - `startJob(url, opts)` → show `#progressPanel`, hide others, call `stage.setProgress(0)`, start S1–S9 ladder animation
     - `poll(id)` every 400 ms → `setProgress(st.progress)`, `stage.setProgress(st.progress)`, mark completed stages on ladder
     - `getResult(id)` → hide `#progressPanel`, show `#results`, call `render(rep)`, trigger verdict slab rake-sweep, call `stage.setOverlay(rep.images.overlay, rep.verdict_class)`
   - **S1–S9 ladder**: 9 `<div class="stage-marker" data-stage="S1">` with `::before { content: 'S1'; }`. As each stage completes, add `.done` class (glow effect). The 9 stages map to:
     - S1 = Acquire, S2 = Preprocess, S3 = Features, S4 = Detect, S5 = ML, S6 = Fuzzy, S7 = Fuse, S8 = Grade, S9 = Report
     - The backend `stages` array contains `[{stage: "acquire", ...}, ...]` — map `"acquire"` → `"S1"`, etc.
   - **Verdict slab rake-sweep**: on result, add `.sweep` class to `#verdictPanel`, which triggers the `:before` animation once
   - **Canvas dials**: replace the SVG dials with `<canvas>` elements; draw arcs with `ctx.arc()` + custom gradient for severity (green→amber→red), solid phosphor for confidence
   - **Path meters**: three horizontal bars (ML / Fuzzy / Fused) with a vertical marker at x=45% labelled "0.45" (the detection threshold)
   - **3D stage controls**: `<input type="range" id="lampAngle" min="2" max="20" value="6">` calls `stage.setLampAngle(val)`; drag on canvas calls `stage.orbit(dx, dy)`
   - **Tabs, tables, figures**: same as current `app.js`
   - **End marker**: keep `})();` on the last line so `build_static_preview.py` can parse it

### **TASK #6: Align backend, figures and preview builder**
   **STATUS**: Not started
   **DESCRIPTION**:
   - **Retheme matplotlib figures** in `src/inference_pipeline.py`:
     - Change `_BG = "#1E252B"`, `_FG = "#E4E7EB"`, `_ACCENT = "#4FD8E8"` (booth palette)
     - Update `plt.style.use('dark_background')` and `rcParams` to match
   - **Fix `/samples` and `/sample_image` in `web/app.py`**:
     - Change `TEST_DIR` to `ROOT / "data" / "custom_samples"`
     - Glob all subdirs: `good/`, `scratch/`, `crack/`, `dent/`, `corrosion/`
     - Return `[{ filename: relative_path, label: subdir_name, band: "custom" }, ...]`
   - **Expose fusion threshold** via `/system`:
     - Add `"fusion": { "detect_threshold": fusion_engine.DETECT_THRESHOLD }` to the `/system` JSON response
     - The frontend reads this to draw the 0.45 marker on the path meters
   - **Update `web/build_static_preview.py`**:
     - Inline **3 JS files** in order: `nf-gl.js`, `nf-intro.js`, `nf-stage.js`, `app.js`
     - Inline `style.css`
     - Change default image glob to `"../data/custom_samples/dent/*.png"` (one that exists)
     - Regex: find the closing `})();` in `app.js` to locate the injection point for the auto-run code

---

## Code Conventions (already established in tasks #1–2)

- **Colour names**: every hex is named for the real-world material/instrument it represents (`sodium`, `penetrant`, `phosphor`, `ceramic`, `hiVis`, `hardHat`, `trouser`, `boot`, `lampBody`)
- **Easing**: use `Ease.inOutSine`, `Ease.outCubic`, `Ease.inQuad`, etc. from `nf-gl.js`; never raw `t` where motion should ease
- **Camera keyframes**: array of `{ t, eye: [x,y,z], at: [x,y,z], fov }`, sampled with `span(t, a, b, easing)`
- **Timeline anchors**: one object `T = { total: ..., cutB: ..., rake0: ..., rake1: ... }` so shot timings read like a shot list
- **Node rig pattern**: each joint is a `Node` at its pivot; the visual (box/cylinder) is an offset child. Rotate the joint node to swing the limb. Use `node.add(child).set(x,y,z).sized(sx,sy,sz).rotate(rx,ry,rz)` chaining.
- **Material properties**: `{ color, roughness, specular, emissive?, mode?, uvScale?, defect?, tangent?, bitangent?, bump? }`
- **Volumetric draw**: `R.drawVolume(mesh, worldMatrix, colourRGB, alpha, fadeMode)` where `fadeMode` 0=cone, 1=pool
- **Dust**: `R.createDust(count, boxSize, drift)` → `R.drawDust(dustObj, colourRGB, brightness)`
- **Post settings**: `R.post.exposure`, `.bloom`, `.bloomThreshold`, `.vignette`, `.chroma`, `.grain`, `.scanlines`, `.flash`, `.flashColor`
- **Renderer methods**: `R.setCamera(eye, at, up, fov, near, far)`, `R.beginScene()`, `R.drawSurface(mesh, worldMat, material)`, `R.endScene()`, `R.project(worldPoint)` → `{x, y, depth}` or `null`
- **Comments**: `/* ---- section ---- */` for major blocks, `/* ... */` for inline notes, no `//`

---

## Next Steps

**Continue from where the summary left off:**

1. Mark **task #2 as completed** (the intro cinematic is done)
2. Mark **task #3 as in_progress**
3. **Write `web/static/js/nf-stage.js`** (~400–600 lines):
   - Build the bench + specimen plate geometry (tilt 35° on X axis)
   - Texture the plate with the user image (load via `Image`, upload to WebGL texture)
   - Orbit controls: track mouse/touch drag, update camera `theta`/`phi` around a fixed target
   - User-controlled lamp: bind to a global `lampAngle` var, position light to maintain that incidence angle relative to plate normal
   - Scan sweep: a vertical bright bar (additive quad) that lerps `x = lerp(-plateWidth/2, plateWidth/2, progress/100)`
   - Overlay cross-fade: on result, upload the annotated image to a second texture, lerp `uMixOverlay` from 0→1 over 1.2s
   - Verdict glow: add an emissive rim to the plate edges, colour = verdict class
   - Fallback: `if (!NFGL.supported) { /* draw 2D static */ }`
   - API: `window.NFStage = { create, supported: NFGL.supported }`
4. Once #3 is done, move to **task #4** (markup + CSS)
5. Then **task #5** (controller)
6. Finally **task #6** (backend alignment)
7. At the end, **audit all code** with review agents (since the shell is unavailable for running the server)

---

## Key Constraints

- **No Python execution**: the Bash tool is completely non-functional in this session. We cannot run `python app.py`, cannot test in a browser, cannot verify the Flask server. Build everything with file tools and plan to audit at the end.
- **No dependencies**: the 3D engine must work offline on a lab machine with no internet. No three.js, no CDN imports (fonts are OK via Google Fonts `@import` in CSS, as that's a static link the preview builder can inline).
- **Exact data contract**: the frontend displays real numbers from `fusion_engine.py` and `inference_pipeline.py`. Never invent verdict thresholds or stage names — they come from the Python source.
- **All existing DOM IDs must remain**: the current `app.js` binds to 40+ element IDs. The new HTML must supply every one of them, even if the markup around them changes.
- **Intro is optional**: if `sessionStorage` says it was seen, or if `prefers-reduced-motion`, skip straight to the interface. The interface must work standalone.
- **Write very long code**: the user explicitly asked for this. Each file should be as complete and self-contained as possible. Inline the full shader sources, full geometry builders, full walk cycle, full camera path — no "… rest of the code here" placeholders.

---

## How to Continue

Read this entire prompt, then:

1. Call `TaskUpdate` to mark task #2 (`completed`) and task #3 (`in_progress`)
2. Write `web/static/js/nf-stage.js` in full
3. Move through tasks #4, #5, #6 in order
4. At the end, report completion and recommend spawning review agents to audit the code

The goal: when a user opens `index.html` in a browser, they see the 16-second cinematic of engineers in the inspection bay, the rack reveal under raking light, then the ◈ NeuraFuzz Inspect interface appears with a live 3D specimen stage, a dark booth aesthetic, and the world's best defect-inspection UI.
