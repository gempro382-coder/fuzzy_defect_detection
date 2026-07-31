/* ==========================================================================
   NeuraFuzz Inspect — interface controller
   --------------------------------------------------------------------------
   Upload or pick a specimen → background job → staged polling → the inspection
   report, animated in as an instrument would settle: the severity needle on a
   damped spring, readouts counting up, the consensus marks sliding onto a
   shared probability axis.
   ========================================================================== */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ======================================================================
     MOTION PRIMITIVES
     ====================================================================== */

  /* expo-out: fast departure, long settle — reads as mechanical, not bouncy */
  function easeOut(t) { return t === 1 ? 1 : 1 - Math.pow(2, -10 * t); }

  function tween(dur, onStep, onDone) {
    if (REDUCED) { onStep(1); if (onDone) onDone(); return; }
    var t0 = performance.now();
    function frame(now) {
      var t = Math.min((now - t0) / dur, 1);
      onStep(easeOut(t));
      if (t < 1) requestAnimationFrame(frame); else if (onDone) onDone();
    }
    requestAnimationFrame(frame);
  }

  /* A critically-under-damped spring, so the needle overshoots once and settles
     the way a real moving-coil meter does. */
  function spring(from, to, onStep, opts) {
    opts = opts || {};
    if (REDUCED) { onStep(to); return; }
    var stiffness = opts.stiffness || 140, damping = opts.damping || 15;
    var x = from, v = 0, last = performance.now();
    function frame(now) {
      var dt = Math.min((now - last) / 1000, 1 / 30); last = now;
      /* substep for stability */
      for (var i = 0; i < 3; i++) {
        var a = -stiffness * (x - to) - damping * v;
        v += a * (dt / 3); x += v * (dt / 3);
      }
      onStep(x);
      if (Math.abs(x - to) > 0.02 || Math.abs(v) > 0.02) requestAnimationFrame(frame);
      else onStep(to);
    }
    requestAnimationFrame(frame);
  }

  function countUp(el, to, decimals, suffix) {
    if (!el) return;
    var d = decimals || 0;
    tween(900, function (t) {
      el.textContent = (to * t).toFixed(d) + (suffix || "");
    }, function () { el.textContent = to.toFixed(d) + (suffix || ""); });
  }

  /* scroll-reveal */
  var revealIO = ('IntersectionObserver' in window) ? new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); revealIO.unobserve(e.target); } });
  }, { rootMargin: '0px 0px -8% 0px', threshold: .06 }) : null;

  function observeReveals(root) {
    var nodes = (root || document).querySelectorAll('.reveal:not(.in)');
    if (!revealIO) { Array.prototype.forEach.call(nodes, function (n) { n.classList.add('in'); }); return; }
    Array.prototype.forEach.call(nodes, function (n) { revealIO.observe(n); });
  }

  /* ======================================================================
     ELEMENTS
     ====================================================================== */
  var dropzone = $("dropzone"), fileInput = $("fileInput");
  var previewWrap = $("previewWrap"), previewImg = $("previewImg");
  var analyzeBtn = $("analyzeBtn"), clearBtn = $("clearBtn");
  var progressPanel = $("progressPanel"), bar = $("bar"), progressMsg = $("progressMsg"), progressPct = $("progressPct");
  var stagetrack = $("stagetrack"), ladderFill = $("ladderFill");
  var emptyState = $("emptyState"), errorState = $("errorState"), errorMsg = $("errorMsg");
  var results = $("results"), engineState = $("engineState");

  var selectedFile = null, pollTimer = null, stage = null;
  var detectThreshold = 0.45;

  var STAGES = {
    acquire: "S1", preprocess: "S2", features: "S3", detect: "S4",
    ml: "S5", fuzzy: "S6", fusion: "S7", grade: "S8", compile: "S9", report: "S9"
  };
  var STAGE_NAMES = [
    ["S1", "Acquire"], ["S2", "Preprocess"], ["S3", "Measure features"], ["S4", "Detect regions"],
    ["S5", "Model inference"], ["S6", "Fuzzy inference"], ["S7", "Fuse"], ["S8", "Grade"], ["S9", "Compile report"]
  ];

  /* ======================================================================
     SYSTEM METADATA
     ====================================================================== */
  fetch("/system").then(function (r) { return r.json(); }).then(function (s) {
    if (s.model) {
      $("mMdl").textContent = s.model.selected_model || "—";
      var b = s.model.binary_defect_detection;
      if (b) {
        $("mAcc").textContent = (b.accuracy * 100).toFixed(1) + "%";
        $("mAuc").textContent = Number(b.roc_auc).toFixed(3);
      }
    }
    if (s.fusion && s.fusion.detect_threshold) detectThreshold = s.fusion.detect_threshold;
    layoutAxisStatics();
  }).catch(function () { layoutAxisStatics(); });

  function layoutAxisStatics() {
    var p = (detectThreshold * 100).toFixed(1) + "%";
    $("consThresh").style.left = p;
    $("zonePass").style.width = p;
    $("zoneFail").style.width = (100 - detectThreshold * 100).toFixed(1) + "%";
    $("threshLbl").textContent = "threshold " + detectThreshold.toFixed(2);
  }

  /* ======================================================================
     3D STAGE + INTRO
     ====================================================================== */
  function initStage() {
    if (window.NFStage) {
      try { stage = NFStage.create({ canvas: $("stageCanvas"), onOrbit: function () {} }); }
      catch (e) { stage = null; }
    }
  }

  var seenIntro = false;
  try { seenIntro = sessionStorage.getItem('nf-intro-seen'); } catch (e) {}

  function endIntro() {
    var l = $("introLayer");
    if (l) l.style.display = 'none';
    document.body.style.overflow = '';
    try { sessionStorage.setItem('nf-intro-seen', '1'); } catch (e) {}
    initStage();
    observeReveals();
  }

  if (seenIntro || REDUCED || !window.NFIntro) {
    endIntro();
  } else {
    document.body.style.overflow = 'hidden';
    var ctrl = NFIntro.play({
      canvas: $("introCanvas"), skipButton: $("introSkip"), bar: $("introBar"),
      slate: $("introSlate"), slateCode: $("introSlateCode"), slateLine: $("introSlateLine"),
      title: $("introTitle"), reticleLayer: $("introReticles"), onDone: endIntro
    });
    if (!ctrl) endIntro();
  }

  /* ======================================================================
     SPECIMEN INTAKE
     ====================================================================== */
  dropzone.addEventListener("click", function () { fileInput.click(); });
  dropzone.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  dropzone.addEventListener("dragover", function (e) { e.preventDefault(); dropzone.classList.add("drag"); });
  dropzone.addEventListener("dragleave", function () { dropzone.classList.remove("drag"); });
  dropzone.addEventListener("drop", function (e) {
    e.preventDefault(); dropzone.classList.remove("drag");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", function (e) { if (e.target.files.length) handleFile(e.target.files[0]); });

  clearBtn.addEventListener("click", function () {
    selectedFile = null; fileInput.value = "";
    previewWrap.hidden = true; analyzeBtn.disabled = true;
    document.querySelectorAll('.sample.is-active').forEach(function (n) { n.classList.remove('is-active'); });
  });

  function humanSize(n) {
    return n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(0) + " KB" : (n / 1048576).toFixed(1) + " MB";
  }

  function handleFile(f) {
    if (!f.type.indexOf || f.type.indexOf("image/") !== 0) return showError("That file is not an image. Choose a PNG, JPG, BMP or TIFF.");
    selectedFile = f;
    $("fileName").textContent = f.name;
    $("fileSize").textContent = humanSize(f.size);
    var rd = new FileReader();
    rd.onload = function (e) {
      previewImg.src = e.target.result;
      previewWrap.hidden = false;
      if (stage) stage.setImage(e.target.result);
    };
    rd.readAsDataURL(f);
    analyzeBtn.disabled = false;
  }

  analyzeBtn.addEventListener("click", function () {
    if (!selectedFile) return;
    var fd = new FormData(); fd.append("image", selectedFile);
    startJob("/upload", { method: "POST", body: fd });
  });

  /* ======================================================================
     REFERENCE LIBRARY
     ====================================================================== */
  fetch("/samples").then(function (r) { return r.json(); }).then(function (d) {
    var el = $("samples");
    if (!d.samples || !d.samples.length) { el.innerHTML = '<p class="muted sm">No reference specimens installed.</p>'; return; }
    el.innerHTML = "";
    $("sampleCount").textContent = d.samples.length + " specimens";
    d.samples.forEach(function (s) {
      var btn = document.createElement("button");
      btn.className = "sample"; btn.type = "button";
      btn.title = "Inspect reference specimen: " + s.label;
      var url = "/sample_image/" + encodeURIComponent(s.filename);
      btn.innerHTML = '<img src="' + url + '" alt=""><span>' + s.label + '</span>';
      btn.addEventListener("click", function () {
        document.querySelectorAll('.sample.is-active').forEach(function (n) { n.classList.remove('is-active'); });
        btn.classList.add('is-active');
        selectedFile = null; analyzeBtn.disabled = true;
        previewImg.src = url; previewWrap.hidden = false;
        $("fileName").textContent = s.filename; $("fileSize").textContent = "reference";
        if (stage) stage.setImage(url);
        startJob("/analyze_sample", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: s.filename })
        });
      });
      el.appendChild(btn);
    });
  }).catch(function () {
    var el = $("samples");
    if (el && !el.querySelector(".sample")) el.innerHTML = '<p class="muted sm">Reference library unavailable.</p>';
  });

  /* ======================================================================
     JOB LIFECYCLE
     ====================================================================== */
  function startJob(url, opts) {
    clearTimeout(pollTimer);
    emptyState.hidden = true; errorState.hidden = true; results.hidden = true;
    progressPanel.hidden = false;
    engineState.textContent = "Inspecting";
    setProgress(0, "Submitting specimen…");
    buildStages();
    if (stage) stage.setProgress(0);
    progressPanel.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "nearest" });
    fetch(url, opts).then(function (r) { return r.json(); }).then(function (d) {
      if (d.error) throw new Error(d.error);
      poll(d.job_id);
    }).catch(function (e) { showError(e.message); });
  }

  function poll(id) {
    fetch("/status/" + id).then(function (r) { return r.json(); }).then(function (st) {
      if (st.error) throw new Error(st.error);
      setProgress(st.progress, st.message);
      markStages(st.stages || []);
      if (stage) stage.setProgress(st.progress);
      if (st.status === "done") getResult(id);
      else if (st.status === "error") showError(st.message || "The pipeline stopped before it finished.");
      else pollTimer = setTimeout(function () { poll(id); }, 400);
    }).catch(function (e) { showError(e.message); });
  }

  function getResult(id) {
    fetch("/result/" + id).then(function (r) { return r.json(); }).then(function (rep) {
      if (rep.error) throw new Error(rep.error);
      render(rep);
    }).catch(function (e) { showError(e.message); });
  }

  function setProgress(p, m) {
    bar.style.width = p + "%";
    progressPct.textContent = Math.round(p) + "%";
    progressMsg.textContent = m || "";
  }

  function buildStages() {
    stagetrack.innerHTML = '<div class="ladder-fill" id="ladderFill"></div>' +
      STAGE_NAMES.map(function (s) {
        return '<div class="stage-marker" data-stage="' + s[0] + '"><span>' + s[1] + '</span><span class="st-t"></span></div>';
      }).join("");
    ladderFill = $("ladderFill");
  }

  function markStages(stages) {
    var done = {}, t0 = stages.length ? stages[0].t : 0;
    stages.forEach(function (s) { if (STAGES[s.stage]) done[STAGES[s.stage]] = s; });
    var markers = stagetrack.querySelectorAll(".stage-marker");
    var lastDone = -1;
    Array.prototype.forEach.call(markers, function (el, i) {
      var hit = done[el.dataset.stage];
      el.classList.remove("active");
      if (hit) {
        el.classList.add("done");
        lastDone = i;
        var t = el.querySelector(".st-t");
        if (t && hit.t != null && !t.textContent) t.textContent = "+" + (hit.t - t0).toFixed(2) + "s";
      }
    });
    if (markers[lastDone + 1]) markers[lastDone + 1].classList.add("active");
    if (ladderFill && markers.length) {
      var frac = (lastDone + 1) / markers.length;
      ladderFill.style.height = (frac * 100) + "%";
    }
  }

  function showError(m) {
    clearTimeout(pollTimer);
    progressPanel.hidden = true; results.hidden = true; emptyState.hidden = true;
    errorState.hidden = false;
    errorMsg.textContent = m || "Something went wrong. Try another image.";
    engineState.textContent = "Engine ready";
  }

  /* ======================================================================
     SEVERITY RULER
     ====================================================================== */
  var BANDS = [[22, "none"], [45, "minor"], [68, "moderate"], [101, "severe"]];

  function setGauge(severity) {
    var needle = $("gaugeNeedle"), fill = $("gaugeFill"), val = $("sevVal");
    var band = "severe";
    for (var i = 0; i < BANDS.length; i++) { if (severity < BANDS[i][0]) { band = BANDS[i][1]; break; } }
    $("gaugeBandLabel").textContent = band;

    needle.style.left = "0%";
    fill.style.width = "0%";
    spring(0, severity, function (v) {
      var c = Math.max(0, Math.min(100, v));
      needle.style.left = c + "%";
      fill.style.width = c + "%";
      val.textContent = Math.round(c);
    }, { stiffness: 120, damping: 13 });
  }

  /* ======================================================================
     CONSENSUS AXIS
     ====================================================================== */
  function placeMark(id, valueId, p) {
    var el = $(id);
    el.style.left = (p * 100).toFixed(2) + "%";
    var lbl = el.querySelector(".lbl");
    /* keep the label inside the plate at the extremes */
    lbl.style.transform = p < .1 ? "translateX(0)" : p > .9 ? "translateX(-100%)" : "translateX(-50%)";
    var em = $(valueId);
    tween(900, function (t) { em.textContent = (p * t).toFixed(3); },
          function () { em.textContent = p.toFixed(3); });
  }

  function renderConsensus(ml, fz, fu, agreement) {
    var vals = [ml, fz, fu];
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);

    if (REDUCED) {
      placeMark("markMl", "markMlV", ml); placeMark("markFz", "markFzV", fz); placeMark("markFu", "markFuV", fu);
    } else {
      /* stagger so the eye follows ML → fuzzy → fused */
      setTimeout(function () { placeMark("markMl", "markMlV", ml); }, 60);
      setTimeout(function () { placeMark("markFz", "markFzV", fz); }, 200);
      setTimeout(function () { placeMark("markFu", "markFuV", fu); }, 340);
    }

    var br = $("spreadBracket");
    br.hidden = false;
    br.style.left = (lo * 100).toFixed(2) + "%";
    br.style.width = ((hi - lo) * 100).toFixed(2) + "%";
    $("spreadLbl").textContent = "spread " + (hi - lo).toFixed(3);

    $("agreeNote").textContent = agreement ? "paths agree" : "paths disagree";
  }

  /* ======================================================================
     WIPE COMPARE
     ====================================================================== */
  (function initWipe() {
    var wipe = $("wipe"), clip = $("wipeClip"), handle = $("wipeHandle");
    if (!wipe) return;
    var pct = 50, dragging = false;

    function apply(p) {
      pct = Math.max(0, Math.min(100, p));
      clip.style.clipPath = "inset(0 " + (100 - pct) + "% 0 0)";
      handle.style.left = pct + "%";
      handle.setAttribute("aria-valuenow", Math.round(pct));
    }
    function fromEvent(e) {
      var r = wipe.getBoundingClientRect();
      var x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      apply((x / r.width) * 100);
    }
    handle.addEventListener("pointerdown", function (e) {
      dragging = true; handle.setPointerCapture(e.pointerId); e.preventDefault();
    });
    window.addEventListener("pointermove", function (e) { if (dragging) fromEvent(e); });
    window.addEventListener("pointerup", function () { dragging = false; });
    wipe.addEventListener("pointerdown", function (e) { if (e.target !== handle) fromEvent(e); });
    handle.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") { apply(pct - 4); e.preventDefault(); }
      if (e.key === "ArrowRight") { apply(pct + 4); e.preventDefault(); }
      if (e.key === "Home") { apply(0); e.preventDefault(); }
      if (e.key === "End") { apply(100); e.preventDefault(); }
    });
    apply(50);

    /* on a fresh result, sweep the divider across once so it is obvious the
       annotated layer is hiding underneath */
    window.__wipeIntro = function () {
      if (REDUCED) { apply(50); return; }
      apply(100);
      setTimeout(function () {
        tween(1100, function (t) { apply(100 - 50 * t); });
      }, 260);
    };
  })();

  /* ======================================================================
     TABS
     ====================================================================== */
  var tabInk = $("tabInk");
  function moveInk(btn) {
    if (!btn || !tabInk) return;
    tabInk.style.width = btn.offsetWidth + "px";
    tabInk.style.transform = "translateX(" + btn.offsetLeft + "px)";
  }
  document.querySelectorAll(".tab").forEach(function (t) {
    t.addEventListener("click", function () { activate(t.dataset.tab); });
  });
  function activate(name) {
    var active = null;
    document.querySelectorAll(".tab").forEach(function (t) {
      var on = t.dataset.tab === name;
      t.classList.toggle("active", on);
      if (on) active = t;
    });
    document.querySelectorAll(".tab-panel").forEach(function (p) {
      p.classList.toggle("active", p.dataset.panel === name);
    });
    moveInk(active);
    if (name === "overview" && stage) stage.resize();
    if (name === "ml") growTableBars();
  }
  window.addEventListener("resize", function () { moveInk(document.querySelector(".tab.active")); });

  function growTableBars() {
    document.querySelectorAll("#mlTable td.barcell i").forEach(function (i) {
      i.style.width = i.dataset.w + "%";
    });
  }

  /* ======================================================================
     RENDER
     ====================================================================== */
  function render(rep) {
    progressPanel.hidden = true;
    results.hidden = false;
    engineState.textContent = "Inspection complete";

    var f = rep.final, ml = rep.ml, fz = rep.fuzzy, fu = rep.fusion;

    /* --- booth --- */
    if (stage) {
      stage.setOverlay(rep.images.overlay, rep.verdict_class);
      if (f.bboxes && f.bboxes.length && typeof stage.playSweep === "function") stage.playSweep(rep);
    }

    /* --- verdict plate --- */
    var vp = $("verdictPanel");
    vp.className = "panel verdict " + rep.verdict_class;
    $("verdictTag").textContent = f.decision;
    $("verdictType").textContent = f.defect_type === "none" ? "No defect found" : f.defect_type;
    $("verdictMeta").textContent = "grade " + f.grade + " · fused P(defect) " + f.p_defect_fused.toFixed(3);

    var badges = [];
    badges.push('<span class="pillbadge ' + (f.agreement ? "ok" : "review") + '">' +
      (f.agreement ? "paths agree" : "paths disagree") + '</span>');
    if (f.needs_review) badges.push('<span class="pillbadge review">flagged for review</span>');
    if (fu.safety_boost_applied) badges.push('<span class="pillbadge">safety boost applied</span>');
    $("verdictBadges").innerHTML = badges.join("");

    var conf = f.confidence * 100;
    countUp($("confVal"), conf, 0);
    $("confMeter").style.width = "0%";
    setTimeout(function () { $("confMeter").style.width = conf + "%"; }, 60);
    setGauge(f.severity);

    /* --- consensus --- */
    renderConsensus(ml.p_defect, fz.p_defect, fu.p_fused, f.agreement);
    $("consensusFoot").innerHTML =
      '<span class="pillbadge">ml weight ' + (fu.weights.ml * 100).toFixed(0) + '%</span>' +
      '<span class="pillbadge">fuzzy weight ' + (fu.weights.fuzzy * 100).toFixed(0) + '%</span>' +
      '<span class="pillbadge">model class ' + ml.predicted_class + '</span>' +
      '<span class="pillbadge">fuzzy severity ' + fz.severity_score.toFixed(1) + '</span>';

    /* --- evidence --- */
    $("imgOriginal").src = rep.images.original;
    $("imgOverlay").src = rep.images.overlay;
    $("imgStages").src = rep.images.stage_panel;
    $("imgDetection").src = rep.images.detection_panel;
    $("imgMlProba").src = rep.figures.ml_proba;
    $("imgMembership").src = rep.figures.membership;
    $("imgFusion").src = rep.figures.fusion_bar;
    if (window.__wipeIntro) window.__wipeIntro();

    /* --- narrative, revealed line by line --- */
    var ex = $("explanation");
    ex.innerHTML = "<h3>How this decision was reached</h3>" +
      rep.explanation.map(function (l) { return "<p>" + l + "</p>"; }).join("");
    var lines = ex.querySelectorAll("p");
    Array.prototype.forEach.call(lines, function (p, i) {
      if (REDUCED) { p.classList.add("in"); return; }
      setTimeout(function () { p.classList.add("in"); }, 120 + i * 90);
    });

    $("timings").innerHTML = Object.keys(rep.timings).map(function (k) {
      return '<span class="t">' + k + ' <b>' + rep.timings[k] + 's</b></span>';
    }).join("");

    /* --- model table, top class highlighted --- */
    var entries = Object.keys(ml.proba).map(function (k) { return [k, ml.proba[k]]; })
      .sort(function (a, b) { return b[1] - a[1]; });
    var rows = "<thead><tr><th>Class</th><th></th><th style='text-align:right'>Probability</th></tr></thead><tbody>";
    entries.forEach(function (e, i) {
      var pctv = (e[1] * 100);
      rows += '<tr class="' + (i === 0 ? "is-top" : "") + '">' +
        '<td style="text-transform:capitalize">' + e[0] + '</td>' +
        '<td class="barcell"><i data-w="' + pctv.toFixed(1) + '"></i></td>' +
        '<td class="num">' + pctv.toFixed(1) + '%</td></tr>';
    });
    $("mlTable").innerHTML = rows + "</tbody>";

    /* --- fusion breakdown --- */
    $("fusionBox").innerHTML =
      kv("ML weight", (fu.weights.ml * 100).toFixed(0) + "%") +
      kv("Fuzzy weight", (fu.weights.fuzzy * 100).toFixed(0) + "%") +
      kv("P(defect) — model", fu.p_ml_defect.toFixed(3)) +
      kv("P(defect) — fuzzy", fu.p_fuzzy_defect.toFixed(3)) +
      kv("P(defect) — fused", fu.p_fused.toFixed(3)) +
      kv("Decision threshold", detectThreshold.toFixed(3));

    /* --- feature table --- */
    var frows = "<thead><tr><th>Measured feature</th><th style='text-align:right'>Value</th></tr></thead><tbody>";
    Object.keys(rep.features).forEach(function (k) {
      frows += "<tr><td>" + k.replace(/_/g, " ") + '</td><td class="num">' + rep.features[k] + "</td></tr>";
    });
    $("featTable").innerHTML = frows + "</tbody>";

    activate("overview");
    observeReveals(results);
    results.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
    if (stage) stage.resize();
  }

  function kv(k, v) { return '<div class="kv"><span>' + k + "</span><b>" + v + "</b></div>"; }

  /* first paint */
  observeReveals();
})();
