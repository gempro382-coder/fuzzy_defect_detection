/* ==========================================================================
   nf-stage.js — NeuraFuzz live 3D specimen stage
   --------------------------------------------------------------------------
   The inspection stage that replaces the intro. Displays the specimen plate
   on a workbench, with user-controlled orbit and raking lamp.
   ========================================================================== */
(function (global) {
  "use strict";

  var GL = global.NFGL;

  /* ======================================================================
     FALLBACK CONTROLLER (2D)
     ====================================================================== */
  function FallbackStage(cfg) {
    this.canvas = cfg.canvas;
    this.ctx = this.canvas.getContext('2d');
    this.img = new Image();
    this.overlayImg = new Image();
    this.progress = 0;
    this.verdict = null;
    this.mix = 0;
    
    var self = this;
    this.img.onload = function() { self.draw(); };
    this.overlayImg.onload = function() { self.draw(); };
  }
  
  FallbackStage.prototype.setImage = function (url) {
    this.img.src = url;
    this.mix = 0;
  };
  
  FallbackStage.prototype.setProgress = function (p) {
    this.progress = p;
    this.draw();
  };
  
  FallbackStage.prototype.setOverlay = function (url, verdict) {
    this.overlayImg.src = url;
    this.verdict = verdict;
    this.mix = 1;
    this.draw();
  };
  
  FallbackStage.prototype.setLampAngle = function (deg) { /* ignored in 2D */ };
  
  FallbackStage.prototype.resize = function () {
    var rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.height;
    this.draw();
  };
  
  FallbackStage.prototype.draw = function () {
    if (!this.ctx) return;
    var w = this.canvas.width, h = this.canvas.height;
    /* booth interior: a dark vignette so the specimen is the only lit thing */
    this.ctx.fillStyle = '#10151A';
    this.ctx.fillRect(0, 0, w, h);
    var vig = this.ctx.createRadialGradient(w / 2, h * 0.46, 0, w / 2, h * 0.46, Math.max(w, h) * 0.62);
    vig.addColorStop(0, 'rgba(58,70,82,0.55)');
    vig.addColorStop(1, 'rgba(0,0,0,0)');
    this.ctx.fillStyle = vig;
    this.ctx.fillRect(0, 0, w, h);

    if (!this.img.complete || !this.img.naturalWidth) return;
    
    var iw = this.img.naturalWidth, ih = this.img.naturalHeight;
    var scale = Math.min((w * 0.8) / iw, (h * 0.8) / ih);
    var dw = iw * scale, dh = ih * scale;
    var dx = (w - dw) / 2, dy = (h - dh) / 2;
    
    this.ctx.drawImage(this.img, dx, dy, dw, dh);
    
    if (this.mix > 0 && this.overlayImg.complete && this.overlayImg.naturalWidth) {
      this.ctx.globalAlpha = this.mix;
      this.ctx.drawImage(this.overlayImg, dx, dy, dw, dh);
      this.ctx.globalAlpha = 1.0;
    }
    
    if (this.progress > 0 && this.progress < 100) {
      var px = dx + (this.progress / 100) * dw;
      var beam = this.ctx.createLinearGradient(px - 26, 0, px + 4, 0);
      beam.addColorStop(0, 'rgba(255,176,58,0)');
      beam.addColorStop(1, 'rgba(255,176,58,0.34)');
      this.ctx.fillStyle = beam;
      this.ctx.fillRect(px - 26, dy, 26, dh);
      this.ctx.fillStyle = '#FFB03A';
      this.ctx.fillRect(px - 1, dy, 2, dh);
    }

    if (this.verdict && this.mix > 0) {
      this.ctx.strokeStyle = this.verdict === 'accept' ? '#3FD39E' :
                             (this.verdict === 'note' || this.verdict === 'warn') ? '#F2B341' : '#FF5A6E';
      this.ctx.lineWidth = 4;
      this.ctx.strokeRect(dx - 2, dy - 2, dw + 4, dh + 4);
    }
  };

  if (!GL) {
    global.NFStage = {
      supported: false,
      create: function(cfg) { return new FallbackStage(cfg); }
    };
    return;
  }

  /* ======================================================================
     WEBGL CONTROLLER
     ====================================================================== */
  var Vec3 = GL.Vec3, Mat4 = GL.Mat4, Geo = GL.Geo, Node = GL.Node;
  var lerp = GL.lerp, clamp = GL.clamp, clamp01 = GL.clamp01, hex = GL.hex, span = GL.span, Ease = GL.Ease;

  var C = {
    benchTop:  hex('#3D4750'),
    penetrant: hex('#C8F135'),
    lampBody:  hex('#2E353B'),
    /* the pipeline emits accept | warn | reject; 'note' kept as an alias */
    accept:    hex('#3FD39E'),
    note:      hex('#F2B341'),
    warn:      hex('#F2B341'),
    reject:    hex('#FF5A6E')
  };

  function buildSet(R) {
    var boxM = R.mesh('box', function () { return Geo.box(1, 1, 1); });
    var quadM = R.mesh('quad', function () { return Geo.quad(1, 1); });
    var cylM = R.mesh('cyl', function () { return Geo.cylinder(0.5, 0.5, 1, 14, true); });
    var coneM = R.mesh('rakeCone', function () { return Geo.cylinder(0.03, 0.55, 1.0, 20, false); });
    var sphM = R.mesh('sph', function () { return Geo.sphere(0.5, 14, 10); });

    var root = new Node('stageSet');

    /* bench top */
    var bench = root.add(new Node('bench')).set(0, -0.2, 0).sized(3.40, 0.05, 1.80);
    bench.mesh = boxM;
    bench.material = { color: C.benchTop, roughness: 0.34, specular: 0.72 };

    /* specimen plate tilted 35° on X axis */
    var plateTilt = 35 * Math.PI / 180;
    var plateBase = root.add(new Node('plateBase')).set(0, 0, 0).rotate(plateTilt, 0, 0);
    
    var plate = plateBase.add(new Node('plate')).set(0, 0.0, 0).sized(0.8, 0.02, 0.8);
    plate.mesh = boxM;
    plate.material = {
      mode: 2, 
      color: hex('#FFFFFF'), 
      roughness: 0.6, 
      specular: 0.3,
      mapA: null,
      mapB: null,
      mapMix: 0,
      scan: [0, 0.05, 0, 0], /* pos, halfWidth, intensity, enable */
      emissive: [0, 0, 0]
    };

    /* rake lamp */
    var lampRig = root.add(new Node('lampRig'));
    var lampHead = lampRig.add(new Node('lampHead')).set(0, 0, 0).sized(0.085, 0.055, 0.085);
    lampHead.mesh = sphM;
    lampHead.material = { color: C.penetrant, roughness: 1, specular: 0, emissive: [0, 0, 0] };
    
    var lampBezel = lampRig.add(new Node('lampBezel')).set(0, 0.04, 0).sized(0.10, 0.075, 0.10);
    lampBezel.mesh = cylM;
    lampBezel.material = { color: hex('#31383F'), roughness: 0.34, specular: 0.72 };

    return {
      root: root,
      plateBase: plateBase,
      plate: plate,
      lampRig: lampRig,
      lampHead: lampHead,
      meshes: { box: boxM, quad: quadM, cone: coneM }
    };
  }

  function Stage(cfg) {
    this.cfg = cfg;
    this.canvas = cfg.canvas;
    this.R = new GL.Renderer(this.canvas, { maxDpr: 1.6, post: true });
    
    if (!this.R.gl) {
      /* This shouldn't happen since we check GL.supported, but just in case */
      throw new Error("WebGL context lost or unsupported");
    }

    var R = this.R;
    R.fog.color = new Float32Array([0.021, 0.026, 0.031]);
    R.fog.density = 0.030;
    R.post.exposure = 1.10;
    R.post.bloom = 0.65;
    R.post.bloomThreshold = 0.55;
    R.post.vignette = 0.6;
    R.post.chroma = 0.1;
    R.post.grain = 0.04;

    this.set = buildSet(R);
    
    this.theta = 0;       /* orbit horizontal */
    this.phi = 0.4;       /* orbit vertical */
    this.target = [0, 0.0, 0];
    this.distance = 1.8;
    this.lampAngle = 6;   /* degrees */
    
    this.progress = 0;
    this.verdict = null;
    this.mix = 0;
    this.mixTarget = 0;
    
    this.texA = null;
    this.texB = null;

    this.dragging = false;
    this.lastMouse = { x: 0, y: 0 };

    this._bindEvents();
    
    this.destroyed = false;
    this._last = performance.now();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  Stage.prototype._bindEvents = function () {
    var self = this;
    
    this._onDown = function (e) {
      self.dragging = true;
      self.lastMouse.x = e.clientX || (e.touches && e.touches[0].clientX);
      self.lastMouse.y = e.clientY || (e.touches && e.touches[0].clientY);
    };
    
    this._onMove = function (e) {
      if (!self.dragging) return;
      var cx = e.clientX || (e.touches && e.touches[0].clientX);
      var cy = e.clientY || (e.touches && e.touches[0].clientY);
      var dx = cx - self.lastMouse.x;
      var dy = cy - self.lastMouse.y;
      self.lastMouse.x = cx;
      self.lastMouse.y = cy;
      
      self.theta -= dx * 0.01;
      self.phi = clamp(self.phi - dy * 0.01, 0.1, 1.2);
      
      if (self.cfg.onOrbit) self.cfg.onOrbit(self.theta, self.phi);
    };
    
    this._onUp = function (e) { self.dragging = false; };
    
    this.canvas.addEventListener('mousedown', this._onDown);
    this.canvas.addEventListener('mousemove', this._onMove);
    window.addEventListener('mouseup', this._onUp);
    
    this.canvas.addEventListener('touchstart', this._onDown, { passive: true });
    this.canvas.addEventListener('touchmove', this._onMove, { passive: true });
    window.addEventListener('touchend', this._onUp);
  };
  
  Stage.prototype.destroy = function () {
    this.destroyed = true;
    this.canvas.removeEventListener('mousedown', this._onDown);
    this.canvas.removeEventListener('mousemove', this._onMove);
    window.removeEventListener('mouseup', this._onUp);
    this.canvas.removeEventListener('touchstart', this._onDown);
    this.canvas.removeEventListener('touchmove', this._onMove);
    window.removeEventListener('touchend', this._onUp);
  };

  Stage.prototype.setImage = function (url) {
    var self = this;
    var img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function () {
      self.texA = self.R.createImageTexture(img, self.texA);
      self.set.plate.material.mapA = self.texA;
      self.set.plate.material.mapB = null;
      self.mix = 0;
      self.mixTarget = 0;
      self.progress = 0;
      self.verdict = null;
      self.set.plate.material.emissive = [0, 0, 0];
    };
    img.src = url;
  };

  Stage.prototype.setProgress = function (p) {
    this.progress = p;
  };

  Stage.prototype.setOverlay = function (url, verdict) {
    this.verdict = verdict;
    var self = this;
    var img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function () {
      self.texB = self.R.createImageTexture(img, self.texB);
      self.set.plate.material.mapB = self.texB;
      self.mixTarget = 1.0;
    };
    img.src = url;
  };

  Stage.prototype.setLampAngle = function (deg) {
    this.lampAngle = clamp(deg, 2, 20);
  };

  Stage.prototype.resize = function () {
    this.R.resize();
  };

  Stage.prototype.playSweep = function (report) {
    this.sweeping = true;
    this.sweepStart = performance.now();
    this.sweepReport = report;
    this._reticles = [];
    var layer = document.getElementById('stageReticles');
    if (layer) layer.innerHTML = '';
    
    if (layer && report.final.bboxes) {
      var bboxes = report.final.bboxes;
      for (var i = 0; i < bboxes.length; i++) {
        var box = bboxes[i];
        var lx = (box.x + box.w / 2 - 0.5) * 0.8;
        var lz = (box.y + box.h / 2 - 0.5) * 0.8;
        
        var tilt = 35 * Math.PI / 180;
        var wx = lx;
        var wy = lz * Math.sin(-tilt) + 0.02 * Math.cos(-tilt);
        var wz = lz * Math.cos(-tilt) - 0.02 * Math.sin(-tilt);
        
        var el = document.createElement('div');
        var cls = report.final.defect_type === 'crack' ? 'is-crack' : 
                  report.final.defect_type === 'corrosion' ? 'is-oxide' : 'is-score';
        var tag = report.final.defect_type.toUpperCase();
        var val = 'Sev ' + Math.round(report.final.severity);
        
        el.className = 'reticle ' + cls;
        el.innerHTML =
          '<svg class="reticle-ring" viewBox="0 0 64 64" aria-hidden="true">' +
          '<circle cx="32" cy="32" r="22"></circle>' +
          '<path d="M32 2v10M32 52v10M2 32h10M52 32h10"></path>' +
          '</svg>' +
          '<div class="reticle-tag"><b>' + tag + '</b><span>' + val + '</span></div>';
        el.style.opacity = '0';
        layer.appendChild(el);
        
        this._reticles.push({
          el: el,
          world: [wx, wy, wz],
          t0: 0.8 + i * 0.4
        });
      }
    }
  };

  Stage.prototype._loop = function (now) {
    if (this.destroyed) return;
    requestAnimationFrame(this._loop);
    
    var dt = Math.min(0.05, (now - this._last) / 1000);
    this._last = now;
    
    this.R.time = now / 1000;
    this.R.resize();
    
    if (this.mix < this.mixTarget) {
      this.mix = clamp01(this.mix + dt / 1.2); 
    }
    
    var set = this.set;
    var R = this.R;
    
    var mat = set.plate.material;
    mat.mapMix = this.mix;
    
    if (this.progress > 0 && this.progress < 100) {
      mat.scan[0] = this.progress / 100;
      mat.scan[1] = 0.05;
      mat.scan[2] = 1.0;
      mat.scan[3] = 1.0;
    } else {
      mat.scan[3] = 0.0;
    }
    
    if (this.verdict && this.mix > 0) {
      var col = C[this.verdict] || C.accept;
      mat.emissive = [col[0] * this.mix * 0.1, col[1] * this.mix * 0.1, col[2] * this.mix * 0.1];
    } else {
      mat.emissive = [0, 0, 0];
    }
    
    var cx = this.target[0] + this.distance * Math.cos(this.phi) * Math.sin(this.theta);
    var cy = this.target[1] + this.distance * Math.sin(this.phi);
    var cz = this.target[2] + this.distance * Math.cos(this.phi) * Math.cos(this.theta);
    
    R.setCamera(
      Vec3.create(cx, cy, cz),
      Vec3.create(this.target[0], this.target[1], this.target[2]),
      Vec3.create(0, 1, 0),
      35, 0.1, 100
    );
    
    var tilt = 35 * Math.PI / 180;
    var lx, ly, lz, lpos, ldir;
    var lampIntensity = 0.0;
    
    if (this.sweeping) {
      lampIntensity = 1.0;
      var sweepT = (now - this.sweepStart) / 1000;
      var sweepProgress = clamp01(sweepT / 3.0);
      var sweepAng = lerp(-1.8, 0.5, Ease.inOutSine(sweepProgress));
      var sweepRad = 1.2;
      
      var lampY = Math.tan(6.0 * Math.PI / 180) * sweepRad;
      var lampX = Math.sin(sweepAng) * sweepRad;
      var lampZ = Math.cos(sweepAng) * sweepRad;
      
      lx = lampX;
      ly = lampY * Math.cos(-tilt) - lampZ * Math.sin(-tilt);
      lz = lampY * Math.sin(-tilt) + lampZ * Math.cos(-tilt);
      lpos = Vec3.create(lx, ly, lz);
      ldir = Vec3.normalize(Vec3.create(), Vec3.sub(Vec3.create(), this.target, lpos));
      
      for (var k = 0; k < (this._reticles || []).length; k++) {
        var r = this._reticles[k];
        var on = span(sweepT, r.t0, r.t0 + 0.4, Ease.outCubic);
        var p = R.project(r.world);
        if (!p || on <= 0) { r.el.style.opacity = '0'; continue; }
        r.el.style.opacity = String(on);
        r.el.style.transform = 'translate(' + p.x.toFixed(1) + 'px,' + p.y.toFixed(1) + 'px) translate(-50%,-50%) scale(' + (0.7 + on * 0.3).toFixed(2) + ')';
      }
      
      if (sweepT > 5.0) {
        this.sweeping = false;
      }
    } else {
      if (this.sweepStart) {
        var timeSinceSweep = (now - this.sweepStart) / 1000 - 5.0;
        lampIntensity = Math.max(0.0, 1.0 - timeSinceSweep * 1.5);
      }
      var rad = 1.2;
      var elev = tilt + (this.lampAngle * Math.PI / 180);
      lx = 0;
      ly = Math.sin(elev) * rad;
      lz = Math.cos(elev) * rad;
      lpos = Vec3.create(lx, ly, lz);
      ldir = Vec3.normalize(Vec3.create(), Vec3.sub(Vec3.create(), this.target, lpos));
    }
    
    R.lights.l1.pos = lpos;
    R.lights.l1.dir = ldir;
    R.lights.l1.color = new Float32Array([1.55 * lampIntensity, 1.72 * lampIntensity, 1.15 * lampIntensity]);
    R.lights.l1.cone = new Float32Array([0.985, 0.845, 5.0, 0.22]);
    R.lights.l0.cone[2] = 0; 
    
    var h = Math.sqrt(ldir[0] * ldir[0] + ldir[2] * ldir[2]);
    var rx = Math.atan2(-h, -ldir[1]);
    var ry = Math.atan2(ldir[0], ldir[2]);
    set.lampRig.set(lpos[0], lpos[1], lpos[2]).rotate(rx, ry, 0);
    
    R.beginScene();
    set.root.updateWorld(null);
    set.root.render(R);
    
    if (lampIntensity > 0.01) {
      var tmp = Mat4.create();
      var len = 1.2;
      var mid = [lpos[0] + ldir[0] * len * 0.5, lpos[1] + ldir[1] * len * 0.5, lpos[2] + ldir[2] * len * 0.5];
      var yaw = Math.atan2(ldir[0], ldir[2]);
      var pitch = Math.asin(clamp(-ldir[1], -1, 1));
      Mat4.compose(tmp, mid, [-(Math.PI / 2 - pitch), yaw, 0], [0.30, len, 0.30]);
      R.drawVolume(set.meshes.cone, tmp, C.penetrant, 0.15 * lampIntensity, 0);
    }
    
    R.endScene();
  };

  global.NFStage = {
    supported: true,
    create: function (cfg) {
      return new Stage(cfg);
    }
  };

})(window);
