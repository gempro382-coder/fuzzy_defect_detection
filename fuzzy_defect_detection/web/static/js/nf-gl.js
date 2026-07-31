/* ==========================================================================
   nf-gl.js — NeuraFuzz micro 3D engine
   --------------------------------------------------------------------------
   A small, dependency-free WebGL1 renderer written for this project. No
   three.js, no CDN: the inspection booth has to render on an offline lab
   machine, so everything here is self-contained.

   What it provides
     · Mat4 / Vec3 maths (column-major, GLSL-compatible)
     · Program compilation with cached uniform locations
     · Geometry builders: plane, box, cylinder/cone, sphere, torus
     · One "surface" shader with four modes:
         0  plain      — walls, bench, figures (hemi ambient + 2 spotlights)
         1  specimen   — procedural ceramic coupon whose CRACKS, CORROSION and
                         SCRATCHES are revealed by grazing light
         2  mapped     — a real photograph on a plate, with an annotated
                         overlay cross-fade and a travelling scan bar
         3  floor      — cast concrete with control joints and hazard paint
     · Additive volumetrics (light cones, floor light-pools, glow sprites)
     · Point-sprite dust motes that brighten inside a beam
     · Post chain: bright-pass → separable blur bloom → filmic composite with
       vignette, chromatic aberration and film grain

   Everything degrades: if WebGL is missing the caller gets null and shows a
   flat fallback; if a framebuffer is incomplete the post chain switches off
   and the scene draws straight to the canvas.
   ========================================================================== */
(function (global) {
  "use strict";

  /* ======================================================================
     1. MATHS
     ====================================================================== */

  var Vec3 = {
    create: function (x, y, z) { return new Float32Array([x || 0, y || 0, z || 0]); },
    set: function (o, x, y, z) { o[0] = x; o[1] = y; o[2] = z; return o; },
    copy: function (o, a) { o[0] = a[0]; o[1] = a[1]; o[2] = a[2]; return o; },
    add: function (o, a, b) { o[0] = a[0] + b[0]; o[1] = a[1] + b[1]; o[2] = a[2] + b[2]; return o; },
    sub: function (o, a, b) { o[0] = a[0] - b[0]; o[1] = a[1] - b[1]; o[2] = a[2] - b[2]; return o; },
    scale: function (o, a, s) { o[0] = a[0] * s; o[1] = a[1] * s; o[2] = a[2] * s; return o; },
    len: function (a) { return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); },
    dot: function (a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; },
    cross: function (o, a, b) {
      var ax = a[0], ay = a[1], az = a[2], bx = b[0], by = b[1], bz = b[2];
      o[0] = ay * bz - az * by; o[1] = az * bx - ax * bz; o[2] = ax * by - ay * bx;
      return o;
    },
    normalize: function (o, a) {
      var l = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
      if (l < 1e-8) { o[0] = 0; o[1] = 0; o[2] = 0; return o; }
      o[0] = a[0] / l; o[1] = a[1] / l; o[2] = a[2] / l; return o;
    },
    lerp: function (o, a, b, t) {
      o[0] = a[0] + (b[0] - a[0]) * t;
      o[1] = a[1] + (b[1] - a[1]) * t;
      o[2] = a[2] + (b[2] - a[2]) * t;
      return o;
    }
  };

  /* Column-major 4x4, laid out exactly as GLSL expects it. */
  var Mat4 = {
    create: function () {
      return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    },

    identity: function (o) {
      o[0] = 1; o[1] = 0; o[2] = 0; o[3] = 0;
      o[4] = 0; o[5] = 1; o[6] = 0; o[7] = 0;
      o[8] = 0; o[9] = 0; o[10] = 1; o[11] = 0;
      o[12] = 0; o[13] = 0; o[14] = 0; o[15] = 1;
      return o;
    },

    copy: function (o, a) { for (var i = 0; i < 16; i++) o[i] = a[i]; return o; },

    multiply: function (o, a, b) {
      var a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3],
          a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7],
          a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11],
          a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
      for (var i = 0; i < 4; i++) {
        var b0 = b[i * 4], b1 = b[i * 4 + 1], b2 = b[i * 4 + 2], b3 = b[i * 4 + 3];
        o[i * 4]     = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
        o[i * 4 + 1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
        o[i * 4 + 2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
        o[i * 4 + 3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
      }
      return o;
    },

    fromTranslation: function (o, x, y, z) {
      Mat4.identity(o); o[12] = x; o[13] = y; o[14] = z; return o;
    },

    fromScaling: function (o, x, y, z) {
      Mat4.identity(o); o[0] = x; o[5] = y; o[10] = z; return o;
    },

    fromRotationX: function (o, r) {
      var c = Math.cos(r), s = Math.sin(r);
      Mat4.identity(o); o[5] = c; o[6] = s; o[9] = -s; o[10] = c; return o;
    },
    fromRotationY: function (o, r) {
      var c = Math.cos(r), s = Math.sin(r);
      Mat4.identity(o); o[0] = c; o[2] = -s; o[8] = s; o[10] = c; return o;
    },
    fromRotationZ: function (o, r) {
      var c = Math.cos(r), s = Math.sin(r);
      Mat4.identity(o); o[0] = c; o[1] = s; o[4] = -s; o[5] = c; return o;
    },

    /* Compose translate → rotateY → rotateX → rotateZ → scale (TRS). */
    compose: function (o, pos, rot, scl) {
      var cx = Math.cos(rot[0]), sx = Math.sin(rot[0]);
      var cy = Math.cos(rot[1]), sy = Math.sin(rot[1]);
      var cz = Math.cos(rot[2]), sz = Math.sin(rot[2]);
      /* R = Ry * Rx * Rz  (yaw, then pitch, then roll) */
      var m00 = cy * cz + sy * sx * sz;
      var m01 = cx * sz;
      var m02 = -sy * cz + cy * sx * sz;
      var m10 = -cy * sz + sy * sx * cz;
      var m11 = cx * cz;
      var m12 = sy * sz + cy * sx * cz;
      var m20 = sy * cx;
      var m21 = -sx;
      var m22 = cy * cx;
      var sX = scl[0], sY = scl[1], sZ = scl[2];
      o[0] = m00 * sX; o[1] = m01 * sX; o[2] = m02 * sX; o[3] = 0;
      o[4] = m10 * sY; o[5] = m11 * sY; o[6] = m12 * sY; o[7] = 0;
      o[8] = m20 * sZ; o[9] = m21 * sZ; o[10] = m22 * sZ; o[11] = 0;
      o[12] = pos[0]; o[13] = pos[1]; o[14] = pos[2]; o[15] = 1;
      return o;
    },

    perspective: function (o, fovy, aspect, near, far) {
      var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      o[0] = f / aspect; o[1] = 0; o[2] = 0; o[3] = 0;
      o[4] = 0; o[5] = f; o[6] = 0; o[7] = 0;
      o[8] = 0; o[9] = 0; o[10] = (far + near) * nf; o[11] = -1;
      o[12] = 0; o[13] = 0; o[14] = 2 * far * near * nf; o[15] = 0;
      return o;
    },

    lookAt: function (o, eye, center, up) {
      var z = Vec3.normalize(Vec3.create(), Vec3.sub(Vec3.create(), eye, center));
      var x = Vec3.normalize(Vec3.create(), Vec3.cross(Vec3.create(), up, z));
      if (Vec3.len(x) < 1e-6) {
        /* eye is directly above/below the target — nudge the up vector */
        x = Vec3.normalize(Vec3.create(),
          Vec3.cross(Vec3.create(), Vec3.create(0.0001, 1, 0.0001), z));
      }
      var y = Vec3.cross(Vec3.create(), z, x);
      o[0] = x[0]; o[1] = y[0]; o[2] = z[0]; o[3] = 0;
      o[4] = x[1]; o[5] = y[1]; o[6] = z[1]; o[7] = 0;
      o[8] = x[2]; o[9] = y[2]; o[10] = z[2]; o[11] = 0;
      o[12] = -Vec3.dot(x, eye); o[13] = -Vec3.dot(y, eye); o[14] = -Vec3.dot(z, eye);
      o[15] = 1;
      return o;
    },

    invert: function (o, a) {
      var a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3],
          a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7],
          a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11],
          a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
      var b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10,
          b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11,
          b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12,
          b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30,
          b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31,
          b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
      var det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
      if (!det) return null;
      det = 1.0 / det;
      o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
      o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
      o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
      o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
      o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
      o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
      o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
      o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
      o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
      o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
      o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
      o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
      o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
      o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
      o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
      o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
      return o;
    },

    /* mat3 normal matrix = transpose(inverse(upper-left 3x3)) */
    normalMatrix: function (o3, m4) {
      var inv = Mat4.invert(Mat4.create(), m4);
      if (!inv) {
        o3[0] = 1; o3[1] = 0; o3[2] = 0;
        o3[3] = 0; o3[4] = 1; o3[5] = 0;
        o3[6] = 0; o3[7] = 0; o3[8] = 1;
        return o3;
      }
      /* transpose while extracting */
      o3[0] = inv[0]; o3[1] = inv[4]; o3[2] = inv[8];
      o3[3] = inv[1]; o3[4] = inv[5]; o3[5] = inv[9];
      o3[6] = inv[2]; o3[7] = inv[6]; o3[8] = inv[10];
      return o3;
    },

    transformPoint: function (out, m, p) {
      var x = p[0], y = p[1], z = p[2];
      var w = m[3] * x + m[7] * y + m[11] * z + m[15];
      w = w || 1.0;
      out[0] = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
      out[1] = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
      out[2] = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;
      return out;
    },

    transformDirection: function (out, m, p) {
      var x = p[0], y = p[1], z = p[2];
      out[0] = m[0] * x + m[4] * y + m[8] * z;
      out[1] = m[1] * x + m[5] * y + m[9] * z;
      out[2] = m[2] * x + m[6] * y + m[10] * z;
      return out;
    }
  };

  /* ======================================================================
     2. EASING + SMALL UTILITIES
     ====================================================================== */

  var Ease = {
    linear: function (t) { return t; },
    inQuad: function (t) { return t * t; },
    outQuad: function (t) { return t * (2 - t); },
    inOutQuad: function (t) { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; },
    inCubic: function (t) { return t * t * t; },
    outCubic: function (t) { var u = t - 1; return u * u * u + 1; },
    inOutCubic: function (t) {
      return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    },
    outQuart: function (t) { return 1 - Math.pow(1 - t, 4); },
    inOutQuart: function (t) {
      return t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2;
    },
    outExpo: function (t) { return t >= 1 ? 1 : 1 - Math.pow(2, -10 * t); },
    inOutSine: function (t) { return -(Math.cos(Math.PI * t) - 1) / 2; },
    outBack: function (t) {
      var c1 = 1.70158, c3 = c1 + 1, u = t - 1;
      return 1 + c3 * u * u * u + c1 * u * u;
    },
    outElastic: function (t) {
      var c4 = (2 * Math.PI) / 3;
      return t === 0 ? 0 : t === 1 ? 1
        : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
    }
  };

  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function lerp(a, b, t) { return a + (b - a) * t; }

  /* Normalised, clamped progress of `t` across [a,b] with optional easing. */
  function span(t, a, b, easing) {
    if (b <= a) return t >= b ? 1 : 0;
    var u = clamp01((t - a) / (b - a));
    return easing ? easing(u) : u;
  }

  /* "#C8F135" or "#c8f" → [r,g,b] in 0..1 linear-ish (gamma left in sRGB) */
  function hex(h) {
    h = String(h).replace('#', '');
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    return new Float32Array([
      ((n >> 16) & 255) / 255,
      ((n >> 8) & 255) / 255,
      (n & 255) / 255
    ]);
  }

  /* Multiply a colour by a scalar into a fresh vec3 (light intensity). */
  function tint(rgb, k) {
    return new Float32Array([rgb[0] * k, rgb[1] * k, rgb[2] * k]);
  }

  /* ======================================================================
     3. GEOMETRY BUILDERS
     Each returns { position, normal, uv, index } typed arrays.
     ====================================================================== */

  var Geo = {};

  /* Horizontal grid plane in XZ, normal +Y, uv 0..1 (tiled by `rep`). */
  Geo.plane = function (w, d, sx, sy, rep) {
    sx = Math.max(1, sx | 0); sy = Math.max(1, sy | 0); rep = rep || 1;
    var pos = [], nrm = [], uv = [], idx = [];
    for (var j = 0; j <= sy; j++) {
      for (var i = 0; i <= sx; i++) {
        var u = i / sx, v = j / sy;
        pos.push((u - 0.5) * w, 0, (v - 0.5) * d);
        nrm.push(0, 1, 0);
        uv.push(u * rep, v * rep);
      }
    }
    for (var jj = 0; jj < sy; jj++) {
      for (var ii = 0; ii < sx; ii++) {
        var a = jj * (sx + 1) + ii, b = a + 1, c = a + sx + 1, dd = c + 1;
        idx.push(a, c, b, b, c, dd);
      }
    }
    return Geo._pack(pos, nrm, uv, idx);
  };

  /* Vertical quad in XY, normal +Z. Used for cards, screens, decals. */
  Geo.quad = function (w, h) {
    var hw = w / 2, hh = h / 2;
    return Geo._pack(
      [-hw, -hh, 0, hw, -hh, 0, hw, hh, 0, -hw, hh, 0],
      [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
      [0, 0, 1, 0, 1, 1, 0, 1],
      [0, 1, 2, 0, 2, 3]
    );
  };

  Geo.box = function (w, h, d) {
    var x = w / 2, y = h / 2, z = d / 2;
    var pos = [], nrm = [], uv = [], idx = [];
    /* face: origin, u-axis, v-axis, normal */
    var faces = [
      [[-x, -y, z], [1, 0, 0], [0, 1, 0], [0, 0, 1]],   /* +Z front  */
      [[x, -y, -z], [-1, 0, 0], [0, 1, 0], [0, 0, -1]], /* -Z back   */
      [[x, -y, z], [0, 0, -1], [0, 1, 0], [1, 0, 0]],   /* +X right  */
      [[-x, -y, -z], [0, 0, 1], [0, 1, 0], [-1, 0, 0]], /* -X left   */
      [[-x, y, z], [1, 0, 0], [0, 0, -1], [0, 1, 0]],   /* +Y top    */
      [[-x, -y, -z], [1, 0, 0], [0, 0, 1], [0, -1, 0]]  /* -Y bottom */
    ];
    var scale = [[w, h], [w, h], [d, h], [d, h], [w, d], [w, d]];
    for (var f = 0; f < faces.length; f++) {
      var o = faces[f][0], ua = faces[f][1], va = faces[f][2], n = faces[f][3];
      var su = scale[f][0], sv = scale[f][1];
      var base = pos.length / 3;
      for (var k = 0; k < 4; k++) {
        var cu = (k === 1 || k === 2) ? 1 : 0;
        var cv = (k === 2 || k === 3) ? 1 : 0;
        pos.push(
          o[0] + ua[0] * cu * su + va[0] * cv * sv,
          o[1] + ua[1] * cu * su + va[1] * cv * sv,
          o[2] + ua[2] * cu * su + va[2] * cv * sv
        );
        nrm.push(n[0], n[1], n[2]);
        uv.push(cu, cv);
      }
      idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    return Geo._pack(pos, nrm, uv, idx);
  };

  /* Cylinder / cone / truncated cone along +Y, centred at the origin. */
  Geo.cylinder = function (rTop, rBottom, h, seg, capped) {
    seg = Math.max(3, seg | 0);
    var pos = [], nrm = [], uv = [], idx = [];
    var hy = h / 2;
    var slope = Math.atan2(rBottom - rTop, h);
    var cs = Math.cos(slope), sn = Math.sin(slope);
    for (var i = 0; i <= seg; i++) {
      var a = (i / seg) * Math.PI * 2;
      var ca = Math.cos(a), sa = Math.sin(a);
      /* top ring then bottom ring */
      pos.push(ca * rTop, hy, sa * rTop);
      nrm.push(ca * cs, sn, sa * cs);
      uv.push(i / seg, 1);
      pos.push(ca * rBottom, -hy, sa * rBottom);
      nrm.push(ca * cs, sn, sa * cs);
      uv.push(i / seg, 0);
    }
    for (var s = 0; s < seg; s++) {
      var t0 = s * 2, b0 = t0 + 1, t1 = t0 + 2, b1 = t0 + 3;
      idx.push(t0, b0, t1, t1, b0, b1);
    }
    if (capped) {
      /* top cap */
      if (rTop > 1e-6) {
        var ct = pos.length / 3;
        pos.push(0, hy, 0); nrm.push(0, 1, 0); uv.push(0.5, 0.5);
        for (var ti = 0; ti <= seg; ti++) {
          var ta = (ti / seg) * Math.PI * 2;
          pos.push(Math.cos(ta) * rTop, hy, Math.sin(ta) * rTop);
          nrm.push(0, 1, 0);
          uv.push(0.5 + Math.cos(ta) * 0.5, 0.5 + Math.sin(ta) * 0.5);
        }
        for (var tk = 0; tk < seg; tk++) idx.push(ct, ct + 1 + tk, ct + 2 + tk);
      }
      /* bottom cap */
      if (rBottom > 1e-6) {
        var cb = pos.length / 3;
        pos.push(0, -hy, 0); nrm.push(0, -1, 0); uv.push(0.5, 0.5);
        for (var bi = 0; bi <= seg; bi++) {
          var ba = (bi / seg) * Math.PI * 2;
          pos.push(Math.cos(ba) * rBottom, -hy, Math.sin(ba) * rBottom);
          nrm.push(0, -1, 0);
          uv.push(0.5 + Math.cos(ba) * 0.5, 0.5 + Math.sin(ba) * 0.5);
        }
        for (var bk = 0; bk < seg; bk++) idx.push(cb, cb + 2 + bk, cb + 1 + bk);
      }
    }
    return Geo._pack(pos, nrm, uv, idx);
  };

  Geo.sphere = function (r, segW, segH) {
    segW = Math.max(3, segW | 0); segH = Math.max(2, segH | 0);
    var pos = [], nrm = [], uv = [], idx = [];
    for (var j = 0; j <= segH; j++) {
      var phi = (j / segH) * Math.PI;
      var sp = Math.sin(phi), cp = Math.cos(phi);
      for (var i = 0; i <= segW; i++) {
        var th = (i / segW) * Math.PI * 2;
        var st = Math.sin(th), ct = Math.cos(th);
        var nx = sp * ct, ny = cp, nz = sp * st;
        pos.push(nx * r, ny * r, nz * r);
        nrm.push(nx, ny, nz);
        uv.push(i / segW, 1 - j / segH);
      }
    }
    for (var jj = 0; jj < segH; jj++) {
      for (var ii = 0; ii < segW; ii++) {
        var a = jj * (segW + 1) + ii, b = a + segW + 1;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
    return Geo._pack(pos, nrm, uv, idx);
  };

  Geo.torus = function (R, r, segT, segP) {
    segT = Math.max(3, segT | 0); segP = Math.max(3, segP | 0);
    var pos = [], nrm = [], uv = [], idx = [];
    for (var j = 0; j <= segT; j++) {
      var u = (j / segT) * Math.PI * 2;
      var cu = Math.cos(u), su = Math.sin(u);
      for (var i = 0; i <= segP; i++) {
        var v = (i / segP) * Math.PI * 2;
        var cv = Math.cos(v), sv = Math.sin(v);
        pos.push((R + r * cv) * cu, r * sv, (R + r * cv) * su);
        nrm.push(cv * cu, sv, cv * su);
        uv.push(j / segT, i / segP);
      }
    }
    for (var jj = 0; jj < segT; jj++) {
      for (var ii = 0; ii < segP; ii++) {
        var a = jj * (segP + 1) + ii, b = a + segP + 1;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
    return Geo._pack(pos, nrm, uv, idx);
  };

  Geo._pack = function (pos, nrm, uv, idx) {
    var Index = (pos.length / 3) > 65535 ? Uint32Array : Uint16Array;
    return {
      position: new Float32Array(pos),
      normal: new Float32Array(nrm),
      uv: new Float32Array(uv),
      index: new Index(idx),
      count: idx.length
    };
  };

  /* ======================================================================
     4. SHADER SOURCE
     ====================================================================== */

  var GLSL_COMMON = [
    '#ifdef GL_FRAGMENT_PRECISION_HIGH',
    'precision highp float;',
    '#else',
    'precision mediump float;',
    '#endif'
  ].join('\n');

  /* --- noise toolkit shared by the procedural surfaces ------------------ */
  var GLSL_NOISE = [
    'float hash21(vec2 p){',
    '  p = fract(p * vec2(123.34, 456.21));',
    '  p += dot(p, p + 45.32);',
    '  return fract(p.x * p.y);',
    '}',
    'vec2 hash22(vec2 p){',
    '  return vec2(hash21(p), hash21(p + vec2(19.73, 7.31)));',
    '}',
    'float vnoise(vec2 p){',
    '  vec2 i = floor(p), f = fract(p);',
    '  vec2 u = f * f * (3.0 - 2.0 * f);',
    '  float a = hash21(i);',
    '  float b = hash21(i + vec2(1.0, 0.0));',
    '  float c = hash21(i + vec2(0.0, 1.0));',
    '  float d = hash21(i + vec2(1.0, 1.0));',
    '  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);',
    '}',
    'float fbm(vec2 p){',
    '  float s = 0.0, a = 0.5;',
    '  for (int i = 0; i < 5; i++){ s += a * vnoise(p); p *= 2.03; a *= 0.5; }',
    '  return s;',
    '}',
    'float fbm3(vec2 p){',
    '  float s = 0.0, a = 0.5;',
    '  for (int i = 0; i < 3; i++){ s += a * vnoise(p); p *= 2.11; a *= 0.5; }',
    '  return s;',
    '}',
    /* Worley F2-F1: near zero on cell borders, which is exactly where a
       brittle fracture network wants to run. */
    'float worleyEdge(vec2 p){',
    '  vec2 ip = floor(p), fp = fract(p);',
    '  float d1 = 8.0, d2 = 8.0;',
    '  for (int y = -1; y <= 1; y++){',
    '    for (int x = -1; x <= 1; x++){',
    '      vec2 g = vec2(float(x), float(y));',
    '      vec2 o = hash22(ip + g);',
    '      float d = length(g + o - fp);',
    '      if (d < d1){ d2 = d1; d1 = d; } else if (d < d2){ d2 = d; }',
    '    }',
    '  }',
    '  return d2 - d1;',
    '}',
    'float worleyDist(vec2 p){',
    '  vec2 ip = floor(p), fp = fract(p);',
    '  float d1 = 8.0;',
    '  for (int y = -1; y <= 1; y++){',
    '    for (int x = -1; x <= 1; x++){',
    '      vec2 g = vec2(float(x), float(y));',
    '      vec2 o = hash22(ip + g + vec2(3.7, 11.3));',
    '      d1 = min(d1, length(g + o - fp));',
    '    }',
    '  }',
    '  return d1;',
    '}'
  ].join('\n');

  /* --- the defect field: cracks, corrosion, scratches, pitting --------- */
  var GLSL_DEFECTS = [
    /* Crack network. Domain-warped so the fracture branches instead of
       looking like a Voronoi diagram. Returns 1 on a crack line. */
    'float crackField(vec2 uv, float scale){',
    '  vec2 w = uv * scale + vec2(fbm3(uv * 3.1), fbm3(uv * 3.1 + 17.0)) * 0.85;',
    '  float e = worleyEdge(w);',
    '  float line = 1.0 - smoothstep(0.0, 0.075, e);',
    /*   thin the network so only part of it is an actual propagating crack */
    '  float where = smoothstep(0.42, 0.68, fbm(uv * 2.2 + 4.0));',
    '  return line * where;',
    '}',
    /* Oxide bloom: low-frequency blotches with high-frequency pitting. */
    'float corrosionField(vec2 uv, out float pit){',
    '  float m = fbm(uv * 4.4 + 11.0);',
    '  float blotch = smoothstep(0.50, 0.80, m);',
    '  pit = smoothstep(0.55, 0.15, worleyDist(uv * 26.0));',
    '  return clamp(blotch * (0.55 + 0.75 * pit), 0.0, 1.0);',
    '}',
    /* Machining scratches: strongly anisotropic streaks. */
    'float scratchField(vec2 uv, float ang){',
    '  float c = cos(ang), s = sin(ang);',
    '  vec2 r = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);',
    '  vec2 q = vec2(r.x * 0.55, r.y * 16.0);',
    '  float n = fbm3(q);',
    '  float line = smoothstep(0.60, 0.78, n);',
    '  float where = smoothstep(0.48, 0.72, fbm(uv * 1.7 + 31.0));',
    '  return line * where;',
    '}',
    /* Combined micro-relief height, used for the bump normal. */
    'float reliefHeight(vec2 uv, vec4 amt, float scratchAng){',
    '  float pit = 0.0;',
    '  float cr = crackField(uv, 7.0) * amt.x;',
    '  float co = corrosionField(uv, pit) * amt.y;',
    '  float sc = scratchField(uv, scratchAng) * amt.z;',
    '  float grain = (fbm(uv * 42.0) - 0.5) * 0.10;',
    '  return -cr * 1.0 - co * 0.34 - pit * co * 0.30 - sc * 0.46 + grain;',
    '}'
  ].join('\n');

  /* --- surface vertex shader ------------------------------------------- */
  var VS_SURFACE = [
    'attribute vec3 aPosition;',
    'attribute vec3 aNormal;',
    'attribute vec2 aUv;',
    'uniform mat4 uProjection;',
    'uniform mat4 uView;',
    'uniform mat4 uModel;',
    'uniform mat3 uNormalMat;',
    'varying vec3 vWorld;',
    'varying vec3 vNormal;',
    'varying vec2 vUv;',
    'void main(){',
    '  vec4 wp = uModel * vec4(aPosition, 1.0);',
    '  vWorld = wp.xyz;',
    '  vNormal = normalize(uNormalMat * aNormal);',
    '  vUv = aUv;',
    '  gl_Position = uProjection * uView * wp;',
    '}'
  ].join('\n');

  /* --- surface fragment shader ----------------------------------------- */
  var FS_SURFACE = [
    GLSL_COMMON,
    GLSL_NOISE,
    GLSL_DEFECTS,
    'varying vec3 vWorld;',
    'varying vec3 vNormal;',
    'varying vec2 vUv;',

    'uniform vec3 uCamera;',
    'uniform int  uMode;',           /* 0 plain · 1 specimen · 2 mapped · 3 floor */
    'uniform vec3 uBaseColor;',
    'uniform vec2 uSurface;',        /* x = roughness, y = specular strength */
    'uniform vec3 uEmissive;',
    'uniform float uAlpha;',

    /* two spotlights: ceiling work-lamp + handheld raking lamp */
    'uniform vec3 uL0Pos; uniform vec3 uL0Dir; uniform vec3 uL0Color; uniform vec4 uL0Cone;',
    'uniform vec3 uL1Pos; uniform vec3 uL1Dir; uniform vec3 uL1Color; uniform vec4 uL1Cone;',
    /* hemispherical ambience + a cool directional fill */
    'uniform vec3 uAmbSky; uniform vec3 uAmbGround;',
    'uniform vec3 uFillDir; uniform vec3 uFillColor;',
    /* atmosphere */
    'uniform vec3 uFogColor; uniform float uFogDensity;',

    /* procedural surface controls */
    'uniform vec4 uDefect;',        /* crack, corrosion, scratch, reveal      */
    'uniform vec2 uUvScale;',
    'uniform float uScratchAngle;',
    'uniform vec3 uTangent;',       /* world-space tangent  (for bump frame)  */
    'uniform vec3 uBitangent;',     /* world-space bitangent                  */
    'uniform float uBumpScale;',
    'uniform float uTime;',

    /* mapped mode */
    'uniform sampler2D uMapA;',
    'uniform sampler2D uMapB;',
    'uniform float uMapMix;',
    'uniform vec3 uTintColor;',
    'uniform float uTintAmount;',
    'uniform vec4 uScan;',          /* pos 0..1, halfWidth, intensity, enable */
    'uniform float uGridFade;',

    /* ---------- spotlight contribution ---------- */
    'vec3 spot(vec3 P, vec3 N, vec3 V, vec3 lp, vec3 ld, vec3 lc, vec4 cone,',
    '          vec3 albedo, float rough, float specK, out float grazing){',
    '  vec3 L = lp - P;',
    '  float dist = length(L);',
    '  grazing = 0.0;',
    '  if (dist < 0.0001 || cone.z <= 0.0) return vec3(0.0);',
    '  L /= dist;',
    '  float cd = dot(-L, normalize(ld));',
    '  float shape = smoothstep(cone.y, cone.x, cd);',
    '  if (shape <= 0.0) return vec3(0.0);',
    '  float att = cone.z / (1.0 + cone.w * dist * dist);',
    '  float ndl = dot(N, L);',
    /*  a crack is only visible when the light skims the surface, so report
        how grazing this light is for the caller to use as a reveal weight */
    '  grazing = shape * att * pow(1.0 - abs(ndl), 3.0);',
    '  float diff = max(ndl, 0.0);',
    '  vec3 H = normalize(L + V);',
    '  float shin = mix(90.0, 6.0, clamp(rough, 0.0, 1.0));',
    '  float spec = pow(max(dot(N, H), 0.0), shin) * specK;',
    '  return (albedo * diff + vec3(spec)) * lc * shape * att;',
    '}',

    'void main(){',
    '  vec2 uv = vUv * uUvScale;',
    '  vec3 N = normalize(vNormal);',
    '  vec3 V = normalize(uCamera - vWorld);',
    '  vec3 albedo = uBaseColor;',
    '  float rough = uSurface.x;',
    '  float specK = uSurface.y;',
    '  vec3 emissive = uEmissive;',
    '  float relief = 0.0;',
    '  float crackMask = 0.0;',
    '  float corrMask = 0.0;',
    '  float scratchMask = 0.0;',

    /* ---------------- mode 1 : procedural ceramic specimen ------------- */
    '  if (uMode == 1) {',
    '    float pit = 0.0;',
    '    crackMask   = crackField(uv, 7.0) * uDefect.x;',
    '    corrMask    = corrosionField(uv, pit) * uDefect.y;',
    '    scratchMask = scratchField(uv, uScratchAngle) * uDefect.z;',
    /*   fired-ceramic body with kiln mottling and a faint glaze grid */
    '    float mottle = fbm(uv * 5.5) * 0.16 + fbm(uv * 19.0) * 0.07;',
    '    albedo = uBaseColor * (0.90 + mottle);',
    '    vec2 g = abs(fract(uv * 2.0) - 0.5);',
    '    float joint = 1.0 - smoothstep(0.44, 0.495, max(g.x, g.y));',
    '    albedo = mix(albedo * 0.55, albedo, joint);',
    /*   corrosion recolours the body toward iron oxide */
    '    vec3 oxide = mix(vec3(0.42, 0.17, 0.07), vec3(0.75, 0.36, 0.13),',
    '                     fbm(uv * 9.0));',
    '    albedo = mix(albedo, oxide, corrMask * 0.92);',
    '    albedo = mix(albedo, albedo * 0.35, pit * corrMask);',
    /*   a crack is a void: almost no light comes back out of it */
    '    albedo = mix(albedo, albedo * 0.10, crackMask);',
    '    albedo = mix(albedo, albedo * 1.30, scratchMask * 0.7);',
    '    rough = clamp(rough + corrMask * 0.45 - scratchMask * 0.25, 0.04, 1.0);',
    '    relief = 1.0;',
    '  }',

    /* ---------------- mode 2 : real photograph on a coupon ------------- */
    '  else if (uMode == 2) {',
    '    vec2 t = clamp(vUv, 0.0, 1.0);',
    '    vec3 a = texture2D(uMapA, t).rgb;',
    '    vec3 b = texture2D(uMapB, t).rgb;',
    '    albedo = mix(a, b, clamp(uMapMix, 0.0, 1.0));',
    /*   surface micro-roughness so the rake light still has something to
         catch on an otherwise flat photograph */
    '    float grain = fbm(vUv * 180.0) * 0.10;',
    '    albedo *= (0.94 + grain);',
    '    albedo = mix(albedo, uTintColor, uTintAmount * 0.30);',
    /*   travelling scan bar, driven by job progress */
    '    if (uScan.w > 0.5) {',
    '      float d = abs(t.y - uScan.x);',
    '      float band = 1.0 - smoothstep(0.0, uScan.y, d);',
    '      float edge = 1.0 - smoothstep(0.0, uScan.y * 0.16, d);',
    '      emissive += (vec3(0.78, 0.95, 0.24) * band * 0.30',
    '                 + vec3(0.85, 0.98, 0.55) * edge * 0.85) * uScan.z;',
    '    }',
    '  }',

    /* ---------------- mode 3 : cast concrete bay floor ---------------- */
    '  else if (uMode == 3) {',
    '    float c = fbm(uv * 3.0) * 0.35 + fbm(uv * 14.0) * 0.18;',
    '    albedo = uBaseColor * (0.72 + c);',
    /*   saw-cut control joints every unit */
    '    vec2 j = abs(fract(uv) - 0.5);',
    '    float cut = 1.0 - smoothstep(0.465, 0.498, max(j.x, j.y));',
    '    albedo = mix(albedo * 0.30, albedo, cut);',
    /*   hazard paint: a stripe band running along the bay, worn away */
    '    float lane = 1.0 - smoothstep(0.10, 0.16, abs(fract(uv.y * 0.5) - 0.5));',
    '    float chev = step(0.5, fract(uv.x * 3.0 + uv.y * 0.6));',
    '    float wear = smoothstep(0.35, 0.75, fbm(uv * 8.0));',
    '    vec3 paint = mix(vec3(0.06, 0.06, 0.07), vec3(0.85, 0.62, 0.10), chev);',
    '    albedo = mix(albedo, paint, lane * wear * uGridFade * 0.75);',
    '    float oil = smoothstep(0.62, 0.92, fbm(uv * 2.1 + 60.0));',
    '    albedo = mix(albedo, albedo * 0.35, oil * 0.7);',
    '    rough = clamp(rough - oil * 0.35, 0.05, 1.0);',
    '  }',

    /* ---------------- bump normal from the relief field ---------------- */
    '  if (relief > 0.5 && uBumpScale > 0.0) {',
    '    float e = 0.0035;',
    '    float h0 = reliefHeight(uv, uDefect, uScratchAngle);',
    '    float hx = reliefHeight(uv + vec2(e, 0.0), uDefect, uScratchAngle);',
    '    float hy = reliefHeight(uv + vec2(0.0, e), uDefect, uScratchAngle);',
    '    float dx = (hx - h0) / e;',
    '    float dy = (hy - h0) / e;',
    '    N = normalize(N - (uTangent * dx + uBitangent * dy) * uBumpScale);',
    '  }',

    /* ---------------- lighting ---------------- */
    '  float hemi = 0.5 + 0.5 * N.y;',
    '  vec3 col = albedo * mix(uAmbGround, uAmbSky, hemi);',
    '  float fill = max(dot(N, normalize(uFillDir)), 0.0);',
    '  col += albedo * uFillColor * fill;',

    '  float graze0 = 0.0, graze1 = 0.0;',
    '  col += spot(vWorld, N, V, uL0Pos, uL0Dir, uL0Color, uL0Cone,',
    '              albedo, rough, specK, graze0);',
    '  col += spot(vWorld, N, V, uL1Pos, uL1Dir, uL1Color, uL1Cone,',
    '              albedo, rough, specK, graze1);',

    /* The signature: defects only really appear under grazing light. The
       rake term lifts crack + pit contrast exactly where the beam skims. */
    '  if (uMode == 1) {',
    '    float rake = clamp(graze0 * 0.55 + graze1 * 1.45, 0.0, 2.4);',
    '    float reveal = clamp(uDefect.w, 0.0, 1.0);',
    '    float lift = rake * reveal;',
    '    col = mix(col, col * 0.16, crackMask * clamp(lift * 1.4, 0.0, 0.95));',
    '    col += vec3(0.95, 0.62, 0.22) * corrMask * lift * 0.32;',
    '    col += vec3(1.0, 1.0, 0.96) * scratchMask * lift * 0.42;',
    '  }',

    '  col += emissive;',

    /* ---------------- exponential-squared fog ---------------- */
    '  float dv = length(uCamera - vWorld);',
    '  float fogAmt = 1.0 - exp(-pow(dv * uFogDensity, 2.0));',
    '  col = mix(col, uFogColor, clamp(fogAmt, 0.0, 1.0));',

    '  gl_FragColor = vec4(col, uAlpha);',
    '}'
  ].join('\n');

  /* --- additive volumetric (light cones, floor pools, glows) ----------- */
  var VS_VOLUME = [
    'attribute vec3 aPosition;',
    'attribute vec3 aNormal;',
    'attribute vec2 aUv;',
    'uniform mat4 uProjection; uniform mat4 uView; uniform mat4 uModel;',
    'uniform mat3 uNormalMat;',
    'varying vec3 vWorld; varying vec3 vNormal; varying vec2 vUv;',
    'void main(){',
    '  vec4 wp = uModel * vec4(aPosition, 1.0);',
    '  vWorld = wp.xyz;',
    '  vNormal = normalize(uNormalMat * aNormal);',
    '  vUv = aUv;',
    '  gl_Position = uProjection * uView * wp;',
    '}'
  ].join('\n');

  var FS_VOLUME = [
    GLSL_COMMON,
    GLSL_NOISE,
    'varying vec3 vWorld; varying vec3 vNormal; varying vec2 vUv;',
    'uniform vec3 uCamera;',
    'uniform vec3 uColor;',
    'uniform float uAlpha;',
    'uniform float uTime;',
    'uniform int uShape;',   /* 0 cone shell · 1 radial disc · 2 soft card */
    'void main(){',
    '  vec3 V = normalize(uCamera - vWorld);',
    '  vec3 N = normalize(vNormal);',
    '  float a = uAlpha;',
    '  if (uShape == 0) {',
    /*   fade toward the far end of the beam and soften the silhouette so the
         cone reads as haze rather than a solid cardboard funnel */
    '    a *= pow(clamp(vUv.y, 0.0, 1.0), 1.25);',
    '    a *= pow(abs(dot(N, V)), 1.35);',
    '    a *= 0.75 + 0.25 * vnoise(vec2(vUv.x * 6.0, vUv.y * 3.0 - uTime * 0.35));',
    '  } else if (uShape == 1) {',
    '    float r = length(vUv - 0.5) * 2.0;',
    '    a *= pow(clamp(1.0 - r, 0.0, 1.0), 2.1);',
    '    a *= 0.85 + 0.15 * vnoise(vUv * 9.0 + uTime * 0.2);',
    '  } else {',
    '    vec2 c = vUv - 0.5;',
    '    a *= pow(clamp(1.0 - length(c) * 2.0, 0.0, 1.0), 2.6);',
    '  }',
    '  gl_FragColor = vec4(uColor * a, a);',
    '}'
  ].join('\n');

  /* --- dust motes -------------------------------------------------------
     Points that brighten when they drift into a beam. The cone test happens
     per-vertex, which is plenty for 2-3 px sprites.                        */
  var VS_DUST = [
    'attribute vec3 aPosition;',
    'attribute vec3 aNormal;',   /* .x = phase, .y = size, .z = speed */
    'uniform mat4 uProjection; uniform mat4 uView;',
    'uniform float uTime; uniform float uPixelScale;',
    'uniform vec3 uBounds; uniform vec3 uOrigin;',
    'uniform vec3 uL1Pos; uniform vec3 uL1Dir; uniform vec4 uL1Cone;',
    'uniform vec3 uL0Pos; uniform vec3 uL0Dir; uniform vec4 uL0Cone;',
    'varying float vGlow;',
    'float coneTest(vec3 P, vec3 lp, vec3 ld, vec4 cone){',
    '  vec3 L = lp - P; float d = length(L);',
    '  if (d < 0.0001) return 0.0;',
    '  L /= d;',
    '  float cd = dot(-L, normalize(ld));',
    '  return smoothstep(cone.y, cone.x, cd) * cone.z / (1.0 + cone.w * d * d);',
    '}',
    'void main(){',
    '  float t = uTime * aNormal.z;',
    '  vec3 p = aPosition;',
    /*  slow convective drift, wrapped inside the bay volume */
    '  p.x += sin(t * 0.42 + aNormal.x * 6.28) * 0.55;',
    '  p.y = uOrigin.y + mod(p.y - uOrigin.y + t * 0.11, uBounds.y);',
    '  p.z += cos(t * 0.31 + aNormal.x * 4.11) * 0.45;',
    '  vec4 vp = uView * vec4(p, 1.0);',
    '  gl_Position = uProjection * vp;',
    '  gl_PointSize = clamp(aNormal.y * uPixelScale / max(-vp.z, 0.4), 1.0, 7.0);',
    '  vGlow = coneTest(p, uL1Pos, uL1Dir, uL1Cone) * 1.4',
    '        + coneTest(p, uL0Pos, uL0Dir, uL0Cone) * 0.9;',
    '}'
  ].join('\n');

  var FS_DUST = [
    GLSL_COMMON,
    'varying float vGlow;',
    'uniform vec3 uColor; uniform float uAlpha;',
    'void main(){',
    '  vec2 c = gl_PointCoord - 0.5;',
    '  float r = length(c) * 2.0;',
    '  float a = pow(clamp(1.0 - r, 0.0, 1.0), 1.8) * uAlpha * clamp(vGlow, 0.0, 2.0);',
    '  if (a < 0.004) discard;',
    '  gl_FragColor = vec4(uColor * a, a);',
    '}'
  ].join('\n');

  /* --- post: bright-pass + separable blur ------------------------------ */
  var VS_POST = [
    'attribute vec2 aPosition;',
    'varying vec2 vUv;',
    'void main(){',
    '  vUv = aPosition * 0.5 + 0.5;',
    '  gl_Position = vec4(aPosition, 0.0, 1.0);',
    '}'
  ].join('\n');

  var FS_BLUR = [
    GLSL_COMMON,
    'varying vec2 vUv;',
    'uniform sampler2D uTex;',
    'uniform vec2 uStep;',
    'uniform float uThreshold;',   /* > 0 → also bright-pass */
    'void main(){',
    '  float w[5];',
    '  w[0] = 0.227027; w[1] = 0.194595; w[2] = 0.121622;',
    '  w[3] = 0.054054; w[4] = 0.016216;',
    '  vec3 sum = texture2D(uTex, vUv).rgb * w[0];',
    '  for (int i = 1; i < 5; i++) {',
    '    vec2 o = uStep * float(i);',
    '    sum += texture2D(uTex, vUv + o).rgb * w[i];',
    '    sum += texture2D(uTex, vUv - o).rgb * w[i];',
    '  }',
    '  if (uThreshold > 0.0) {',
    '    float l = dot(sum, vec3(0.2126, 0.7152, 0.0722));',
    '    sum *= smoothstep(uThreshold, uThreshold + 0.45, l);',
    '  }',
    '  gl_FragColor = vec4(sum, 1.0);',
    '}'
  ].join('\n');

  var FS_COMPOSITE = [
    GLSL_COMMON,
    'varying vec2 vUv;',
    'uniform sampler2D uScene;',
    'uniform sampler2D uBloom;',
    'uniform float uBloomAmount;',
    'uniform float uExposure;',
    'uniform float uVignette;',
    'uniform float uChroma;',
    'uniform float uGrain;',
    'uniform float uTime;',
    'uniform float uFlash;',
    'uniform vec3  uFlashColor;',
    'uniform float uScanlines;',
    'uniform vec2  uResolution;',
    'float rnd(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }',
    'void main(){',
    '  vec2 uv = vUv;',
    '  vec2 c = uv - 0.5;',
    '  float r2 = dot(c, c);',
    /*  lateral chromatic aberration, strongest at the edge of the field */
    '  float k = uChroma * r2;',
    '  vec3 col;',
    '  col.r = texture2D(uScene, uv + c * k).r;',
    '  col.g = texture2D(uScene, uv).g;',
    '  col.b = texture2D(uScene, uv - c * k).b;',
    '  col += texture2D(uBloom, uv).rgb * uBloomAmount;',
    '  col *= uExposure;',
    /*  filmic-ish shoulder so the sodium lamp rolls off instead of clipping */
    '  col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);',
    '  col = clamp(col, 0.0, 1.0);',
    '  col = mix(col, uFlashColor, clamp(uFlash, 0.0, 1.0));',
    '  float vig = 1.0 - uVignette * smoothstep(0.12, 0.78, r2);',
    '  col *= vig;',
    '  if (uScanlines > 0.0) {',
    '    float s = 0.5 + 0.5 * sin(uv.y * uResolution.y * 1.4);',
    '    col *= 1.0 - uScanlines * s;',
    '  }',
    '  float g = rnd(uv * uResolution + fract(uTime) * 91.7) - 0.5;',
    '  col += g * uGrain;',
    '  gl_FragColor = vec4(col, 1.0);',
    '}'
  ].join('\n');

  /* ======================================================================
     5. PROGRAM WRAPPER
     ====================================================================== */

  function Program(gl, vsSrc, fsSrc, name) {
    this.gl = gl;
    this.name = name || 'program';
    this.program = null;
    this._uniforms = {};
    this._attribs = {};

    var vs = this._compile(gl.VERTEX_SHADER, vsSrc);
    var fs = this._compile(gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return;

    var p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      /* eslint-disable-next-line no-console */
      console.error('[nf-gl] link failed (' + this.name + '): ' +
        gl.getProgramInfoLog(p));
      gl.deleteProgram(p);
      return;
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    this.program = p;
  }

  Program.prototype._compile = function (type, src) {
    var gl = this.gl;
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      /* eslint-disable-next-line no-console */
      console.error('[nf-gl] compile failed (' + this.name + '):\n' +
        gl.getShaderInfoLog(sh));
      gl.deleteShader(sh);
      return null;
    }
    return sh;
  };

  Program.prototype.use = function () {
    if (this.program) this.gl.useProgram(this.program);
    return this;
  };

  Program.prototype.uniform = function (name) {
    if (!(name in this._uniforms)) {
      this._uniforms[name] = this.gl.getUniformLocation(this.program, name);
    }
    return this._uniforms[name];
  };

  Program.prototype.attrib = function (name) {
    if (!(name in this._attribs)) {
      this._attribs[name] = this.gl.getAttribLocation(this.program, name);
    }
    return this._attribs[name];
  };

  /* Set a bag of uniforms by inferring the setter from the JS value type.
     Missing names resolve to null, which WebGL silently ignores — so a
     shader can drop a uniform without breaking every call site. */
  Program.prototype.set = function (values) {
    var gl = this.gl;
    for (var name in values) {
      if (!Object.prototype.hasOwnProperty.call(values, name)) continue;
      var loc = this.uniform(name);
      if (loc === null) continue;
      var v = values[name];
      if (typeof v === 'number') {
        gl.uniform1f(loc, v);
      } else if (typeof v === 'boolean') {
        gl.uniform1f(loc, v ? 1 : 0);
      } else if (v && v.__int !== undefined) {
        gl.uniform1i(loc, v.__int);
      } else if (v && v.__tex !== undefined) {
        gl.activeTexture(gl.TEXTURE0 + v.__unit);
        gl.bindTexture(gl.TEXTURE_2D, v.__tex);
        gl.uniform1i(loc, v.__unit);
      } else if (v && v.length === 2) {
        gl.uniform2fv(loc, v);
      } else if (v && v.length === 3) {
        gl.uniform3fv(loc, v);
      } else if (v && v.length === 4) {
        gl.uniform4fv(loc, v);
      } else if (v && v.length === 9) {
        gl.uniformMatrix3fv(loc, false, v);
      } else if (v && v.length === 16) {
        gl.uniformMatrix4fv(loc, false, v);
      }
    }
    return this;
  };

  function asInt(n) { return { __int: n | 0 }; }
  function asTex(tex, unit) { return { __tex: tex, __unit: unit | 0 }; }

  /* ======================================================================
     6. MESH (interleaved-free: one buffer per attribute)
     ====================================================================== */

  function Mesh(gl, geo) {
    this.gl = gl;
    this.count = geo.index.length;
    this.indexType = (geo.index instanceof Uint32Array)
      ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    this.position = this._buf(gl.ARRAY_BUFFER, geo.position);
    this.normal = this._buf(gl.ARRAY_BUFFER, geo.normal);
    this.uv = this._buf(gl.ARRAY_BUFFER, geo.uv);
    this.index = this._buf(gl.ELEMENT_ARRAY_BUFFER, geo.index);
  }

  Mesh.prototype._buf = function (target, data) {
    var gl = this.gl;
    var b = gl.createBuffer();
    gl.bindBuffer(target, b);
    gl.bufferData(target, data, gl.STATIC_DRAW);
    return b;
  };

  Mesh.prototype.bind = function (prog) {
    var gl = this.gl;
    var aP = prog.attrib('aPosition');
    var aN = prog.attrib('aNormal');
    var aU = prog.attrib('aUv');
    if (aP >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.position);
      gl.enableVertexAttribArray(aP);
      gl.vertexAttribPointer(aP, 3, gl.FLOAT, false, 0, 0);
    }
    if (aN >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.normal);
      gl.enableVertexAttribArray(aN);
      gl.vertexAttribPointer(aN, 3, gl.FLOAT, false, 0, 0);
    }
    if (aU >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uv);
      gl.enableVertexAttribArray(aU);
      gl.vertexAttribPointer(aU, 2, gl.FLOAT, false, 0, 0);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.index);
    return this;
  };

  Mesh.prototype.draw = function () {
    this.gl.drawElements(this.gl.TRIANGLES, this.count, this.indexType, 0);
  };

  /* ======================================================================
     7. RENDER TARGET
     ====================================================================== */

  function Target(gl, w, h, depth) {
    this.gl = gl;
    this.width = Math.max(1, w | 0);
    this.height = Math.max(1, h | 0);
    this.ok = false;

    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D, this.texture, 0);

    this.depth = null;
    if (depth) {
      this.depth = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, this.depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16,
        this.width, this.height);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
        gl.RENDERBUFFER, this.depth);
    }

    this.ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  Target.prototype.dispose = function () {
    var gl = this.gl;
    if (this.texture) gl.deleteTexture(this.texture);
    if (this.depth) gl.deleteRenderbuffer(this.depth);
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
  };

  /* ======================================================================
     8. RENDERER
     ====================================================================== */

  function Renderer(canvas, opts) {
    opts = opts || {};
    this.canvas = canvas;
    this.maxDpr = opts.maxDpr || 1.75;
    this.wantPost = opts.post !== false;

    var attribs = {
      alpha: false,
      antialias: !this.wantPost,   /* MSAA is pointless when we go via an FBO */
      depth: true,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
      failIfMajorPerformanceCaveat: false
    };
    var gl = canvas.getContext('webgl', attribs)
          || canvas.getContext('experimental-webgl', attribs);
    if (!gl) { this.gl = null; return; }
    this.gl = gl;

    this.progSurface = new Program(gl, VS_SURFACE, FS_SURFACE, 'surface');
    this.progVolume = new Program(gl, VS_VOLUME, FS_VOLUME, 'volume');
    this.progDust = new Program(gl, VS_DUST, FS_DUST, 'dust');
    this.progBlur = new Program(gl, VS_POST, FS_BLUR, 'blur');
    this.progComp = new Program(gl, VS_POST, FS_COMPOSITE, 'composite');

    if (!this.progSurface.program) { this.gl = null; return; }
    if (!this.progBlur.program || !this.progComp.program) this.wantPost = false;

    /* full-screen triangle pair for the post passes */
    this.quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW);

    /* 1x1 neutral texture so `mapped` mode is always safe to bind */
    this.blankTex = this.createSolidTexture(20, 22, 25);

    this.view = Mat4.create();
    this.projection = Mat4.create();
    this.viewProjection = Mat4.create();
    this.camera = Vec3.create(0, 1.6, 6);
    this._model = Mat4.create();
    this._nrm = new Float32Array(9);

    this.width = 1; this.height = 1; this.dpr = 1;
    /* seeded so project() is safe even if resize() early-returns on a
       zero-sized canvas during the first frame */
    this.cssWidth = 1; this.cssHeight = 1;
    this.sceneTarget = null;
    this.bloomA = null;
    this.bloomB = null;
    this.postOK = false;

    /* ---- default booth lighting rig ---- */
    this.lights = {
      ambSky: new Float32Array([0.055, 0.070, 0.085]),
      ambGround: new Float32Array([0.016, 0.018, 0.022]),
      fillDir: Vec3.normalize(Vec3.create(), Vec3.create(-0.35, 0.75, 0.45)),
      fillColor: new Float32Array([0.045, 0.058, 0.075]),
      /* L0 — ceiling sodium work-lamp */
      l0: {
        pos: Vec3.create(0, 5.2, 0),
        dir: Vec3.create(0, -1, 0),
        color: new Float32Array([1.0, 0.68, 0.30]),
        cone: new Float32Array([0.92, 0.62, 0.0, 0.055])
      },
      /* L1 — the handheld raking lamp: the signature light */
      l1: {
        pos: Vec3.create(-1.4, 0.55, 1.1),
        dir: Vec3.create(0.75, -0.30, -0.55),
        color: new Float32Array([0.86, 0.94, 0.62]),
        cone: new Float32Array([0.975, 0.80, 0.0, 0.10])
      }
    };

    this.fog = {
      color: new Float32Array([0.035, 0.041, 0.048]),
      density: 0.055
    };

    this.post = {
      exposure: 1.06,
      bloom: 0.62,
      bloomThreshold: 0.62,
      vignette: 0.55,
      chroma: 0.16,
      grain: 0.045,
      scanlines: 0.0,
      flash: 0.0,
      flashColor: new Float32Array([1, 1, 1])
    };

    this.time = 0;
    this._meshCache = {};

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.clearColor(this.fog.color[0], this.fog.color[1], this.fog.color[2], 1);

    this.resize();
  }

  Renderer.prototype.mesh = function (key, build) {
    if (!this._meshCache[key]) {
      this._meshCache[key] = new Mesh(this.gl, build());
    }
    return this._meshCache[key];
  };

  Renderer.prototype.createSolidTexture = function (r, g, b) {
    var gl = this.gl;
    var t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA,
      gl.UNSIGNED_BYTE, new Uint8Array([r, g, b, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  };

  /* Upload an <img>/<canvas>. NPOT-safe: clamped, linear, no mipmaps. */
  Renderer.prototype.createImageTexture = function (source, existing) {
    var gl = this.gl;
    var t = existing || gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA,
        gl.UNSIGNED_BYTE, source);
    } catch (e) {
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      return null;
    }
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  };

  Renderer.prototype.resize = function () {
    var gl = this.gl;
    if (!gl) return false;
    var dpr = Math.min(global.devicePixelRatio || 1, this.maxDpr);
    var rect = this.canvas.getBoundingClientRect();
    var cssW = Math.max(1, Math.round(rect.width || this.canvas.clientWidth || 1));
    var cssH = Math.max(1, Math.round(rect.height || this.canvas.clientHeight || 1));
    var w = Math.max(1, Math.round(cssW * dpr));
    var h = Math.max(1, Math.round(cssH * dpr));
    if (w === this.width && h === this.height && dpr === this.dpr) return false;

    this.dpr = dpr;
    this.width = w;
    this.height = h;
    this.cssWidth = cssW;
    this.cssHeight = cssH;
    this.canvas.width = w;
    this.canvas.height = h;

    if (this.wantPost) {
      if (this.sceneTarget) this.sceneTarget.dispose();
      if (this.bloomA) this.bloomA.dispose();
      if (this.bloomB) this.bloomB.dispose();
      var bw = Math.max(1, w >> 1), bh = Math.max(1, h >> 1);
      this.sceneTarget = new Target(gl, w, h, true);
      this.bloomA = new Target(gl, bw, bh, false);
      this.bloomB = new Target(gl, bw, bh, false);
      this.postOK = this.sceneTarget.ok && this.bloomA.ok && this.bloomB.ok;
    }
    return true;
  };

  Renderer.prototype.setCamera = function (eye, target, up, fovDeg, near, far) {
    Vec3.copy(this.camera, eye);
    Mat4.lookAt(this.view, eye, target, up || Vec3.create(0, 1, 0));
    var aspect = this.width / Math.max(1, this.height);
    Mat4.perspective(this.projection, (fovDeg || 42) * Math.PI / 180, aspect,
      near || 0.08, far || 220);
    Mat4.multiply(this.viewProjection, this.projection, this.view);
  };

  /* World point → CSS pixel coordinates, or null if behind the camera. */
  Renderer.prototype.project = function (worldPoint) {
    var m = this.viewProjection;
    var x = worldPoint[0], y = worldPoint[1], z = worldPoint[2];
    var cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    var cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    var cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (cw <= 0.0001) return null;
    return {
      x: (cx / cw * 0.5 + 0.5) * this.cssWidth,
      y: (1 - (cy / cw * 0.5 + 0.5)) * this.cssHeight,
      depth: cw
    };
  };

  Renderer.prototype.beginScene = function () {
    var gl = this.gl;
    if (this.postOK) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneTarget.fbo);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    gl.viewport(0, 0, this.width, this.height);
    gl.clearColor(this.fog.color[0], this.fog.color[1], this.fog.color[2], 1);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    this._lightUniformsCache = null;
  };

  Renderer.prototype._lightUniforms = function () {
    if (this._lightUniformsCache) return this._lightUniformsCache;
    var L = this.lights;
    this._lightUniformsCache = {
      uCamera: this.camera,
      uView: this.view,
      uProjection: this.projection,
      uAmbSky: L.ambSky,
      uAmbGround: L.ambGround,
      uFillDir: L.fillDir,
      uFillColor: L.fillColor,
      uL0Pos: L.l0.pos, uL0Dir: L.l0.dir, uL0Color: L.l0.color, uL0Cone: L.l0.cone,
      uL1Pos: L.l1.pos, uL1Dir: L.l1.dir, uL1Color: L.l1.color, uL1Cone: L.l1.cone,
      uFogColor: this.fog.color,
      uFogDensity: this.fog.density,
      uTime: this.time
    };
    return this._lightUniformsCache;
  };

  var DEFAULT_MAT = {
    mode: 0,
    color: [0.5, 0.5, 0.52],
    roughness: 0.6,
    specular: 0.35,
    emissive: [0, 0, 0],
    alpha: 1,
    uvScale: [1, 1],
    defect: [0, 0, 0, 0],
    scratchAngle: 0.35,
    tangent: [1, 0, 0],
    bitangent: [0, 0, 1],
    bump: 0,
    mapA: null,
    mapB: null,
    mapMix: 0,
    tintColor: [1, 1, 1],
    tintAmount: 0,
    scan: [0, 0.05, 0, 0],
    gridFade: 1,
    cull: true
  };

  /* Draw an opaque/solid surface. `mat` is sparse — anything omitted falls
     back to DEFAULT_MAT. */
  Renderer.prototype.drawSurface = function (mesh, model, mat) {
    var gl = this.gl;
    var p = this.progSurface;
    mat = mat || {};
    function get(k) { return mat[k] !== undefined ? mat[k] : DEFAULT_MAT[k]; }

    p.use();
    Mat4.normalMatrix(this._nrm, model);

    if (get('cull')) gl.enable(gl.CULL_FACE); else gl.disable(gl.CULL_FACE);

    var alpha = get('alpha');
    if (alpha < 0.999) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
    } else {
      gl.disable(gl.BLEND);
      gl.depthMask(true);
    }

    p.set(this._lightUniforms());
    p.set({
      uModel: model,
      uNormalMat: this._nrm,
      uMode: asInt(get('mode')),
      uBaseColor: get('color'),
      uSurface: [get('roughness'), get('specular')],
      uEmissive: get('emissive'),
      uAlpha: alpha,
      uDefect: get('defect'),
      uUvScale: get('uvScale'),
      uScratchAngle: get('scratchAngle'),
      uTangent: get('tangent'),
      uBitangent: get('bitangent'),
      uBumpScale: get('bump'),
      uMapA: asTex(get('mapA') || this.blankTex, 0),
      uMapB: asTex(get('mapB') || get('mapA') || this.blankTex, 1),
      uMapMix: get('mapMix'),
      uTintColor: get('tintColor'),
      uTintAmount: get('tintAmount'),
      uScan: get('scan'),
      uGridFade: get('gridFade')
    });
    mesh.bind(p).draw();
    gl.depthMask(true);
  };

  /* Additive haze: light cones (shape 0), floor pools (1), soft cards (2). */
  Renderer.prototype.drawVolume = function (mesh, model, color, alpha, shape) {
    var gl = this.gl;
    var p = this.progVolume;
    if (!p.program) return;
    p.use();
    Mat4.normalMatrix(this._nrm, model);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.depthMask(false);
    gl.disable(gl.CULL_FACE);
    p.set({
      uProjection: this.projection,
      uView: this.view,
      uModel: model,
      uNormalMat: this._nrm,
      uCamera: this.camera,
      uColor: color,
      uAlpha: alpha,
      uTime: this.time,
      uShape: asInt(shape || 0)
    });
    mesh.bind(p).draw();
    gl.depthMask(true);
    gl.enable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
  };

  /* ---- dust motes ---- */
  Renderer.prototype.createDust = function (count, bounds, origin) {
    var gl = this.gl;
    var pos = new Float32Array(count * 3);
    var att = new Float32Array(count * 3);
    for (var i = 0; i < count; i++) {
      pos[i * 3] = origin[0] + (Math.random() - 0.5) * bounds[0];
      pos[i * 3 + 1] = origin[1] + Math.random() * bounds[1];
      pos[i * 3 + 2] = origin[2] + (Math.random() - 0.5) * bounds[2];
      att[i * 3] = Math.random();                    /* phase */
      att[i * 3 + 1] = 14 + Math.random() * 34;      /* size  */
      att[i * 3 + 2] = 0.4 + Math.random() * 1.1;    /* speed */
    }
    var pb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pb);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    var ab = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, ab);
    gl.bufferData(gl.ARRAY_BUFFER, att, gl.STATIC_DRAW);
    return {
      count: count, posBuf: pb, attBuf: ab,
      bounds: new Float32Array(bounds), origin: new Float32Array(origin)
    };
  };

  Renderer.prototype.drawDust = function (dust, color, alpha) {
    var gl = this.gl;
    var p = this.progDust;
    if (!p.program || !dust) return;
    p.use();
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.depthMask(false);
    var L = this.lights;
    p.set({
      uProjection: this.projection,
      uView: this.view,
      uTime: this.time,
      uPixelScale: this.height * 0.05,
      uBounds: dust.bounds,
      uOrigin: dust.origin,
      uColor: color,
      uAlpha: alpha,
      uL0Pos: L.l0.pos, uL0Dir: L.l0.dir, uL0Cone: L.l0.cone,
      uL1Pos: L.l1.pos, uL1Dir: L.l1.dir, uL1Cone: L.l1.cone
    });
    var aP = p.attrib('aPosition');
    var aN = p.attrib('aNormal');
    if (aP >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, dust.posBuf);
      gl.enableVertexAttribArray(aP);
      gl.vertexAttribPointer(aP, 3, gl.FLOAT, false, 0, 0);
    }
    if (aN >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, dust.attBuf);
      gl.enableVertexAttribArray(aN);
      gl.vertexAttribPointer(aN, 3, gl.FLOAT, false, 0, 0);
    }
    gl.drawArrays(gl.POINTS, 0, dust.count);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  };

  Renderer.prototype._drawQuad = function (prog) {
    var gl = this.gl;
    var a = prog.attrib('aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    if (a >= 0) {
      gl.enableVertexAttribArray(a);
      gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  };

  Renderer.prototype.endScene = function () {
    var gl = this.gl;
    if (!this.postOK) return;

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);

    /* bright-pass + horizontal blur → bloomA */
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomA.fbo);
    gl.viewport(0, 0, this.bloomA.width, this.bloomA.height);
    this.progBlur.use().set({
      uTex: asTex(this.sceneTarget.texture, 0),
      uStep: [1.35 / this.bloomA.width, 0],
      uThreshold: this.post.bloomThreshold
    });
    this._drawQuad(this.progBlur);

    /* vertical blur → bloomB */
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomB.fbo);
    gl.viewport(0, 0, this.bloomB.width, this.bloomB.height);
    this.progBlur.use().set({
      uTex: asTex(this.bloomA.texture, 0),
      uStep: [0, 1.35 / this.bloomB.height],
      uThreshold: 0
    });
    this._drawQuad(this.progBlur);

    /* second widening pass, back into bloomA */
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomA.fbo);
    gl.viewport(0, 0, this.bloomA.width, this.bloomA.height);
    this.progBlur.use().set({
      uTex: asTex(this.bloomB.texture, 0),
      uStep: [3.1 / this.bloomA.width, 0],
      uThreshold: 0
    });
    this._drawQuad(this.progBlur);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomB.fbo);
    gl.viewport(0, 0, this.bloomB.width, this.bloomB.height);
    this.progBlur.use().set({
      uTex: asTex(this.bloomA.texture, 0),
      uStep: [0, 3.1 / this.bloomB.height],
      uThreshold: 0
    });
    this._drawQuad(this.progBlur);

    /* composite to the canvas */
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    this.progComp.use().set({
      uScene: asTex(this.sceneTarget.texture, 0),
      uBloom: asTex(this.bloomB.texture, 1),
      uBloomAmount: this.post.bloom,
      uExposure: this.post.exposure,
      uVignette: this.post.vignette,
      uChroma: this.post.chroma,
      uGrain: this.post.grain,
      uScanlines: this.post.scanlines,
      uFlash: this.post.flash,
      uFlashColor: this.post.flashColor,
      uTime: this.time,
      uResolution: [this.width, this.height]
    });
    this._drawQuad(this.progComp);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
  };

  /* ======================================================================
     9. NODE — a tiny transform helper for articulated rigs
     ====================================================================== */

  function Node(name) {
    this.name = name || '';
    this.position = Vec3.create(0, 0, 0);
    this.rotation = Vec3.create(0, 0, 0);
    this.scale = Vec3.create(1, 1, 1);
    this.local = Mat4.create();
    this.world = Mat4.create();
    this.children = [];
    this.parent = null;
    this.mesh = null;
    this.material = null;
    this.visible = true;
  }

  Node.prototype.add = function (child) {
    child.parent = this;
    this.children.push(child);
    return child;
  };

  Node.prototype.set = function (px, py, pz) {
    Vec3.set(this.position, px, py, pz);
    return this;
  };

  Node.prototype.rotate = function (rx, ry, rz) {
    Vec3.set(this.rotation, rx, ry, rz);
    return this;
  };

  Node.prototype.sized = function (sx, sy, sz) {
    Vec3.set(this.scale, sx, sy, sz);
    return this;
  };

  Node.prototype.updateWorld = function (parentWorld) {
    Mat4.compose(this.local, this.position, this.rotation, this.scale);
    if (parentWorld) {
      Mat4.multiply(this.world, parentWorld, this.local);
    } else {
      Mat4.copy(this.world, this.local);
    }
    for (var i = 0; i < this.children.length; i++) {
      this.children[i].updateWorld(this.world);
    }
    return this;
  };

  Node.prototype.render = function (renderer) {
    if (!this.visible) return;
    if (this.mesh) renderer.drawSurface(this.mesh, this.world, this.material);
    for (var i = 0; i < this.children.length; i++) {
      this.children[i].render(renderer);
    }
  };

  Node.prototype.worldPosition = function (out) {
    out = out || Vec3.create();
    out[0] = this.world[12]; out[1] = this.world[13]; out[2] = this.world[14];
    return out;
  };

  /* ======================================================================
     10. EXPORTS
     ====================================================================== */

  global.NFGL = {
    Vec3: Vec3,
    Mat4: Mat4,
    Ease: Ease,
    Geo: Geo,
    Mesh: Mesh,
    Node: Node,
    Program: Program,
    Target: Target,
    Renderer: Renderer,
    hex: hex,
    tint: tint,
    clamp: clamp,
    clamp01: clamp01,
    lerp: lerp,
    span: span,
    asInt: asInt,
    asTex: asTex,
    supported: (function () {
      try {
        var c = document.createElement('canvas');
        return !!(global.WebGLRenderingContext &&
          (c.getContext('webgl') || c.getContext('experimental-webgl')));
      } catch (e) { return false; }
    })()
  };
})(window);
