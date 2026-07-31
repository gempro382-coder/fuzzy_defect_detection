/* ==========================================================================
   nf-intro.js — "Bay 04", the opening cinematic
   --------------------------------------------------------------------------
   A real-time 3D short that plays before the interface appears. Four shots:

     S H O T   A   06:30. The roller shutter lifts on a dark inspection bay
                   and two engineers step into the daylight wedge.
     S H O T   B   They cross to the bench where the lot is laid out. The
                   sodium work-lamp strikes on. Under flat overhead light the
                   coupons look fine.
     S H O T   C   One of them lifts the raking lamp and skims it across the
                   surface at a few degrees. The crack network and the oxide
                   bloom come out of nowhere. Reticles lock on.
     S H O T   D   Pull back. Everything falls away except the coupon, and
                   the title resolves.

   The whole thing is one argument: grazing light shows you what flat light
   hides — which is exactly what the fuzzy engine does with its features.
   ========================================================================== */
(function (global) {
  "use strict";

  var GL = global.NFGL;
  if (!GL) return;

  var Vec3 = GL.Vec3, Mat4 = GL.Mat4, Geo = GL.Geo, Node = GL.Node;
  var Ease = GL.Ease, span = GL.span, lerp = GL.lerp;
  var clamp = GL.clamp, clamp01 = GL.clamp01, hex = GL.hex;

  /* ======================================================================
     PALETTE — every colour is named for the thing it actually is
     ====================================================================== */
  var C = {
    concrete:  hex('#3A3E42'),
    wallPaint: hex('#2B3238'),
    wallLower: hex('#1D262C'),
    steelDark: hex('#20272D'),
    benchTop:  hex('#3D4750'),
    shutter:   hex('#4A5259'),
    daylight:  hex('#BFD8E8'),
    sodium:    hex('#FFB03A'),
    penetrant: hex('#C8F135'),
    phosphor:  hex('#4FD8E8'),
    ceramic:   hex('#C3C0B4'),
    hiVis:     hex('#D8E840'),
    hardHat:   hex('#E8E2D6'),
    hardHat2:  hex('#E0872F'),
    trouser:   hex('#232C36'),
    skin:      hex('#6B5A4E'),
    boot:      hex('#14181C'),
    lampBody:  hex('#2E353B')
  };

  /* Timeline anchors (seconds). Kept in one place so the shot list reads
     like a shot list. */
  var T = {
    total:   16.40,
    aIn:      0.55,
    shutter0: 0.90, shutter1: 3.05,
    walkIn:   2.85,
    cutB:     4.30,
    lampOn:   5.55,
    arriveAt: 7.60,
    cutC:     8.10,
    rake0:    8.90, rake1: 12.10,
    retic1:  10.15, retic2: 10.95,
    cutD:    12.65,
    title:   13.55,
    outro:   15.70
  };

  var CAPTIONS = [
    { t0: T.aIn,  t1: T.cutB,  code: 'BAY 04 · 06:30 · SHIFT 1',
      line: 'Lot 8821 comes in for surface inspection.' },
    { t0: T.cutB, t1: T.cutC,  code: '240 CERAMIC COUPONS',
      line: 'Under flat overhead light almost every one of them looks sound.' },
    { t0: T.cutC, t1: T.cutD,  code: 'RAKING LAMP · 6° INCIDENCE',
      line: 'Skim the light across the surface and it gives itself up.' },
    { t0: T.cutD, t1: T.total, code: 'CRACK · OXIDE BLOOM · SCORING',
      line: 'Two engines read the same surface. One verdict comes out.' }
  ];

  /* Camera path. Each key is a cut-free segment; cuts are the moments where
     two keys share a time and the position jumps. */
  var CAM = [
    { t: 0.00,     eye: [3.90, 1.78, 6.90], at: [0.30, 1.62, 9.40], fov: 38 },
    { t: T.cutB,   eye: [2.85, 1.70, 5.05], at: [0.45, 1.44, 9.10], fov: 38 },

    { t: T.cutB,   eye: [4.55, 2.42, 3.55], at: [0.55, 1.05, 0.10], fov: 44 },
    { t: T.cutC,   eye: [2.75, 1.86, 1.35], at: [0.15, 0.92, -0.60], fov: 44 },

    { t: T.cutC,   eye: [0.86, 1.06, 0.42], at: [0.02, 0.775, -0.62], fov: 34 },
    { t: T.cutD,   eye: [0.30, 0.93, 0.06], at: [0.00, 0.780, -0.62], fov: 30 },

    { t: T.cutD,   eye: [0.34, 0.98, 0.22], at: [0.00, 0.780, -0.62], fov: 33 },
    { t: T.total,  eye: [1.55, 1.72, 2.65], at: [0.05, 0.860, -0.70], fov: 42 }
  ];

  /* ======================================================================
     CAMERA HELPERS
     ====================================================================== */

  function sampleCam(t) {
    var i;
    for (i = 0; i < CAM.length - 1; i++) {
      if (t <= CAM[i + 1].t) break;
    }
    i = Math.min(i, CAM.length - 2);
    var a = CAM[i], b = CAM[i + 1];
    var d = b.t - a.t;
    var u = d <= 0 ? 0 : Ease.inOutSine(clamp01((t - a.t) / d));
    return {
      eye: [lerp(a.eye[0], b.eye[0], u), lerp(a.eye[1], b.eye[1], u), lerp(a.eye[2], b.eye[2], u)],
      at:  [lerp(a.at[0], b.at[0], u),   lerp(a.at[1], b.at[1], u),   lerp(a.at[2], b.at[2], u)],
      fov: lerp(a.fov, b.fov, u)
    };
  }

  /* A shoulder-rig wobble so the shots read as filmed rather than rendered. */
  function handheld(t, amount) {
    return [
      (Math.sin(t * 1.31) * 0.6 + Math.sin(t * 2.77) * 0.4) * amount,
      (Math.sin(t * 1.07 + 2.0) * 0.5 + Math.sin(t * 3.13 + 1.0) * 0.3) * amount,
      (Math.sin(t * 0.89 + 4.0) * 0.4) * amount
    ];
  }

  /* Short dip to black across each cut — the eye reads it as an edit. */
  function cutDip(t) {
    var cuts = [T.cutB, T.cutC, T.cutD];
    var dip = 0;
    for (var i = 0; i < cuts.length; i++) {
      var d = Math.abs(t - cuts[i]);
      if (d < 0.13) dip = Math.max(dip, 1 - d / 0.13);
    }
    return dip;
  }

  /* ======================================================================
     THE ENGINEER RIG
     Boxes and cylinders on a joint hierarchy. Every limb node sits AT its
     joint and carries its visual as an offset child, so rotating the node
     swings the limb from the correct pivot.
     ====================================================================== */

  function buildEngineer(R, opts) {
    var boxM = R.mesh('box', function () { return Geo.box(1, 1, 1); });
    var cylM = R.mesh('cyl', function () { return Geo.cylinder(0.5, 0.5, 1, 14, true); });
    var sphM = R.mesh('sph', function () { return Geo.sphere(0.5, 16, 12); });

    var matSkin   = { color: C.skin, roughness: 0.85, specular: 0.12 };
    var matVest   = { color: opts.vest, roughness: 0.62, specular: 0.30 };
    var matShirt  = { color: opts.shirt, roughness: 0.80, specular: 0.14 };
    var matLeg    = { color: C.trouser, roughness: 0.86, specular: 0.10 };
    var matBoot   = { color: C.boot, roughness: 0.72, specular: 0.22 };
    var matHat    = { color: opts.hat, roughness: 0.42, specular: 0.55 };

    var root = new Node('engineer');
    var hips = root.add(new Node('hips')).set(0, 0.92, 0);

    /* --- torso --- */
    var torso = hips.add(new Node('torso')).set(0, 0.02, 0);
    torso.add(new Node('torsoBox'))
      .set(0, 0.24, 0).sized(0.40, 0.50, 0.23);
    torso.children[0].mesh = boxM;
    torso.children[0].material = matShirt;

    /* high-visibility vest, slightly proud of the shirt */
    var vest = torso.add(new Node('vest')).set(0, 0.25, 0).sized(0.425, 0.40, 0.255);
    vest.mesh = boxM; vest.material = matVest;
    /* two retro-reflective bands — they catch the rake light later */
    var band1 = torso.add(new Node('band1')).set(0, 0.34, 0).sized(0.435, 0.045, 0.265);
    band1.mesh = boxM;
    band1.material = { color: hex('#E9EEF2'), roughness: 0.18, specular: 0.95 };
    var band2 = torso.add(new Node('band2')).set(0, 0.17, 0).sized(0.435, 0.045, 0.265);
    band2.mesh = boxM; band2.material = band1.material;

    /* --- neck + head --- */
    var neck = torso.add(new Node('neck')).set(0, 0.50, 0);
    var head = neck.add(new Node('head')).set(0, 0.10, 0).sized(0.20, 0.235, 0.205);
    head.mesh = sphM; head.material = matSkin;
    var hat = neck.add(new Node('hat')).set(0, 0.175, 0).sized(0.245, 0.20, 0.245);
    hat.mesh = sphM; hat.material = matHat;
    var brim = neck.add(new Node('brim')).set(0, 0.145, 0.03).sized(0.28, 0.022, 0.30);
    brim.mesh = cylM; brim.material = matHat;

    /* --- arms --- */
    function arm(side) {
      var shoulder = torso.add(new Node('shoulder' + side)).set(0.245 * side, 0.44, 0);
      var upper = shoulder.add(new Node('upper' + side)).set(0, -0.135, 0)
        .sized(0.088, 0.29, 0.088);
      upper.mesh = cylM; upper.material = matVest;
      var elbow = shoulder.add(new Node('elbow' + side)).set(0, -0.28, 0);
      var fore = elbow.add(new Node('fore' + side)).set(0, -0.125, 0)
        .sized(0.076, 0.265, 0.076);
      fore.mesh = cylM; fore.material = matShirt;
      var hand = elbow.add(new Node('hand' + side)).set(0, -0.275, 0)
        .sized(0.085, 0.095, 0.075);
      hand.mesh = sphM; hand.material = matSkin;
      return { shoulder: shoulder, elbow: elbow, hand: hand };
    }
    var armL = arm(-1), armR = arm(1);

    /* --- legs --- */
    function leg(side) {
      var hip = hips.add(new Node('hip' + side)).set(0.108 * side, -0.02, 0);
      var thigh = hip.add(new Node('thigh' + side)).set(0, -0.205, 0)
        .sized(0.125, 0.42, 0.125);
      thigh.mesh = cylM; thigh.material = matLeg;
      var knee = hip.add(new Node('knee' + side)).set(0, -0.42, 0);
      var shin = knee.add(new Node('shin' + side)).set(0, -0.195, 0)
        .sized(0.105, 0.40, 0.105);
      shin.mesh = cylM; shin.material = matLeg;
      var boot = knee.add(new Node('boot' + side)).set(0, -0.415, 0.035)
        .sized(0.135, 0.10, 0.245);
      boot.mesh = boxM; boot.material = matBoot;
      return { hip: hip, knee: knee };
    }
    var legL = leg(-1), legR = leg(1);

    return {
      root: root, hips: hips, torso: torso, neck: neck,
      armL: armL, armR: armR, legL: legL, legR: legR,
      phase: opts.phase || 0
    };
  }

  /* Drive one rig. `speed` 0 = standing, 1 = walking pace. */
  function poseEngineer(rig, t, speed, extras) {
    extras = extras || {};
    var p = t * 5.4 + rig.phase;
    var s = clamp01(speed);
    var sw = Math.sin(p), cw = Math.cos(p);

    rig.hips.position[1] = 0.92 + Math.sin(p * 2) * 0.020 * s;
    rig.hips.rotation[2] = Math.sin(p) * 0.035 * s;
    rig.torso.rotation[1] = -sw * 0.085 * s;
    rig.torso.rotation[0] = 0.045 + Math.sin(p * 2) * 0.02 * s + (extras.lean || 0);
    rig.neck.rotation[0] = (extras.headPitch || 0);
    rig.neck.rotation[1] = (extras.headYaw || 0);

    /* legs — contralateral swing with a knee that only bends backward */
    rig.legL.hip.rotation[0] = sw * 0.62 * s;
    rig.legR.hip.rotation[0] = -sw * 0.62 * s;
    rig.legL.knee.rotation[0] = Math.max(0, -Math.sin(p - 0.55)) * 0.95 * s;
    rig.legR.knee.rotation[0] = Math.max(0, -Math.sin(p - 0.55 + Math.PI)) * 0.95 * s;

    /* arms — opposite to the legs, damped when a hand is holding something */
    var armGain = extras.armGain === undefined ? 1 : extras.armGain;
    rig.armL.shoulder.rotation[0] = (-sw * 0.52 * s) * armGain;
    rig.armR.shoulder.rotation[0] = (sw * 0.52 * s) * armGain;
    rig.armL.shoulder.rotation[2] = 0.10 + cw * 0.03 * s;
    rig.armR.shoulder.rotation[2] = -0.10 - cw * 0.03 * s;
    rig.armL.elbow.rotation[0] = -(0.18 + Math.max(0, -sw) * 0.45 * s) * armGain;
    rig.armR.elbow.rotation[0] = -(0.18 + Math.max(0, sw) * 0.45 * s) * armGain;

    /* pose overrides — used when the rake lamp is being held out */
    if (extras.rightArm) {
      rig.armR.shoulder.rotation[0] = extras.rightArm[0];
      rig.armR.shoulder.rotation[2] = extras.rightArm[1];
      rig.armR.elbow.rotation[0] = extras.rightArm[2];
    }
    if (extras.leftArm) {
      rig.armL.shoulder.rotation[0] = extras.leftArm[0];
      rig.armL.shoulder.rotation[2] = extras.leftArm[1];
      rig.armL.elbow.rotation[0] = extras.leftArm[2];
    }
  }

  /* ======================================================================
     THE SET
     ====================================================================== */

  function buildSet(R) {
    var boxM = R.mesh('box', function () { return Geo.box(1, 1, 1); });
    var cylM = R.mesh('cyl', function () { return Geo.cylinder(0.5, 0.5, 1, 14, true); });
    var floorM = R.mesh('floor', function () { return Geo.plane(70, 70, 1, 1, 18); });
    var quadM = R.mesh('quad', function () { return Geo.quad(1, 1); });
    var shadeM = R.mesh('shade', function () {
      return Geo.cylinder(0.07, 0.34, 0.30, 20, true);
    });
    var beamM = R.mesh('beam', function () {
      return Geo.cylinder(0.05, 1.0, 1.0, 22, false);
    });
    var coneM = R.mesh('rakeCone', function () {
      return Geo.cylinder(0.03, 0.55, 1.0, 20, false);
    });

    var set = { nodes: [], meshes: {
      box: boxM, cyl: cylM, floor: floorM, quad: quadM,
      shade: shadeM, beam: beamM, cone: coneM
    } };

    var root = new Node('set');

    /* ---- floor ---- */
    var floor = root.add(new Node('floor')).set(0, 0, 0);
    floor.mesh = floorM;
    floor.material = {
      mode: 3, color: C.concrete, roughness: 0.82, specular: 0.30,
      uvScale: [1, 1], gridFade: 1
    };

    /* ---- shell: back wall, side walls, ceiling ---- */
    function wall(name, x, y, z, sx, sy, sz, col, rough) {
      var n = root.add(new Node(name)).set(x, y, z).sized(sx, sy, sz);
      n.mesh = boxM;
      n.material = { color: col, roughness: rough || 0.88, specular: 0.14 };
      return n;
    }
    wall('backWall', 0, 3.6, -16.0, 26, 7.2, 0.4, C.wallPaint);
    wall('backDado', 0, 0.85, -15.76, 26, 1.7, 0.06, C.wallLower);
    wall('leftWall', -11.5, 3.6, -3.5, 0.4, 7.2, 26, C.wallPaint);
    wall('rightWall', 11.5, 3.6, -3.5, 0.4, 7.2, 26, C.wallPaint);
    wall('ceiling', 0, 7.1, -3.5, 24, 0.3, 26, hex('#171C21'), 0.95);

    /* ---- the shutter bay at z = +9 ---- */
    /* the opening is x ∈ [-2.5, 2.5]; the jambs must meet the side walls at
       ±11.3 exactly or you see straight through the corner */
    wall('frontWallL', -6.9, 3.6, 9.2, 8.8, 7.2, 0.5, C.wallPaint);
    wall('frontWallR', 6.9, 3.6, 9.2, 8.8, 7.2, 0.5, C.wallPaint);
    wall('frontWallTop', 0, 6.1, 9.2, 5.0, 2.2, 0.5, C.wallPaint);

    /* daylight beyond the door — an emissive card + a soft spill volume */
    var sky = root.add(new Node('sky')).set(0, 2.4, 9.62).sized(5.0, 5.0, 1);
    sky.mesh = quadM;
    sky.material = {
      color: C.daylight, roughness: 1, specular: 0,
      emissive: [C.daylight[0] * 1.25, C.daylight[1] * 1.30, C.daylight[2] * 1.38],
      cull: false
    };

    /* the shutter itself: scales down in Y as it rolls up */
    var shutter = root.add(new Node('shutter')).set(0, 2.5, 9.34).sized(5.0, 5.0, 0.12);
    shutter.mesh = boxM;
    shutter.material = { color: C.shutter, roughness: 0.55, specular: 0.42 };

    /* ---- roof trusses, for scale and for the light to graze ---- */
    for (var tI = 0; tI < 7; tI++) {
      var tz = 6.0 - tI * 3.0;
      var tr = root.add(new Node('truss' + tI)).set(0, 6.35, tz).sized(22, 0.14, 0.16);
      tr.mesh = boxM;
      tr.material = { color: C.steelDark, roughness: 0.5, specular: 0.45 };
      var brace = root.add(new Node('brace' + tI)).set(0, 6.05, tz).sized(0.12, 0.5, 0.12);
      brace.mesh = boxM; brace.material = tr.material;
    }

    /* ---- the inspection bench ---- */
    var benchTop = root.add(new Node('benchTop')).set(0, 0.70, -1.0)
      .sized(3.40, 0.075, 1.30);
    benchTop.mesh = boxM;
    benchTop.material = { color: C.benchTop, roughness: 0.34, specular: 0.72 };

    var apron = root.add(new Node('apron')).set(0, 0.615, -1.55).sized(3.40, 0.13, 0.06);
    apron.mesh = boxM;
    apron.material = { color: hex('#2B333A'), roughness: 0.5, specular: 0.4 };

    for (var lx = -1; lx <= 1; lx += 2) {
      for (var lz = -1; lz <= 1; lz += 2) {
        var lg = root.add(new Node('leg'))
          .set(lx * 1.55, 0.335, -1.0 + lz * 0.52).sized(0.075, 0.67, 0.075);
        lg.mesh = boxM;
        lg.material = { color: hex('#252C33'), roughness: 0.6, specular: 0.3 };
      }
    }
    /* lower shelf with stacked trays */
    var shelf = root.add(new Node('shelf')).set(0, 0.20, -1.0).sized(3.10, 0.05, 1.05);
    shelf.mesh = boxM;
    shelf.material = { color: hex('#232A31'), roughness: 0.7, specular: 0.2 };

    /* ---- the lot: coupons laid out on the bench ---- */
    var coupons = [];
    var layout = [
      /* x,      z,     size, crack, corrosion, scratch, rotY */
      [-1.16, -0.72, 0.30, 0.15, 0.10, 0.55, 0.05],
      [-0.74, -0.70, 0.30, 0.05, 0.05, 0.12, -0.03],
      [-1.14, -1.24, 0.30, 0.35, 0.62, 0.10, 0.09],
      [-0.72, -1.26, 0.30, 0.08, 0.06, 0.08, -0.06],
      [0.72, -0.74, 0.30, 0.55, 0.15, 0.30, -0.04],
      [1.16, -0.72, 0.30, 0.10, 0.42, 0.10, 0.07],
      [0.74, -1.26, 0.30, 0.06, 0.05, 0.06, 0.02],
      [1.18, -1.24, 0.30, 0.28, 0.20, 0.44, -0.08]
    ];
    for (var ci = 0; ci < layout.length; ci++) {
      var L = layout[ci];
      var cp = root.add(new Node('coupon' + ci))
        .set(L[0], 0.7565, L[1]).rotate(0, L[6], 0).sized(L[2], 0.018, L[2]);
      cp.mesh = boxM;
      cp.material = {
        mode: 1, color: C.ceramic, roughness: 0.42, specular: 0.55,
        uvScale: [1.5, 1.5], defect: [L[3], L[4], L[5], 0],
        scratchAngle: 0.3 + ci * 0.4,
        tangent: [1, 0, 0], bitangent: [0, 0, -1], bump: 0.55
      };
      coupons.push(cp);
    }

    /* the hero coupon — larger, front and centre, the one that gets raked */
    var hero = root.add(new Node('hero'))
      .set(0.0, 0.7585, -0.62).rotate(0, 0.04, 0).sized(0.46, 0.022, 0.46);
    hero.mesh = boxM;
    hero.material = {
      mode: 1, color: C.ceramic, roughness: 0.38, specular: 0.62,
      uvScale: [1.35, 1.35], defect: [1.0, 0.62, 0.40, 0],
      scratchAngle: 0.62, tangent: [1, 0, 0], bitangent: [0, 0, -1], bump: 0.85
    };

    /* a machinist's rule beside it, for scale */
    var rule = root.add(new Node('rule')).set(-0.42, 0.7585, -0.62)
      .rotate(0, 0.02, 0).sized(0.055, 0.006, 0.50);
    rule.mesh = boxM;
    rule.material = { color: hex('#8E9AA3'), roughness: 0.20, specular: 0.95 };

    /* ---- overhead sodium work-lamp on a drop rod ---- */
    var lampRod = root.add(new Node('lampRod')).set(0, 3.15, -1.0).sized(0.035, 1.6, 0.035);
    lampRod.mesh = cylM;
    lampRod.material = { color: hex('#1E252B'), roughness: 0.6, specular: 0.3 };

    var lampShade = root.add(new Node('lampShade')).set(0, 2.32, -1.0);
    lampShade.mesh = shadeM;
    lampShade.material = { color: C.lampBody, roughness: 0.38, specular: 0.6 };

    var bulb = root.add(new Node('bulb')).set(0, 2.20, -1.0).sized(0.16, 0.09, 0.16);
    bulb.mesh = R.mesh('sph', function () { return Geo.sphere(0.5, 16, 12); });
    bulb.material = { color: C.sodium, roughness: 1, specular: 0, emissive: [0, 0, 0] };

    /* ---- the handheld raking lamp ----------------------------------------
       Kept as a free node in world space rather than parented to the hand.
       The whole point of the shot is that the beam meets the coupon at a
       measured 6°, and only a rig driven directly by that geometry can
       guarantee it. During the walk it is snapped onto the hand instead.
       Convention: the rig's local -Y is the beam axis, so aimRot() below
       can point it with two angles.                                        */
    var lampRig = new Node('lampRig');

    var lampHead = lampRig.add(new Node('lampHead')).set(0, 0.012, 0)
      .sized(0.085, 0.055, 0.085);
    lampHead.mesh = R.mesh('sph', function () { return Geo.sphere(0.5, 14, 10); });
    lampHead.material = {
      color: C.penetrant, roughness: 1, specular: 0, emissive: [0, 0, 0]
    };

    var lampBezel = lampRig.add(new Node('lampBezel')).set(0, 0.055, 0)
      .sized(0.10, 0.075, 0.10);
    lampBezel.mesh = R.mesh('bezel', function () {
      return Geo.cylinder(0.5, 0.40, 1, 16, true);
    });
    lampBezel.material = { color: hex('#31383F'), roughness: 0.34, specular: 0.72 };

    var lampGrip = lampRig.add(new Node('lampGrip')).set(0, 0.205, 0.012)
      .rotate(-0.22, 0, 0).sized(0.055, 0.235, 0.055);
    lampGrip.mesh = cylM;
    lampGrip.material = { color: hex('#232A31'), roughness: 0.62, specular: 0.30 };

    var lampCuff = lampRig.add(new Node('lampCuff')).set(0, 0.135, 0.006)
      .sized(0.062, 0.030, 0.062);
    lampCuff.mesh = cylM;
    lampCuff.material = { color: C.penetrant, roughness: 0.4, specular: 0.5 };

    /* ---- a wall-mounted terminal, quietly hinting at the software ---- */
    var termFrame = root.add(new Node('termFrame')).set(-3.25, 1.62, -1.9)
      .rotate(0, 0.42, 0).sized(0.86, 0.56, 0.05);
    termFrame.mesh = boxM;
    termFrame.material = { color: hex('#1B2127'), roughness: 0.5, specular: 0.4 };
    var termGlass = root.add(new Node('termGlass')).set(-3.235, 1.62, -1.87)
      .rotate(0, 0.42, 0).sized(0.78, 0.48, 0.01);
    termGlass.mesh = boxM;
    termGlass.material = {
      color: hex('#0B1218'), roughness: 0.1, specular: 0.9,
      emissive: [0.02, 0.10, 0.13]
    };

    return {
      root: root, meshes: set.meshes, coupons: coupons, hero: hero,
      shutter: shutter, sky: sky, bulb: bulb, lampShade: lampShade,
      lampRig: lampRig, lampHead: lampHead, lampCuff: lampCuff,
      termGlass: termGlass
    };
  }

  /* Euler angles that point a node's local -Y down `dir`, under the engine's
     Ry·Rx·Rz composition order. Derivation: local -Y maps to
     (-sy·sx, -cx, -cy·sx), so rx = atan2(-|d.xz|, -d.y) and ry = atan2(d.x, d.z). */
  function aimRot(dir) {
    var h = Math.sqrt(dir[0] * dir[0] + dir[2] * dir[2]);
    return [Math.atan2(-h, -dir[1]), Math.atan2(dir[0], dir[2]), 0];
  }

  /* ======================================================================
     THE CINEMATIC
     ====================================================================== */

  function Intro(cfg) {
    this.cfg = cfg;
    this.canvas = cfg.canvas;
    this.onDone = cfg.onDone || function () {};
    this.finished = false;
    this.t = 0;
    this._raf = 0;
    this._last = 0;
    this._captionIdx = -1;
    this._reticles = [];

    this.R = new GL.Renderer(this.canvas, { maxDpr: 1.6, post: true });
    if (!this.R.gl) { this.finish(true); return; }

    var R = this.R;
    R.fog.color = new Float32Array([0.021, 0.026, 0.031]);
    R.fog.density = 0.030;
    R.post.exposure = 1.10;
    R.post.bloom = 0.78;
    R.post.bloomThreshold = 0.55;
    R.post.vignette = 0.72;
    R.post.chroma = 0.22;
    R.post.grain = 0.055;

    this.set = buildSet(R);

    this.engA = buildEngineer(R, {
      vest: C.hiVis, shirt: hex('#39434D'), hat: C.hardHat, phase: 0
    });
    this.engB = buildEngineer(R, {
      vest: hex('#E09A2E'), shirt: hex('#2C353E'), hat: C.hardHat2, phase: 2.1
    });
    /* the rake lamp lives in engineer A's right hand */
    this.engA.armR.hand.add(this.set.lampRig).set(0, -0.10, 0.0)
      .rotate(1.15, 0, 0);

    this.dust = R.createDust(560, [17, 6.2, 22], [0, 0.1, -3.0]);

    this._bindEvents();
    this._loop = this._loop.bind(this);
    this._last = (global.performance && performance.now ? performance.now() : Date.now());
    this._raf = requestAnimationFrame(this._loop);
  }

  Intro.prototype._bindEvents = function () {
    var self = this;
    this._onResize = function () { self.R.resize(); };
    global.addEventListener('resize', this._onResize, { passive: true });

    this._onKey = function (e) {
      if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        self.finish(false);
      }
    };
    global.addEventListener('keydown', this._onKey);

    if (this.cfg.skipButton) {
      this._onSkip = function () { self.finish(false); };
      this.cfg.skipButton.addEventListener('click', this._onSkip);
    }
  };

  Intro.prototype._teardown = function () {
    if (this._raf) cancelAnimationFrame(this._raf);
    global.removeEventListener('resize', this._onResize);
    global.removeEventListener('keydown', this._onKey);
    if (this.cfg.skipButton && this._onSkip) {
      this.cfg.skipButton.removeEventListener('click', this._onSkip);
    }
    this._clearReticles();
  };

  Intro.prototype.finish = function (immediate) {
    if (this.finished) return;
    this.finished = true;
    this._teardown();
    this.onDone(!!immediate);
  };

  /* ---------------- captions ---------------- */
  Intro.prototype._caption = function (t) {
    var idx = -1;
    for (var i = 0; i < CAPTIONS.length; i++) {
      if (t >= CAPTIONS[i].t0 && t < CAPTIONS[i].t1) { idx = i; break; }
    }
    if (idx === this._captionIdx) return;
    this._captionIdx = idx;
    var slate = this.cfg.slate;
    if (!slate) return;
    if (idx < 0) {
      slate.classList.remove('is-on');
      return;
    }
    var c = CAPTIONS[idx];
    if (this.cfg.slateCode) this.cfg.slateCode.textContent = c.code;
    if (this.cfg.slateLine) this.cfg.slateLine.textContent = c.line;
    slate.classList.remove('is-on');
    /* force a reflow so the entrance animation replays on every caption */
    void slate.offsetWidth;
    slate.classList.add('is-on');
  };

  /* ---------------- reticles ---------------- */
  Intro.prototype._clearReticles = function () {
    if (!this.cfg.reticleLayer) return;
    this.cfg.reticleLayer.innerHTML = '';
    this._reticles = [];
  };

  Intro.prototype._ensureReticles = function () {
    if (this._reticles.length || !this.cfg.reticleLayer) return;
    var defs = [
      { world: [-0.075, 0.772, -0.700], label: 'CRACK',     value: 'branching · 0.42 mm', cls: 'is-crack' },
      { world: [0.112, 0.772, -0.545],  label: 'OXIDE',     value: 'Fe₂O₃ bloom · 9 %',   cls: 'is-oxide' },
      { world: [-0.140, 0.772, -0.470], label: 'SCORING',   value: 'directional · 3 µm',  cls: 'is-score' }
    ];
    for (var i = 0; i < defs.length; i++) {
      var el = document.createElement('div');
      el.className = 'reticle ' + defs[i].cls;
      el.innerHTML =
        '<svg class="reticle-ring" viewBox="0 0 64 64" aria-hidden="true">' +
        '<circle cx="32" cy="32" r="22"></circle>' +
        '<path d="M32 2v10M32 52v10M2 32h10M52 32h10"></path>' +
        '</svg>' +
        '<div class="reticle-tag"><b>' + defs[i].label + '</b><span>' +
        defs[i].value + '</span></div>';
      this.cfg.reticleLayer.appendChild(el);
      this._reticles.push({ el: el, world: defs[i].world, t0: 0 });
    }
    this._reticles[0].t0 = T.retic1;
    this._reticles[1].t0 = T.retic2;
    this._reticles[2].t0 = T.retic2 + 0.55;
  };

  Intro.prototype._updateReticles = function (t) {
    if (t < T.retic1 - 0.4 || t > T.cutD) {
      for (var k = 0; k < this._reticles.length; k++) {
        this._reticles[k].el.style.opacity = '0';
      }
      return;
    }
    this._ensureReticles();
    for (var i = 0; i < this._reticles.length; i++) {
      var r = this._reticles[i];
      var on = span(t, r.t0, r.t0 + 0.42, Ease.outCubic);
      var off = 1 - span(t, T.cutD - 0.45, T.cutD, Ease.inQuad);
      var p = this.R.project(r.world);
      if (!p || on <= 0) { r.el.style.opacity = '0'; continue; }
      r.el.style.opacity = String(on * off);
      r.el.style.transform =
        'translate(' + p.x.toFixed(1) + 'px,' + p.y.toFixed(1) + 'px) ' +
        'translate(-50%,-50%) scale(' + (0.72 + on * 0.28).toFixed(3) + ')';
    }
  };

  /* ---------------- per-frame state ---------------- */
  Intro.prototype._apply = function (t) {
    var R = this.R, set = this.set;
    R.time = t;

    /* ---------- camera ---------- */
    var cam = sampleCam(t);
    var wob = handheld(t, t < T.cutC ? 0.020 : 0.006);
    R.setCamera(
      Vec3.create(cam.eye[0] + wob[0], cam.eye[1] + wob[1], cam.eye[2] + wob[2]),
      Vec3.create(cam.at[0] + wob[2] * 0.3, cam.at[1] + wob[1] * 0.3, cam.at[2]),
      Vec3.create(0, 1, 0), cam.fov, 0.05, 120
    );

    /* ---------- the shutter ---------- */
    var open = span(t, T.shutter0, T.shutter1, Ease.inOutQuart);
    var shutterH = lerp(5.0, 0.38, open);
    set.shutter.scale[1] = shutterH;
    set.shutter.position[1] = 5.0 - shutterH / 2;
    /* daylight strength ramps with the opening */
    var day = 0.25 + open * 1.35;
    var fadeOutDay = 1 - span(t, T.cutC - 0.4, T.cutC + 0.3, Ease.inOutQuad) * 0.85;
    set.sky.material.emissive = [
      C.daylight[0] * day * 1.3 * fadeOutDay,
      C.daylight[1] * day * 1.35 * fadeOutDay,
      C.daylight[2] * day * 1.45 * fadeOutDay
    ];

    /* ---------- the walk ---------- */
    var walk = span(t, T.walkIn, T.arriveAt, Ease.inOutCubic);
    var settle = span(t, T.arriveAt, T.arriveAt + 0.7, Ease.outCubic);
    var speed = (t > T.walkIn && t < T.arriveAt) ? 1 : (1 - settle);

    var az = lerp(9.30, 0.34, walk);
    var bz = lerp(10.10, 0.42, walk);
    this.engA.root.set(0.30, 0, az).rotate(0, Math.PI, 0);
    this.engB.root.set(-0.72, 0, bz).rotate(0, Math.PI - 0.12, 0);

    /* Engineer A raises the rake lamp over the coupon in shot C. */
    var lift = span(t, T.cutC - 0.5, T.rake0, Ease.outCubic);
    var extrasA = {
      armGain: 1 - lift * 0.9,
      headPitch: lerp(0, 0.42, lift),
      lean: lerp(0, 0.30, lift)
    };
    if (lift > 0.001) {
      extrasA.rightArm = [
        lerp(0, -1.02, lift),      /* shoulder pitch: reach forward + down */
        lerp(-0.10, -0.34, lift),  /* shoulder roll: bring it inboard      */
        lerp(-0.18, -0.62, lift)   /* elbow                                */
      ];
    }
    poseEngineer(this.engA, t, speed, extrasA);
    poseEngineer(this.engB, t, speed, {
      headPitch: lerp(0, 0.34, lift), lean: lerp(0, 0.22, lift)
    });

    this.engA.root.updateWorld(null);
    this.engB.root.updateWorld(null);

    /* ---------- the overhead sodium lamp ---------- */
    var lampWarm = span(t, T.lampOn, T.lampOn + 0.9, Ease.outQuad);
    /* fluorescent-style strike: two stutters before it holds */
    var strike = 1;
    if (t > T.lampOn && t < T.lampOn + 0.55) {
      var f = (t - T.lampOn);
      strike = (f < 0.06 || (f > 0.14 && f < 0.20) || (f > 0.30 && f < 0.34))
        ? 0.15 : 1.0;
    }
    /* it dims right down once the rake lamp takes over — that is the point */
    var dim = 1 - span(t, T.cutC, T.rake0, Ease.inOutQuad) * 0.86;
    var l0Power = lampWarm * strike * dim;
    R.lights.l0.pos = Vec3.create(0, 2.18, -1.0);
    R.lights.l0.dir = Vec3.create(0, -1, 0);
    R.lights.l0.color = new Float32Array([
      C.sodium[0] * 2.5, C.sodium[1] * 2.0, C.sodium[2] * 1.25
    ]);
    R.lights.l0.cone[2] = 3.6 * l0Power;
    set.bulb.material.emissive = [
      C.sodium[0] * 2.2 * l0Power, C.sodium[1] * 1.7 * l0Power, C.sodium[2] * 0.9 * l0Power
    ];

    /* ---------- the raking lamp: the signature move ---------- */
    var rakeUp = span(t, T.cutC - 0.35, T.rake0, Ease.outCubic);
    var sweep = span(t, T.rake0, T.rake1, Ease.inOutSine);
    var rakeOut = 1 - span(t, T.cutD + 0.5, T.outro, Ease.inQuad);

    /* The lamp orbits the coupon at a height that keeps the incidence angle
       at roughly 6°, which is where surface topography reads best. */
    var sweepAngle = lerp(-2.35, -0.62, sweep);
    var radius = 0.52;
    var lampY = 0.7585 + Math.tan(6.0 * Math.PI / 180) * radius;   /* ≈ 6° */
    var lampX = 0.0 + Math.cos(sweepAngle) * radius;
    var lampZ = -0.62 + Math.sin(sweepAngle) * radius;
    var target = Vec3.create(0.0, 0.7585, -0.62);
    var lampPos = Vec3.create(lampX, lampY + 0.028, lampZ);
    var lampDir = Vec3.normalize(Vec3.create(), Vec3.sub(Vec3.create(), target, lampPos));

    R.lights.l1.pos = lampPos;
    R.lights.l1.dir = lampDir;
    R.lights.l1.color = new Float32Array([1.55, 1.72, 1.15]);
    R.lights.l1.cone[0] = 0.985;
    R.lights.l1.cone[1] = 0.845;
    R.lights.l1.cone[2] = 5.2 * rakeUp * rakeOut;
    R.lights.l1.cone[3] = 0.22;
    set.lampHead.material.emissive = [
      C.penetrant[0] * 1.5 * rakeUp * rakeOut,
      C.penetrant[1] * 1.6 * rakeUp * rakeOut,
      C.penetrant[2] * 0.9 * rakeUp * rakeOut
    ];
    this._lampPos = lampPos;
    this._lampDir = lampDir;
    this._rakeUp = rakeUp * rakeOut;
    this._l0Power = l0Power;

    /* ---------- the reveal ---------- */
    /* Defects only surface where the beam has already passed. Reveal trails
       the sweep slightly, so you watch the crack network open up. */
    var reveal = clamp01(span(t, T.rake0 - 0.25, T.rake1 - 0.55, Ease.outCubic));
    set.hero.material.defect = [1.0, 0.62, 0.40, reveal];
    for (var ci = 0; ci < set.coupons.length; ci++) {
      var d = set.coupons[ci].material.defect;
      d[3] = reveal * 0.65;
    }

    /* terminal glow comes up as the story turns toward the software */
    var termOn = span(t, T.cutD, T.title, Ease.outQuad);
    set.termGlass.material.emissive = [
      0.02 + 0.05 * termOn, 0.10 + 0.30 * termOn, 0.13 + 0.38 * termOn
    ];

    /* ---------- atmosphere per shot ---------- */
    R.lights.ambSky = new Float32Array([
      0.030 + open * 0.030, 0.038 + open * 0.036, 0.048 + open * 0.044
    ]);
    R.lights.ambGround = new Float32Array([0.010, 0.012, 0.015]);
    R.fog.density = lerp(0.030, 0.012, span(t, T.cutB, T.cutC, Ease.inOutQuad));

    /* ---------- grade ---------- */
    var fadeIn = 1 - span(t, 0, T.aIn, Ease.outQuad);
    var fadeOut = span(t, T.outro, T.total, Ease.inQuad);
    R.post.flash = Math.max(fadeIn, Math.max(cutDip(t) * 0.9, fadeOut));
    R.post.flashColor = new Float32Array([0.008, 0.010, 0.012]);
    R.post.vignette = lerp(0.78, 0.58, span(t, T.cutC, T.cutD, Ease.inOutQuad));
    R.post.grain = lerp(0.062, 0.030, span(t, T.cutC, T.total, Ease.linear));
  };

  /* ---------------- draw ---------------- */
  Intro.prototype._draw = function (t) {
    var R = this.R, set = this.set, M = set.meshes;
    var tmp = Mat4.create();

    R.beginScene();

    set.root.updateWorld(null);
    set.root.render(R);
    this.engA.root.render(R);
    this.engB.root.render(R);

    /* ---- volumetrics, drawn last so they read as haze over the set ---- */

    /* daylight wedge coming through the open shutter */
    var open = span(t, T.shutter0, T.shutter1, Ease.inOutQuart);
    if (open > 0.02 && t < T.cutC) {
      var dayA = open * 0.13 * (1 - span(t, T.cutB + 0.6, T.cutC, Ease.inQuad));
      Mat4.compose(tmp, [0, 2.2, 7.2], [Math.PI / 2, 0, 0], [3.4, 4.6, 3.4]);
      R.drawVolume(M.beam, tmp, C.daylight, dayA, 0);
    }

    /* the overhead sodium cone + the pool it throws on the bench */
    if (this._l0Power > 0.02) {
      Mat4.compose(tmp, [0, 1.44, -1.0], [0, 0, 0], [1.55, 1.55, 1.55]);
      R.drawVolume(M.beam, tmp, C.sodium, 0.115 * this._l0Power, 0);
      Mat4.compose(tmp, [0, 0.7625, -1.0], [-Math.PI / 2, 0, 0], [2.5, 2.5, 1]);
      R.drawVolume(M.quad, tmp, C.sodium, 0.16 * this._l0Power, 1);
    }

    /* the rake beam: a narrow cone lying almost flat across the coupon */
    if (this._rakeUp > 0.02) {
      var lp = this._lampPos, ld = this._lampDir;
      /* orient a cone from the lamp toward the coupon */
      var len = 0.72;
      var mid = [lp[0] + ld[0] * len * 0.5, lp[1] + ld[1] * len * 0.5, lp[2] + ld[2] * len * 0.5];
      var yaw = Math.atan2(ld[0], ld[2]);
      var pitch = Math.asin(clamp(-ld[1], -1, 1));
      /* the cone mesh points +Y, so pitch it to -Y then aim it */
      Mat4.compose(tmp, mid, [-(Math.PI / 2 - pitch), yaw, 0], [0.30, len, 0.30]);
      R.drawVolume(M.cone, tmp, C.penetrant, 0.14 * this._rakeUp, 0);

      Mat4.compose(tmp, [0.0, 0.7635, -0.62], [-Math.PI / 2, 0, 0], [0.95, 0.95, 1]);
      R.drawVolume(M.quad, tmp, C.penetrant, 0.20 * this._rakeUp, 1);
    }

    /* airborne dust — the reason a beam is visible at all */
    R.drawDust(this.dust, C.daylight, 0.30);

    R.endScene();
  };

  Intro.prototype._loop = function (now) {
    if (this.finished) return;
    var dt = Math.min(0.05, (now - this._last) / 1000);
    this._last = now;
    this.t += dt;

    this.R.resize();
    this._apply(this.t);
    this._draw(this.t);
    this._caption(this.t);
    this._updateReticles(this.t);

    if (this.cfg.bar) {
      this.cfg.bar.style.transform =
        'scaleX(' + clamp01(this.t / T.total).toFixed(4) + ')';
    }
    if (this.cfg.title) {
      var ti = span(this.t, T.title, T.title + 1.0, Ease.outCubic);
      var to = 1 - span(this.t, T.total - 0.45, T.total, Ease.inQuad);
      this.cfg.title.style.opacity = String(ti * to);
      this.cfg.title.classList.toggle('is-on', ti > 0.02);
    }

    if (this.t >= T.total) { this.finish(false); return; }
    this._raf = requestAnimationFrame(this._loop);
  };

  /* ======================================================================
     PUBLIC ENTRY
     ====================================================================== */

  global.NFIntro = {
    duration: T.total,
    /* Returns a controller, or null if the intro cannot run (no WebGL,
       reduced-motion, already seen this session). The caller shows the app
       immediately in that case. */
    play: function (cfg) {
      if (!GL.supported) return null;
      return new Intro(cfg);
    }
  };
})(window);
