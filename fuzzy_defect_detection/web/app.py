"""
================================================================================
 app.py  —  PRODUCTION FLASK APPLICATION
================================================================================

Serves the modern light-theme dashboard for the hybrid ML + Fuzzy defect
detection system, and exposes the background-job API.

Endpoints
=========
 GET  /                     dashboard SPA
 GET  /healthz              health probe
 GET  /system               model / dataset / evaluation metadata
 POST /upload               accept an image → start a background job → {job_id}
 GET  /status/<id>          staged progress
 GET  /result/<id>          full fused report (when done)
 GET  /samples              bundled test-set samples (per class)
 GET  /sample_image/<name>  serve a sample thumbnail/full image
 POST /analyze_sample       analyse a bundled sample by filename

Run:  cd web && python app.py    →  http://127.0.0.1:5000
================================================================================
"""

import os
import uuid
import glob
import shutil

from flask import (Flask, request, jsonify, render_template,
                   send_file, abort)
from werkzeug.utils import secure_filename

from service import JobManager, system_info

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
UPLOAD_DIR = os.path.join(HERE, "static", "uploads")
TEST_DIR = os.path.join(ROOT, "data", "custom_samples")
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
JOBS = JobManager()


def _ok(fn):
    return os.path.splitext(fn.lower())[1] in ALLOWED


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/system")
def system():
    return jsonify(system_info())


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file part."}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400
    if not _ok(f.filename):
        return jsonify({"error": "Unsupported file type."}), 400
    name = f"{uuid.uuid4().hex[:10]}_{secure_filename(f.filename)}"
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)
    return jsonify({"job_id": JOBS.submit(path)})


@app.route("/status/<jid>")
def status(jid):
    st = JOBS.status(jid)
    return (jsonify(st) if st else (jsonify({"error": "Unknown job."}), 404))


@app.route("/result/<jid>")
def result(jid):
    st = JOBS.status(jid)
    if not st:
        return jsonify({"error": "Unknown job."}), 404
    if st["status"] != "done":
        return jsonify({"error": "Not ready.", "status": st}), 409
    return jsonify(JOBS.result(jid))


@app.route("/samples")
def samples():
    items = []
    if os.path.isdir(TEST_DIR):
        for cls in sorted(os.listdir(TEST_DIR)):
            cdir = os.path.join(TEST_DIR, cls)
            if not os.path.isdir(cdir):
                continue
            files = sorted(glob.glob(os.path.join(cdir, "*.png")))[:4]
            for fp in files:
                fn = os.path.basename(fp)
                items.append({"filename": f"{cls}/{fn}", "label": cls,
                              "band": "custom"})
    return jsonify({"samples": items})


@app.route("/sample_image/<path:name>")
def sample_image(name):
    # name is "<class>/<file>"
    parts = name.split("/")
    if len(parts) != 2:
        abort(404)
    cls, fn = secure_filename(parts[0]), secure_filename(parts[1])
    fp = os.path.join(TEST_DIR, cls, fn)
    if not os.path.isfile(fp):
        abort(404)
    return send_file(fp, mimetype="image/png")


@app.route("/analyze_sample", methods=["POST"])
def analyze_sample():
    data = request.get_json(silent=True) or {}
    name = data.get("filename", "")
    parts = name.split("/")
    if len(parts) != 2:
        return jsonify({"error": "Bad sample name."}), 400
    cls, fn = secure_filename(parts[0]), secure_filename(parts[1])
    src = os.path.join(TEST_DIR, cls, fn)
    if not os.path.isfile(src):
        return jsonify({"error": "Unknown sample."}), 404
    dst = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:10]}_{fn}")
    shutil.copyfile(src, dst)
    return jsonify({"job_id": JOBS.submit(dst)})


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File too large (max 16 MB)."}), 413


if __name__ == "__main__":
    print("=" * 60)
    print(" Hybrid ML + Fuzzy Defect Detection — Production Web UI")
    print(" http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
