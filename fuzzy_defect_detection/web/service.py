"""
================================================================================
 service.py  —  PRODUCTION BACKGROUND-JOB SERVICE FOR THE WEB APP
================================================================================

Wraps the real-time InferencePipeline in a thread-based JobManager so the web
UI can submit an image, get a job id instantly, and poll for staged progress
while the (ML + fuzzy + fusion) pipeline runs on a worker thread.

Also exposes model/dataset/evaluation metadata for the dashboard "system" panel.
================================================================================
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import threading
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
ROOT = os.path.join(HERE, "..")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from inference_pipeline import InferencePipeline   # noqa: E402


# --------------------------------------------------------------------------- #
#  Lazy pipeline singleton (loads the model once)
# --------------------------------------------------------------------------- #
_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()


def get_pipeline() -> InferencePipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = InferencePipeline()
    return _PIPELINE


# --------------------------------------------------------------------------- #
#  System metadata (model + dataset + evaluation)
# --------------------------------------------------------------------------- #
def system_info() -> dict:
    from fusion_engine import DETECT_THRESHOLD
    info = {"model": None, "dataset": None, "evaluation": None, "fusion": {"detect_threshold": DETECT_THRESHOLD}}
    tm = os.path.join(ROOT, "models", "training_metrics.json")
    if os.path.isfile(tm):
        m = json.load(open(tm))
        info["model"] = {
            "selected_model": m.get("selected_model"),
            "test_accuracy": m.get("test_accuracy"),
            "test_macro_f1": m.get("test_macro_f1"),
            "binary_defect_detection": m.get("binary_defect_detection"),
            "n_features": m.get("n_features"),
            "trained_at": m.get("trained_at"),
        }
    dm = os.path.join(ROOT, "data", "dataset_manifest.json")
    if os.path.isfile(dm):
        d = json.load(open(dm))
        info["dataset"] = {"counts": d.get("counts"), "classes": d.get("classes")}
    se = os.path.join(ROOT, "results", "system_evaluation.json")
    if os.path.isfile(se):
        s = json.load(open(se))
        info["evaluation"] = {
            "detection": s.get("detection"),
            "borderline_recall": s.get("borderline_recall"),
            "type_accuracy": s.get("type_accuracy"),
            "n_test": s.get("n_test"),
        }
    return info


# --------------------------------------------------------------------------- #
#  Job manager
# --------------------------------------------------------------------------- #
class JobManager:
    def __init__(self, max_jobs: int = 60):
        self._jobs: dict = {}
        self._lock = threading.Lock()
        self._max = max_jobs

    def submit(self, image_path: str) -> str:
        jid = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[jid] = {
                "status": "queued", "progress": 0, "message": "Queued…",
                "stage": None, "stages": [], "result": None, "error": None,
                "created": time.time(),
            }
            self._prune()
        threading.Thread(target=self._work, args=(jid, image_path),
                         daemon=True).start()
        return jid

    def _work(self, jid: str, image_path: str):
        def cb(p, msg, stage=None):
            with self._lock:
                j = self._jobs.get(jid)
                if not j:
                    return
                j["progress"] = p
                j["message"] = msg
                j["status"] = "running"
                if stage and stage not in [s["stage"] for s in j["stages"]]:
                    j["stages"].append({"stage": stage, "message": msg,
                                        "progress": p, "t": time.time()})
                j["stage"] = stage
        try:
            result = get_pipeline().run(image_path, progress_cb=cb)
            with self._lock:
                self._jobs[jid].update(status="done", progress=100,
                                       message="Done.", result=result)
        except Exception as exc:   # noqa: BLE001
            with self._lock:
                self._jobs[jid].update(status="error", message=str(exc),
                                       error=traceback.format_exc())
        finally:
            try:
                os.remove(image_path)
            except OSError:
                pass

    def status(self, jid: str):
        with self._lock:
            j = self._jobs.get(jid)
            if not j:
                return None
            return {"status": j["status"], "progress": j["progress"],
                    "message": j["message"], "stage": j["stage"],
                    "stages": j["stages"], "error": j["error"]}

    def result(self, jid: str):
        with self._lock:
            j = self._jobs.get(jid)
            return j["result"] if j else None

    def _prune(self):
        if len(self._jobs) <= self._max:
            return
        old = sorted(self._jobs.items(), key=lambda kv: kv[1]["created"])
        for k, _ in old[:len(self._jobs) - self._max]:
            self._jobs.pop(k, None)
