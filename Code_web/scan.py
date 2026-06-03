"""
scan.py
-------
Quản lý trạng thái + vòng lặp phiên scan 5 giây.
"""

import time
import threading

from camera import Camera
from inference import push_frame, get_latest
from logic import ScanSession

SCAN_DURATION = 5

_lock   = threading.Lock()
_active = False
_session: ScanSession | None = None
_result:  dict | None        = None


def is_active() -> bool:
    with _lock:
        return _active


def get_result() -> dict | None:
    with _lock:
        return _result


def start(cam: Camera) -> dict:
    global _active, _session, _result

    with _lock:
        if _active:
            return {"status": "already_running", "message": "Phiên scan đang chạy, vui lòng chờ."}
        _active  = True
        _result  = None
        _session = ScanSession()

    threading.Thread(target=_scan_loop, args=(cam,), daemon=True).start()
    return {"status": "started", "duration": SCAN_DURATION}


def status() -> dict:
    with _lock:
        if _active:
            return {"status": "running"}
        if _result:
            return {"status": "done", "result": _result}
        return {"status": "idle"}


def _scan_loop(cam: Camera):
    global _active, _result

    deadline       = time.time() + SCAN_DURATION
    last_push_time = 0.0

    while time.time() < deadline:
        with _lock:
            if not _active:
                break

        ret, frame = cam.read()
        if ret and frame is not None:
            now = time.time()
            if now - last_push_time >= 0.033:   # ~30fps push
                push_frame(frame)
                last_push_time = now

        detections = get_latest()
        if detections:
            best = max(detections, key=lambda d: d["score"])
            with _lock:
                if _active and _session:
                    _session.add_frame_raw(
                        label=best["label"],
                        confidence=best["score"] / 100,
                        all_scores=best["all_scores"],
                    )

        time.sleep(0.001)

    # Finalize
    with _lock:
        _active = False
        res     = _session.finalize() if _session else None
        _result = {
            "star_rating":        res.star_rating,
            "average_raw":        res.average_raw,
            "average_confidence": res.average_confidence,
            "dominant_emotion":   res.dominant_emotion,
            "emotion_counts":     res.emotion_counts,
            "total_frames":       res.total_frames,
            "summary":            res.summary,
        } if res else {"error": "Không phát hiện khuôn mặt nào trong 5 giây."}