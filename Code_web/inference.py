"""
inference.py
------------
Inference pipeline: YOLO face detection + PAttLite emotion.
Worker thread chạy liên tục, luôn xử lý frame mới nhất.
"""

import queue
import threading
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

try:
    from Pattlite import PAttLite
except ImportError:
    print("Không tìm thấy file Pattlite.py!")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES  = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
USE_HALF = DEVICE.type == "cuda"

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
def _find_emo_model() -> str:
    path = os.path.join(BASE_DIR, 'runs', 'PAtt_Lite_final.pth')
    if os.path.exists(path):
        return path
    alt = os.path.join(BASE_DIR, 'runs', 'runs', 'PAtt_Lite_final.pth')
    if os.path.exists(alt):
        return alt
    matches = glob.glob(os.path.join(BASE_DIR, '**', 'PAtt_Lite_final.pth'), recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError("Không tìm thấy PAtt_Lite_final.pth")


face_model = YOLO(os.path.join(BASE_DIR, 'runs/detect/yolov26_e20/weights/best.pt'))

emo_net = PAttLite(num_classes=7)
emo_net.load_state_dict(torch.load(_find_emo_model(), map_location=DEVICE, weights_only=True))
emo_net.to(DEVICE).eval()
if USE_HALF:
    emo_net.half()

# Warmup
with torch.no_grad():
    _dummy = torch.zeros(1, 3, 112, 112, device=DEVICE)
    if USE_HALF:
        _dummy = _dummy.half()
    emo_net(_dummy)

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
_infer_queue:   queue.Queue = queue.Queue(maxsize=1)
_latest_result: list[dict]  = []
_result_lock    = threading.Lock()


def _run_inference(frame: np.ndarray) -> list[dict]:
    """YOLO + PAttLite trên 1 frame. Chỉ gọi từ worker thread."""
    try:
        h, w  = frame.shape[:2]
        scale = 320 / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LINEAR) if scale < 1.0 else frame
        if scale >= 1.0:
            scale = 1.0

        results = face_model(small, verbose=False, conf=0.5, imgsz=320)[0]
        output  = []

        for box in results.boxes:
            x1, y1, x2, y2 = (int(v / scale) for v in map(float, box.xyxy[0]))
            x1, y1 = max(0, x1), max(0, y1)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            tensor = data_transform(
                Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            ).unsqueeze(0).to(DEVICE)
            if USE_HALF:
                tensor = tensor.half()

            with torch.no_grad():
                probs = torch.nn.functional.softmax(emo_net(tensor), dim=1)
                probs = probs.float().cpu().numpy()[0]
            idx = int(np.argmax(probs))

            output.append({
                "bbox":       [x1, y1, x2, y2],
                "label":      CLASSES[idx],
                "score":      float(probs[idx] * 100),
                "all_scores": {c: float(probs[i] * 100) for i, c in enumerate(CLASSES)},
            })
        return output
    except Exception:
        return []


def _worker():
    while True:
        try:
            frame = _infer_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if frame is None:
            break
        detections = _run_inference(frame)
        with _result_lock:
            _latest_result.clear()
            _latest_result.extend(detections)


# Public API
def push_frame(frame: np.ndarray):
    """Đẩy frame mới vào pipeline, bỏ frame cũ nếu queue đầy."""
    try:
        _infer_queue.put_nowait(frame)
    except queue.Full:
        try:
            _infer_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _infer_queue.put_nowait(frame)
        except queue.Full:
            pass


def get_latest() -> list[dict]:
    """Lấy kết quả inference mới nhất (non-blocking)."""
    with _result_lock:
        return list(_latest_result)


def stop():
    """Dừng worker thread."""
    _infer_queue.put(None)


# Khởi động worker
threading.Thread(target=_worker, daemon=True).start()