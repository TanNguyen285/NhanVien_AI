"""
logic.py
--------
Module xử lý điểm cảm xúc + tổng hợp kết quả phiên 5 giây.
Tách biệt hoàn toàn khỏi Flask / AI model để dễ tái sử dụng.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Bảng điểm cảm xúc → mức độ hài lòng khách hàng
# Thang gốc: -5 đến +5  →  quy về 1–5 sao ở cuối
# ---------------------------------------------------------------------------
EMOTION_SCORES: dict[str, float] = {
    "Angry":    -5.0,   # rất không hài lòng
    "Disgust":  -4.0,   # không hài lòng
    "Fear":     -2.0,   # lo lắng / tiêu cực nhẹ
    "Sad":      -1.0,   # buồn / hơi tiêu cực
    "Neutral":   2.0,   # bình thường
    "Surprise":  3.0,   # ngạc nhiên (thường tích cực)
    "Happy":     5.0,   # rất hài lòng
}

# Giá trị min/max lý thuyết của thang gốc (dùng để chuẩn hoá)
_RAW_MIN = min(EMOTION_SCORES.values())   # -5
_RAW_MAX = max(EMOTION_SCORES.values())   # +5
_STAR_MIN, _STAR_MAX = 1.0, 5.0


@dataclass
class FrameResult:
    """Kết quả 1 frame: nhãn cảm xúc + confidence."""
    label: str
    confidence: float          # 0.0 – 1.0
    all_scores: dict[str, float] = field(default_factory=dict)   # {"Happy": 87.3, ...}


@dataclass
class SessionResult:
    """Kết quả tổng hợp sau phiên 5 giây."""
    star_rating: float         # 1.0 – 5.0
    average_raw: float         # điểm thô trung bình (-5 đến +5)
    average_confidence: float  # FIX: độ tin cậy trung bình (0-100%)
    dominant_emotion: str      # cảm xúc xuất hiện nhiều nhất
    emotion_counts: dict[str, int]
    total_frames: int
    summary: str               # nhãn text dễ đọc


# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------

def score_emotion(label: str) -> float:
    """Tra điểm thô của 1 nhãn cảm xúc."""
    return EMOTION_SCORES.get(label, 0.0)


def raw_to_stars(raw: float) -> float:
    """Chuyển điểm thô [-5, +5] → thang 5 sao [1, 5], làm tròn 0.5."""
    normalized = (raw - _RAW_MIN) / (_RAW_MAX - _RAW_MIN)   # 0–1
    stars = _STAR_MIN + normalized * (_STAR_MAX - _STAR_MIN)  # 1–5
    # Làm tròn đến 0.5 gần nhất
    return round(stars * 2) / 2


def _stars_label(stars: float) -> str:
    if stars >= 4.5:
        return "Rất hài lòng"
    if stars >= 3.5:
        return "Hài lòng"
    if stars >= 2.5:
        return "Bình thường"
    if stars >= 1.5:
        return "Không hài lòng"
    return "Rất không hài lòng"


# ---------------------------------------------------------------------------
# ScanSession – tích lũy frame trong 5 giây rồi xuất kết quả
# ---------------------------------------------------------------------------

class ScanSession:
    """
    Dùng:
        session = ScanSession()
        session.add_frame(FrameResult("Happy", 0.92))
        ...
        result = session.finalize()
    """

    def __init__(self):
        self._frames: list[FrameResult] = []

    def add_frame(self, frame_result: FrameResult) -> None:
        """Thêm kết quả 1 frame vào phiên (gọi mỗi frame có mặt)."""
        self._frames.append(frame_result)

    def add_frame_raw(self, label: str, confidence: float,
                      all_scores: Optional[dict] = None) -> None:
        """Shortcut không cần tạo FrameResult thủ công."""
        self.add_frame(FrameResult(label, confidence, all_scores or {}))

    def finalize(self) -> Optional[SessionResult]:
        """
        Tổng hợp toàn bộ frame → SessionResult.
        Trả None nếu không có frame nào.
        """
        if not self._frames:
            return None

        # Đếm số lần xuất hiện mỗi cảm xúc (weighted by confidence)
        counts: dict[str, int] = {}
        weighted_score_sum = 0.0
        weight_sum = 0.0
        confidence_sum = 0.0  # FIX: tính confidence trung bình

        for fr in self._frames:
            counts[fr.label] = counts.get(fr.label, 0) + 1
            w = fr.confidence
            weighted_score_sum += score_emotion(fr.label) * w
            weight_sum += w
            confidence_sum += w  # lưu ý: confidence đã là 0.0-1.0, cần convert thành %

        avg_raw = weighted_score_sum / weight_sum if weight_sum > 0 else 0.0
        avg_conf = (confidence_sum / len(self._frames) * 100) if self._frames else 0.0  # FIX: convert to %
        stars = raw_to_stars(avg_raw)
        dominant = max(counts, key=counts.get)

        return SessionResult(
            star_rating=stars,
            average_raw=round(avg_raw, 2),
            average_confidence=round(avg_conf, 1),  # FIX: add this
            dominant_emotion=dominant,
            emotion_counts=counts,
            total_frames=len(self._frames),
            summary=_stars_label(stars),
        )

    def reset(self) -> None:
        self._frames.clear()