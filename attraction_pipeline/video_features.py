from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

EMOTION_KEYS = ["happy", "neutral", "surprise"]


@dataclass
class FrameSignals:
    facial_symmetry: float
    face_width_height_ratio: float
    eye_spacing_ratio: float
    jaw_ratio: float
    smile_intensity: float
    eye_contact: float
    eye_openness: float
    gaze_stability: float
    head_tilt: float
    expression_speed: float
    happy: float
    neutral: float
    surprise: float


class VideoFeatureExtractor:
    """Extract frame-level signals and aggregate them into video-level statistics."""

    def __init__(self, min_face_confidence: float = 0.5):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=min_face_confidence,
            min_tracking_confidence=min_face_confidence,
            refine_landmarks=True,
        )
        self._prev_expr: Optional[np.ndarray] = None

    @staticmethod
    def _dist(a, b) -> float:
        return float(np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y])))

    @staticmethod
    def _safe_div(x: float, y: float) -> float:
        return x / y if abs(y) > 1e-6 else 0.0

    def _extract_geometry(self, lm) -> Dict[str, float]:
        left_cheek, right_cheek = lm[234], lm[454]
        chin, forehead = lm[152], lm[10]
        nose = lm[1]
        left_eye_outer, right_eye_outer = lm[33], lm[263]
        jaw_left, jaw_right = lm[172], lm[397]

        face_w = self._dist(left_cheek, right_cheek)
        face_h = self._dist(chin, forehead)
        eye_spacing = self._dist(left_eye_outer, right_eye_outer)
        jaw_w = self._dist(jaw_left, jaw_right)

        left_nose = self._dist(left_cheek, nose)
        right_nose = self._dist(right_cheek, nose)
        symmetry = 1.0 - min(abs(left_nose - right_nose), 1.0)

        return {
            "facial_symmetry": float(np.clip(symmetry, 0.0, 1.0)),
            "face_width_height_ratio": self._safe_div(face_w, face_h),
            "eye_spacing_ratio": self._safe_div(eye_spacing, face_w),
            "jaw_ratio": self._safe_div(jaw_w, face_w),
        }

    def _extract_behavior(self, lm) -> Dict[str, float]:
        mouth_left, mouth_right = lm[61], lm[291]
        mouth_top, mouth_bottom = lm[13], lm[14]
        left_eye_top, left_eye_bottom = lm[159], lm[145]
        right_eye_top, right_eye_bottom = lm[386], lm[374]
        left_iris, right_iris = lm[468], lm[473]
        nose_tip = lm[1]

        mouth_w = self._dist(mouth_left, mouth_right)
        mouth_h = self._dist(mouth_top, mouth_bottom)
        smile = float(np.clip(self._safe_div(mouth_h, mouth_w) * 2.5, 0.0, 1.0))

        left_open = self._dist(left_eye_top, left_eye_bottom)
        right_open = self._dist(right_eye_top, right_eye_bottom)
        eye_open = float(np.clip((left_open + right_open) * 25.0, 0.0, 1.0))

        eye_mid_x = (lm[33].x + lm[263].x) / 2
        gaze_center = (left_iris.x + right_iris.x) / 2
        eye_contact = float(np.clip(1.0 - abs(gaze_center - eye_mid_x) * 6.0, 0.0, 1.0))

        head_tilt = float(np.clip(abs(lm[33].y - lm[263].y) * 5.0, 0.0, 1.0))

        expr_vector = np.array([smile, eye_open, eye_contact, nose_tip.x, nose_tip.y], dtype=np.float32)
        if self._prev_expr is None:
            expr_speed = 0.0
            gaze_stability = 1.0
        else:
            expr_speed = float(np.clip(np.linalg.norm(expr_vector - self._prev_expr), 0.0, 1.0))
            gaze_stability = float(np.clip(1.0 - abs(expr_vector[2] - self._prev_expr[2]), 0.0, 1.0))

        self._prev_expr = expr_vector

        return {
            "smile_intensity": smile,
            "eye_contact": eye_contact,
            "eye_openness": eye_open,
            "gaze_stability": gaze_stability,
            "head_tilt": head_tilt,
            "expression_speed": expr_speed,
        }

    def _extract_emotions(self, frame_bgr: np.ndarray) -> Dict[str, float]:
        try:
            result = DeepFace.analyze(
                frame_bgr,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
            )
            if isinstance(result, list):
                result = result[0]
            emo = result.get("emotion", {})
            total = sum(float(v) for v in emo.values()) or 1.0
            return {key: float(emo.get(key, 0.0)) / total for key in EMOTION_KEYS}
        except Exception:
            return {key: 0.0 for key in EMOTION_KEYS}

    def process_video(self, video_path: str, frame_step: int = 5, max_frames: Optional[int] = None) -> Dict[str, float]:
        cap = cv2.VideoCapture(video_path)
        frame_features: List[Dict[str, float]] = []
        frame_idx = 0
        self._prev_expr = None

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if frame_idx % frame_step != 0:
                continue
            if max_frames is not None and len(frame_features) >= max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._mesh.process(rgb)
            if not result.multi_face_landmarks:
                continue

            lm = result.multi_face_landmarks[0].landmark
            geom = self._extract_geometry(lm)
            beh = self._extract_behavior(lm)
            emo = self._extract_emotions(frame)
            frame_features.append({**geom, **beh, **emo})

        cap.release()

        if not frame_features:
            return {}

        stats: Dict[str, float] = {}
        keys = frame_features[0].keys()
        for key in keys:
            arr = np.array([f[key] for f in frame_features], dtype=np.float32)
            stats[f"{key}_mean"] = float(np.mean(arr))
            stats[f"{key}_std"] = float(np.std(arr))
            stats[f"{key}_median"] = float(np.median(arr))
            stats[f"{key}_max"] = float(np.max(arr))

        stats["processed_frames"] = float(len(frame_features))
        return stats
