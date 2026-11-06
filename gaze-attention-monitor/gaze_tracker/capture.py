"""Cámara con límite de FPS y detector de landmarks (MediaPipe FaceLandmarker)."""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .geometry import LANDMARK_IDS

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_IDS = np.array(LANDMARK_IDS)


def ensure_model(path: Path) -> Path:
    """Descarga el modelo (3.7 MB) una sola vez; la inferencia sigue siendo local."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, partial)
    os.replace(partial, path)
    return path


class CameraStream:
    """Entrega frames a la tasa pedida y duerme entre lecturas en vez de girar."""

    def __init__(self, cfg: Config) -> None:
        source = cfg.camera
        if isinstance(source, int) and sys.platform == "win32":
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {source!r}.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.fps_active)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._next = 0.0

    def read(self, fps: float) -> tuple[np.ndarray, float] | None:
        """Devuelve (frame BGR, instante de captura) o None si la cámara falla."""
        wait = self._next - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        if not self._cap.grab():
            return None
        t = time.perf_counter()
        self._next = max(self._next + 1.0 / fps, t)
        ok, frame = self._cap.retrieve()
        return (frame, t) if ok else None

    def close(self) -> None:
        """Libera el dispositivo."""
        self._cap.release()


class LandmarkDetector:
    """Una cara, sin blendshapes ni matrices: solo lo que usa el sistema."""

    def __init__(self, model_path: Path) -> None:
        # Import diferido: MediaPipe tarda en cargar y así no bloquea la UI.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame_bgr: np.ndarray, t: float) -> np.ndarray | None:
        """Devuelve los puntos de `LANDMARK_IDS` en píxeles, o None sin rostro."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._last_ms = max(self._last_ms + 1, int(t * 1000.0))
        faces = self._landmarker.detect_for_video(image, self._last_ms).face_landmarks
        if not faces:
            return None
        # La API entrega objetos Python: se copian solo los 24 puntos necesarios.
        face = faces[0]
        n = _IDS.size
        flat = np.fromiter(
            (v for i in LANDMARK_IDS for v in (face[i].x, face[i].y)),
            dtype=np.float64,
            count=2 * n,
        )
        h, w = frame_bgr.shape[:2]
        return flat.reshape(n, 2) * (w, h)

    def close(self) -> None:
        """Libera el grafo de MediaPipe."""
        self._landmarker.close()
