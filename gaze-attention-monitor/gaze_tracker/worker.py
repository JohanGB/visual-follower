"""Hilos de fondo: captura e inferencia continua, y ajuste del modelo."""

from __future__ import annotations

import time
from enum import Enum

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .capture import CameraStream, LandmarkDetector, ensure_model
from .config import Config
from .display import ScreenInfo
from .model import GazeModel
from .personal import assess_model, fit_session, save_session
from .pipeline import GazePipeline


class Mode(Enum):
    """Qué hace el worker con cada frame."""

    PAUSED = 0
    CALIBRATE = 1
    TRACK = 2


class GazeWorker(QThread):
    """Lee cámara e infiere fuera del hilo de la UI.

    La UI solo recibe señales encoladas por Qt, y `alert_changed` únicamente al
    cambiar de estado. El modelo y el modo se reemplazan por asignación simple,
    atómica bajo el GIL, de modo que este bucle no necesita bloqueos.
    """

    ready = pyqtSignal()
    failed = pyqtSignal(str)
    gaze_moved = pyqtSignal(float, float, bool)
    alert_changed = pyqtSignal(bool)
    face_changed = pyqtSignal(bool)
    observed = pyqtSignal(float, object)  # t, landmarks crudos

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self._cfg = cfg
        self._mode = Mode.PAUSED
        self._pending_model: GazeModel | None = None
        self._running = True
        self.pipeline: GazePipeline | None = None
        self.frame_size = (cfg.frame_width, cfg.frame_height)

    def set_mode(self, mode: Mode) -> None:
        """Cambia de modo desde cualquier hilo."""
        self._mode = mode

    def set_model(self, model: GazeModel) -> None:
        """Entrega un modelo; se aplica entre dos frames."""
        self._pending_model = model

    def stop(self) -> None:
        """Pide salir y espera el fin del bucle."""
        self._running = False
        self.wait(3000)

    def _fps(self, face_absent_for: float) -> float:
        if self._mode is Mode.CALIBRATE:
            return self._cfg.fps_calibration
        if self._mode is Mode.PAUSED or face_absent_for >= self._cfg.idle_after_s:
            return self._cfg.fps_idle
        return self._cfg.fps_active

    def run(self) -> None:
        """Bucle principal del hilo."""
        cv2.setNumThreads(1)
        camera = detector = None
        try:
            model_path = ensure_model(self._cfg.model_path)
            detector = LandmarkDetector(model_path)
            camera = CameraStream(self._cfg)
            self._loop(camera, detector)
        except Exception as exc:  # se muestra en la UI en vez de morir en silencio
            self.failed.emit(str(exc))
        finally:
            if camera is not None:
                camera.close()
            if detector is not None:
                detector.close()

    def _loop(self, camera: CameraStream, detector: LandmarkDetector) -> None:
        first = camera.read(self._cfg.fps_active)
        if first is None:
            raise RuntimeError("La cámara no entrega imágenes.")
        height, width = first[0].shape[:2]
        self.frame_size = (width, height)
        pipeline = GazePipeline(self._cfg, width, height)
        self.pipeline = pipeline
        self.ready.emit()

        last_seen = first[1]
        face_visible: bool | None = None
        alert = False
        while self._running:
            pending, self._pending_model = self._pending_model, None
            if pending is not None:
                pipeline.set_model(pending)

            item = camera.read(self._fps(time.perf_counter() - last_seen))
            if item is None:
                raise RuntimeError("Se perdió la conexión con la cámara.")
            frame, t = item

            pts = detector.detect(frame, t)
            if pts is not None:
                last_seen = t
            visible = pts is not None
            if visible != face_visible:
                face_visible = visible
                self.face_changed.emit(visible)

            mode = self._mode
            if mode is Mode.CALIBRATE:
                if pts is not None:
                    self.observed.emit(t, pts)
            elif mode is Mode.TRACK and pipeline.model is not None:
                sample = pipeline.track(pts, t)
                self.gaze_moved.emit(sample.x, sample.y, sample.has_point)
                if sample.alert != alert:
                    alert = sample.alert
                    self.alert_changed.emit(alert)


class FitWorker(QThread):
    """Ajusta o evalúa en segundo plano para no congelar la UI."""

    done = pyqtSignal(object, object)  # modelo, reporte
    failed = pyqtSignal(str)

    def __init__(
        self,
        data: dict[str, np.ndarray],
        screen: tuple[int, int],
        cfg: Config,
        frame_size: tuple[int, int],
        info: ScreenInfo | None = None,
        existing: GazeModel | None = None,
    ) -> None:
        super().__init__()
        self._data = data
        self._screen = screen
        self._cfg = cfg
        self._frame_size = frame_size
        self._info = info
        self._existing = existing

    def run(self) -> None:
        """Con `existing` solo mide; sin él ajusta y elige el mejor modelo."""
        try:
            save_session(
                self._data, self._screen, self._frame_size, self._info, self._cfg.session_path
            )
            if self._existing is not None:
                report = assess_model(
                    self._existing, self._data, self._cfg, self._frame_size, self._info
                )
                self.done.emit(self._existing, report)
            else:
                model, report = fit_session(
                    self._data, self._screen, self._cfg, self._frame_size, self._info
                )
                self.done.emit(model, report)
        except Exception as exc:
            self.failed.emit(str(exc))
