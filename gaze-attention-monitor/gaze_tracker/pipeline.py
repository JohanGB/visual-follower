"""Inferencia por frame, independiente de Qt y de la cámara."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attention import AttentionMonitor, Gaze
from .config import Config
from .geometry import GeometryEstimator, Observation
from .heatmap import FixationHeatmap
from .model import GazeModel, OneEuroFilter


@dataclass(frozen=True, slots=True)
class GazeSample:
    """Resultado de un frame en modo seguimiento."""

    t: float
    x: float
    y: float
    gaze: Gaze
    alert: bool
    has_point: bool


class GazePipeline:
    """Encadena geometría, modelo, filtro, atención y mapa de calor."""

    def __init__(self, cfg: Config, width: int, height: int) -> None:
        self._cfg = cfg
        self._size = (width, height)
        self._geometry = GeometryEstimator(cfg, width, height)
        self._filter = OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta)
        self._attention = AttentionMonitor(cfg.alert_after_s, cfg.recover_after_s, cfg.blink_grace_s)
        self._model: GazeModel | None = None
        self._ear_limit = cfg.blink_ratio * cfg.default_ear
        self._last_xy = np.array([0.5, 0.5])
        self.heatmap: FixationHeatmap | None = None

    @property
    def model(self) -> GazeModel | None:
        """Modelo activo."""
        return self._model

    def set_model(self, model: GazeModel) -> None:
        """Activa un modelo y conserva el mapa si la pantalla no cambió."""
        self._model = model
        self._geometry = GeometryEstimator(self._cfg, *self._size, model.params)
        self._ear_limit = model.ear_closed
        self._filter.reset()
        self._attention.reset()
        if self.heatmap is None or self.heatmap.screen != model.screen:
            self.heatmap = FixationHeatmap(
                self._cfg.heatmap_cols, model.screen, self._cfg.fixation_speed
            )

    def observe(self, pts: np.ndarray | None) -> Observation | None:
        """Medición geométrica cruda, usada también por la calibración."""
        if pts is None:
            self._geometry.reset()
            return None
        return self._geometry.observe(pts)

    def _classify(self, pts: np.ndarray | None, obs: Observation | None) -> tuple[Gaze, bool]:
        if pts is None:
            return (Gaze.OFF if self._cfg.no_face_counts_as_away else Gaze.UNKNOWN), False
        if obs is None:
            return Gaze.UNKNOWN, False
        return Gaze.UNKNOWN if obs.ear < self._ear_limit else Gaze.ON, obs.ear >= self._ear_limit

    def track(self, pts: np.ndarray | None, t: float) -> GazeSample:
        """Procesa un frame en seguimiento. Requiere un modelo activo."""
        obs = self.observe(pts)
        gaze, has_point = self._classify(pts, obs)
        heatmap = self.heatmap

        if has_point:
            xy = self._filter(self._model.predict(obs.por, obs.pose, obs.lids), t)
            self._last_xy = xy
            margin = self._cfg.off_margin
            inside = bool(np.all((xy >= -margin) & (xy <= 1.0 + margin)))
            gaze = Gaze.ON if inside else Gaze.OFF
            heatmap.add(float(xy[0]), float(xy[1]), t)
        else:
            self._filter.reset()
            heatmap.break_sequence()

        alert = self._attention.update(gaze, t)
        return GazeSample(t, float(self._last_xy[0]), float(self._last_xy[1]), gaze, alert, has_point)
