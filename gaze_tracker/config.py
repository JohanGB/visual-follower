"""Parámetros del sistema. Inmutables para poder compartirlos entre hilos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path.home() / ".gaze_tracker"


@dataclass(frozen=True, slots=True)
class Config:
    """Configuración global; cada campo tiene un valor razonable por defecto."""

    # Captura
    camera: int | str = 0
    frame_width: int = 640
    frame_height: int = 480
    fps_active: float = 15.0
    fps_calibration: float = 20.0
    fps_idle: float = 5.0
    idle_after_s: float = 2.0

    # Geometría. La focal real es desconocida: se aproxima a f = focal_scale * ancho.
    focal_scale: float = 0.95
    eye_radius_mm: float = 10.0
    eye_depth_mm: float = 5.0
    min_gaze_forward: float = 0.25

    # Parpadeo: ojo cerrado si EAR < blink_ratio * EAR de referencia.
    blink_ratio: float = 0.55
    default_ear: float = 0.30

    # Filtro One Euro, en fracciones de pantalla.
    filter_min_cutoff: float = 0.7
    filter_beta: float = 1.2

    # Atención
    off_margin: float = 0.08
    alert_after_s: float = 3.0
    recover_after_s: float = 0.3
    blink_grace_s: float = 0.5
    no_face_counts_as_away: bool = True

    # Calibración
    still_s: float = 30.0
    moving_s: float = 35.0
    point_s: float = 1.6
    point_settle_s: float = 0.6
    min_samples: int = 400
    max_lag_s: float = 0.5

    # Mapa de calor
    heatmap_cols: int = 192
    fixation_speed: float = 0.45

    # Rutas
    model_path: Path = APP_DIR / "face_landmarker.task"
    calibration_path: Path = APP_DIR / "calibration.npz"
    history_path: Path = APP_DIR / "validation_history.jsonl"
    session_path: Path = APP_DIR / "last_session.npz"
    output_dir: Path = Path.home() / "Pictures" / "GazeTracker"
