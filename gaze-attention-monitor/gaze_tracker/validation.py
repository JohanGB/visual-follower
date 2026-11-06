"""Validación con puntos fijos: mide el error real con datos que el ajuste no vio.

Separa las dos cantidades que suelen confundirse:
    exactitud: distancia entre el punto mirado y la estimación media (sesgo).
    precisión: dispersión de las muestras alrededor de esa media (temblor).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_MIN_POINT_SAMPLES = 5
_MIN_POINTS = 5


def validation_targets() -> tuple[np.ndarray, np.ndarray]:
    """Devuelve puntos normalizados y su grupo: 0 cabeza natural, 1 cabeza movida."""
    grid = np.array([0.08, 0.5, 0.92])
    natural = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    moved = np.array([[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    points = np.vstack((natural, moved))
    groups = np.repeat([0, 1], (natural.shape[0], moved.shape[0]))
    return points, groups


@dataclass(frozen=True, slots=True)
class Score:
    """Resultado de un modelo sobre los puntos de validación, en píxeles."""

    accuracy_px: float  # todos los puntos
    natural_px: float | None  # cabeza como en la calibración
    moved_px: float | None  # cabeza en otra posición
    precision_px: float
    points_used: int
    accuracy_deg: float | None = None
    estimates: list[list[float]] | None = None  # posición media por punto
    vertical_r2: float | None = None
    vertical_slope: float | None = None
    low_zone_px: float | None = None  # error mediano en el tercio inferior


def score_points(
    xy: np.ndarray,
    index: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    screen: tuple[int, int],
    deg_per_px: float | None = None,
) -> Score | None:
    """Puntúa predicciones normalizadas `xy` etiquetadas con su punto `index`."""
    pixels = np.array(screen, np.float64)
    counts = np.bincount(index, minlength=targets.shape[0])
    used = np.flatnonzero(counts >= _MIN_POINT_SAMPLES)
    if used.size < _MIN_POINTS:
        return None

    # La mediana por punto descarta muestras de un parpadeo o una distracción.
    centers = np.stack([np.median(xy[index == i], axis=0) for i in used])
    errors = np.linalg.norm((centers - targets[used]) * pixels, axis=1)
    lookup = np.full(targets.shape[0], -1)
    lookup[used] = np.arange(used.size)
    keep = lookup[index] >= 0
    spread = (xy[keep] - centers[lookup[index[keep]]]) * pixels
    precision = float(np.median(np.linalg.norm(spread, axis=1)))

    def group_error(group: int) -> float | None:
        chosen = errors[groups[used] == group]
        return float(np.median(chosen)) if chosen.size else None

    accuracy = float(np.median(errors))
    estimates = np.full_like(targets, np.nan)
    estimates[used] = centers
    y_true, y_est = targets[used, 1], centers[:, 1]
    slope = float(np.polyfit(y_true, y_est, 1)[0])
    r2 = float(np.corrcoef(y_true, y_est)[0, 1] ** 2) if y_est.std() > 0 else 0.0
    low = errors[y_true > 2 / 3]
    return Score(
        accuracy, group_error(0), group_error(1), precision, int(used.size),
        None if deg_per_px is None else accuracy * deg_per_px,
        [[None if np.isnan(v) else float(v) for v in row] for row in estimates],
        r2, slope, float(np.median(low)) if low.size else None,
    )


def balance(score: Score) -> float:
    """Criterio de selección: pesa igual cabeza natural y movida."""
    parts = [v for v in (score.natural_px, score.moved_px) if v is not None]
    return float(np.mean(parts))


@dataclass(frozen=True, slots=True)
class Report:
    """Resumen de una calibración o validación, sin imágenes del usuario."""

    created: str
    screen: tuple[int, int]
    chosen: str  # "personal", "generico" o "actual"
    reason: str
    current: Score
    baseline: Score | None
    params: dict[str, float]
    head_range_deg: float | None
    targets: list[list[float]]
    before_correction: Score | None = None  # sin el corrector kernel
    correction_active: bool = False
    lids_active: bool = False
    screen_mm: tuple[float, float] | None = None
    screen_source: str | None = None
    distance_mm: float | None = None


def append_history(report: Report, path: Path) -> None:
    """Agrega una línea JSON para poder comparar sesiones en el tiempo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(report), ensure_ascii=False) + "\n")
