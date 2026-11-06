"""Trayectoria de calibración: curva, aleatoria y con cobertura garantizada."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Muestras uniformes en el tiempo; x e y normalizadas a la pantalla."""

    t: np.ndarray
    xy: np.ndarray
    radius: np.ndarray  # multiplicador alrededor de 1.0


def _waypoints(n: int, rng: np.random.Generator) -> np.ndarray:
    """Puntos estratificados: se recorre toda la rejilla antes de repetir celda."""
    cols, rows = 4, 3
    cells = np.stack(np.unravel_index(np.arange(cols * rows), (cols, rows)), axis=1)
    order = np.concatenate([rng.permutation(cols * rows) for _ in range(n // 12 + 1)])
    picked = cells[order[:n]]
    return (picked + rng.uniform(0.1, 0.9, (n, 2))) / (cols, rows)


def _catmull_rom(points: np.ndarray, per_segment: int) -> np.ndarray:
    """Spline que pasa por todos los puntos, evaluado de forma vectorizada."""
    p = np.vstack((points[0], points, points[-1]))
    s = np.linspace(0.0, 1.0, per_segment, endpoint=False)[None, :, None]
    p0, p1, p2, p3 = (p[i:i + points.shape[0] - 1, None, :] for i in range(4))
    curve = 0.5 * (
        2 * p1
        + (p2 - p0) * s
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s**2
        + (3 * p1 - p0 - 3 * p2 + p3) * s**3
    )
    return curve.reshape(-1, 2)


def _wobble(t: np.ndarray, rng: np.random.Generator, depth: float) -> np.ndarray:
    """Oscilación suave y no periódica alrededor de 1.0."""
    freq = rng.uniform(0.06, 0.28, 3)[:, None]
    phase = rng.uniform(0.0, 2 * np.pi, 3)[:, None]
    return 1.0 + depth * np.sin(2 * np.pi * freq * t + phase).mean(axis=0) * 1.6


def generate_trajectory(
    duration: float,
    rate: float = 60.0,
    seed: int | None = None,
    margin: float = 0.04,
    mean_speed: float = 0.27,
    aspect: float = 16 / 9,
) -> Trajectory:
    """Genera posición y radio. `mean_speed` en anchos de pantalla por segundo."""
    rng = np.random.default_rng(seed)
    n_way = max(6, int(np.ceil(duration * mean_speed / 0.48)))
    curve = _catmull_rom(_waypoints(n_way, rng), per_segment=48)
    low, high = curve.min(0), curve.max(0)
    curve = margin + (curve - low) / (high - low) * (1.0 - 2.0 * margin)

    # Longitud de arco en espacio físico, para que la velocidad sea real.
    steps = np.diff(curve, axis=0) * (1.0, 1.0 / aspect)
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(*steps.T))))

    t = np.arange(0.0, duration, 1.0 / rate)
    speed = _wobble(t, rng, 0.35).clip(0.45)
    s = np.cumsum(speed)
    s = (s - s[0]) / (s[-1] - s[0]) * arc[-1]
    xy = np.column_stack((np.interp(s, arc, curve[:, 0]), np.interp(s, arc, curve[:, 1])))
    return Trajectory(t, xy, _wobble(t, rng, 0.22))
