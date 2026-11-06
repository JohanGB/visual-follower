"""Banco sintético: genera landmarks desde una verdad geométrica conocida.

El simulador usa parámetros de ojo, cámara y escala facial DISTINTOS a los que
asume el estimador, para no medir un caso ideal. Compara el vector
iris-comisuras clásico con la intersección 3D de `geometry.py`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np

from gaze_tracker.config import Config
from gaze_tracker.geometry import (FACE_MODEL_MM, LANDMARK_IDS, GeometryEstimator)
from gaze_tracker.model import fit_gaze_model
from gaze_tracker.personal import TAG_MOVING, TAG_POINT, TAG_STILL, fit_session
from gaze_tracker.validation import validation_targets
from gaze_tracker.trajectory import generate_trajectory

W, H = 640, 480
SCREEN_PX = (1920, 1080)


@dataclass(frozen=True)
class Truth:
    """Parámetros reales, desconocidos para el estimador."""

    focal: float = 0.80 * W
    face_scale: float = 1.04
    eye_radius: float = 10.6
    eye_depth: float = 6.5
    kappa_deg: tuple[float, float] = (4.0, 1.5)
    screen_mm: tuple[float, float] = (530.0, 300.0)
    tilt_deg: float = 8.0
    noise_px: float = 0.25
    bias_mm: float = 0.0  # sesgo no lineal del detector de iris
    bias_shape: str = "poly"  # "poly" u "wave" (forma no polinómica)
    pursuit_error: float = 0.0  # error de seguimiento, en fracción de pantalla
    lid_coupling: float = 0.0  # los párpados siguen a la mirada vertical
    down_bias: float = 0.0  # el párpado oculta el iris y frena su descenso aparente


def screen_points(uv: np.ndarray, tr: Truth) -> np.ndarray:
    """Pantalla bajo la cámara; su x va invertida respecto a la imagen."""
    a = np.radians(tr.tilt_deg)
    u, v = uv[..., 0], uv[..., 1]
    return np.stack(
        (-(u - 0.5) * tr.screen_mm[0],
         12.0 + v * tr.screen_mm[1] * np.cos(a),
         v * tr.screen_mm[1] * np.sin(a)),
        axis=-1,
    )


def render(uv: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray, tr: Truth,
           rng: np.random.Generator) -> np.ndarray:
    """Devuelve landmarks (n, 24, 2) en píxeles, ordenados como LANDMARK_IDS."""
    n = uv.shape[0]
    rots = np.stack([cv2.Rodrigues(r)[0] for r in rvecs])
    model = FACE_MODEL_MM * tr.face_scale
    face = model @ rots.transpose(0, 2, 1) + tvecs[:, None]  # (n, 10, 3)

    mids = 0.5 * (model[[0, 2]] + model[[1, 3]]) + (0.0, 0.0, tr.eye_depth)
    eyes = mids @ rots.transpose(0, 2, 1) + tvecs[:, None]  # (n, 2, 3)
    target = screen_points(uv, tr)[:, None]
    gaze = target - eyes
    gaze /= np.linalg.norm(gaze, axis=2, keepdims=True)
    gaze[..., :2] += np.tan(np.radians(tr.kappa_deg))
    gaze /= np.linalg.norm(gaze, axis=2, keepdims=True)
    iris = eyes + gaze * tr.eye_radius  # (n, 2, 3)
    if tr.bias_mm:
        # Sesgo plausible: el detector comprime giros grandes, el párpado desplaza
        # el iris en vertical y la cara girada altera la estimación.
        local = gaze @ rots  # mirada en ejes de la cabeza
        ex, ey = local[..., 0], local[..., 1]
        yaw = rots[:, 0, 2][:, None]
        dx = 3.0 * ex**3 + 0.8 * ex * ey + 0.6 * yaw * ey
        dy = 2.0 * ey**2 - 0.8 * ey + 0.7 * yaw**2
        if tr.bias_shape == "wave":
            dx = 0.9 * np.sin(5.0 * ex) * np.cos(3.0 * ey) + 0.5 * np.tanh(9.0 * yaw) * ey
            dy = 0.8 * np.tanh(7.0 * (ey - 0.1)) + 0.4 * np.sin(4.0 * yaw + 3.0 * ex)
        shift = np.stack((dx, dy, np.zeros_like(dx)), axis=-1) * tr.bias_mm
        iris = iris + shift @ rots.transpose(0, 2, 1)

    local = gaze @ rots
    ey = local[..., 1]
    if tr.down_bias:
        down = np.clip(ey, 0.0, None)
        shift = np.stack((np.zeros_like(down), -tr.down_bias * down, np.zeros_like(down)), -1)
        iris = iris + shift @ rots.transpose(0, 2, 1)

    ring = np.array([[0, 0, 0], [5.8, 0, 0], [0, -5.8, 0], [-5.8, 0, 0], [0, 5.8, 0]], float)
    iris_pts = iris[:, :, None] + (ring @ rots.transpose(0, 2, 1))[:, None]
    lid = np.array([[0, -3.3, 0], [0, 3.3, 0]], float)
    moved = np.stack((ey * 4.5, ey * 1.2), axis=-1) * tr.lid_coupling
    lid_head = mids[:, None] + lid[None] + np.stack(
        (np.zeros_like(moved), moved, np.zeros_like(moved)), axis=-1
    )[None].squeeze(0) if tr.lid_coupling else mids[:, None] + lid[None]
    if lid_head.ndim == 3:
        lid_head = np.broadcast_to(lid_head, (n, 2, 2, 3))
    lids = lid_head @ rots[:, None].transpose(0, 1, 3, 2) + tvecs[:, None, None]

    pts = np.concatenate((face, iris_pts.reshape(n, 10, 3), lids.reshape(n, 4, 3)), axis=1)
    px = pts[..., :2] / pts[..., 2:] * tr.focal + (W / 2, H / 2)
    return px + rng.normal(0.0, tr.noise_px, px.shape)


def naive_feature(px: np.ndarray) -> np.ndarray:
    """Iris relativo a las comisuras, normalizado por ancho de ojo (línea base)."""
    corners = 0.5 * (px[:, [0, 2]] + px[:, [1, 3]])
    width = np.linalg.norm(px[:, [0, 2]] - px[:, [1, 3]], axis=2, keepdims=True)
    iris = px[:, 10:20].reshape(-1, 2, 5, 2).mean(2)
    return ((iris - corners) / width).mean(axis=1)


def geometric_feature(px, cfg: Config, params=None):
    """Aplica el estimador real; devuelve características y máscara de validez."""
    est = GeometryEstimator(cfg, W, H, params)
    out, ok = np.full((px.shape[0], 2), np.nan), np.zeros(px.shape[0], bool)
    for i, frame in enumerate(px):
        obs = est.observe(frame)
        if obs is not None:
            out[i], ok[i] = obs.por, True
    return out, ok


def head_state(n: int, rng, yaw=0.0, pitch=0.0, dx=0.0, dy=0.0, dz=0.0, sway=True):
    """Pose base mirando a la pantalla, con desplazamiento y vaivén leve."""
    base_t = np.array([dx, 90.0 + dy, 620.0 + dz])
    base_r = np.radians([-8.0 + pitch, yaw, 0.0])
    t = np.tile(base_t, (n, 1))
    r = np.tile(base_r, (n, 1))
    if sway:
        t = t + rng.normal(0, 2.0, t.shape)
        r = r + rng.normal(0, np.radians(0.5), r.shape)
    return r, t


def calibrate(tr: Truth, cfg: Config, rng, seed=7):
    """Corre la calibración completa a 20 FPS con retraso real de 180 ms."""
    duration = cfg.still_s + cfg.moving_s
    traj = generate_trajectory(duration, seed=seed)
    t_obs = np.arange(0.5, duration - 0.5, 1 / cfg.fps_calibration)
    lag = 0.18
    uv = np.column_stack([np.interp(t_obs - lag, traj.t, traj.xy[:, i]) for i in (0, 1)])
    r, t = head_state(t_obs.size, rng)
    px = render(uv, r, t, tr, rng)
    por, ok = geometric_feature(px, cfg)
    ear = np.full(t_obs.size, 0.26)
    model = fit_gaze_model(t_obs[ok], por[ok], ear[ok], traj.t, traj.xy, SCREEN_PX, cfg.max_lag_s)
    naive = fit_gaze_model(t_obs, naive_feature(px), ear, traj.t, traj.xy, SCREEN_PX, cfg.max_lag_s)
    return model, naive, lag


def session(tr: Truth, cfg: Config, rng, head_moves: bool = True) -> dict:
    """Reproduce una sesión completa: quieta, en movimiento y puntos de validación."""
    a = generate_trajectory(cfg.still_s, seed=7)
    b = generate_trajectory(cfg.moving_s, seed=8)
    gap = 3.0
    t_tgt = np.concatenate((a.t, a.t[-1] + gap + b.t))
    xy_tgt = np.vstack((a.xy, b.xy))
    lag = 0.18

    t_a = np.arange(0.5, cfg.still_s - 0.2, 1 / cfg.fps_calibration)
    t_b = np.arange(a.t[-1] + gap + 0.5, t_tgt[-1], 1 / cfg.fps_calibration)
    t = np.concatenate((t_a, t_b))
    uv = np.column_stack([np.interp(t - lag, t_tgt, xy_tgt[:, i]) for i in (0, 1)])
    if tr.pursuit_error:
        # Error de seguimiento: correlacionado en ~0.4 s, como microcorrecciones.
        noise = rng.normal(0, 1, uv.shape)
        kernel = np.ones(9) / 3.0
        uv = uv + tr.pursuit_error * np.column_stack(
            [np.convolve(noise[:, i], kernel, "same") for i in (0, 1)]
        )
    r, tv = head_state(t.size, rng)
    if head_moves:
        phase = (t_b - t_b[0]) / cfg.moving_s * 2 * np.pi
        k = t_a.size
        r[k:, 1] += np.radians(14) * np.sin(3 * phase)
        r[k:, 0] += np.radians(8) * np.sin(2.1 * phase)
        tv[k:, 0] += 70 * np.sin(1.5 * phase)
        tv[k:, 2] += 90 * np.cos(2.4 * phase)
    pts = render(uv, r, tv, tr, rng)
    tag = np.concatenate((np.full(t_a.size, TAG_STILL), np.full(t_b.size, TAG_MOVING)))

    targets, groups = validation_targets()
    per_point = round((cfg.point_s - cfg.point_settle_s) * cfg.fps_calibration)
    index = np.repeat(np.arange(targets.shape[0]), per_point)
    rv, tv2 = head_state(index.size, rng)
    moved = groups[index] == 1
    rv[moved] += np.radians([-6.0, 12.0, 0.0])
    tv2[moved] += (-50.0, 0.0, 80.0)
    # Una fijación real no cae exacta: se añade microdesviación.
    uv_val = targets[index] + rng.normal(0, 0.004, (index.size, 2))
    pts_val = render(uv_val, rv, tv2, tr, rng)
    t_val = t[-1] + 1.0 + np.arange(index.size) / cfg.fps_calibration
    return {
        "t": np.concatenate((t, t_val)),
        "pts": np.vstack((pts, pts_val)),
        "tag": np.concatenate((tag, TAG_POINT + index)),
        "t_tgt": t_tgt,
        "xy_tgt": xy_tgt,
    }


def evaluate(model, naive, tr, cfg, rng, n=800):
    """Mide el error mediano por escenario de pose, en píxeles de 1920×1080."""
    uv = rng.uniform(0.02, 0.98, (n, 2))
    scenes = {
        "calibración": dict(),
        "x +100 mm": dict(dx=100),
        "y +60 mm": dict(dy=60),
        "z -150 mm": dict(dz=-150),
        "z +200 mm": dict(dz=200),
        "yaw +20°": dict(yaw=20),
        "yaw -20°": dict(yaw=-20),
        "pitch +12°": dict(pitch=12),
        "combinado": dict(yaw=15, pitch=-8, dx=-80, dz=120),
    }
    scale = np.array(SCREEN_PX, float)
    rows = []
    for name, kw in scenes.items():
        r, t = head_state(n, rng, **kw)
        px = render(uv, r, t, tr, rng)
        por, ok = geometric_feature(px, cfg)
        e_geo = np.linalg.norm((model.predict(por[ok]) - uv[ok]) * scale, axis=1)
        e_naive = np.linalg.norm((naive.predict(naive_feature(px)) - uv) * scale, axis=1)
        rows.append((name, float(np.median(e_naive)), float(np.median(e_geo)), float(ok.mean())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", type=float, default=0.25, help="ruido de landmarks en px")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--bias", type=float, default=0.0, help="sesgo no lineal del detector, en mm")
    args = ap.parse_args()
    cfg = Config()
    tr = Truth(noise_px=args.noise, bias_mm=args.bias, bias_shape="wave", pursuit_error=0.012)
    rng = np.random.default_rng(args.seed)
    model, naive, lag = calibrate(tr, cfg, rng)
    print(f"retraso real {lag*1e3:.0f} ms, estimado {model.lag_s*1e3:.0f} ms; "
          f"error de validación {model.error*100:.2f}% de la diagonal")
    print(f"{'escenario':<14}{'iris-comisuras':>16}{'geometría 3D':>14}{'válidos':>10}")
    for name, a, b, v in evaluate(model, naive, tr, cfg, rng):
        print(f"{name:<14}{a:>13.0f} px{b:>11.0f} px{v:>10.0%}")

    personal, report = fit_session(session(tr, cfg, rng), SCREEN_PX, cfg, (W, H))
    print(f"\nCalibración personal ({report.chosen}): {personal.params}")
    print(f"{'validación':<18}{'genérico':>10}{'actual':>10}")
    base, now = report.baseline, report.current
    print(f"{'postura normal':<18}{base.natural_px:>7.0f} px{now.natural_px:>7.0f} px")
    print(f"{'cabeza movida':<18}{base.moved_px:>7.0f} px{now.moved_px:>7.0f} px")
    print(f"corrector kernel activo: {report.correction_active}")


if __name__ == "__main__":
    main()
