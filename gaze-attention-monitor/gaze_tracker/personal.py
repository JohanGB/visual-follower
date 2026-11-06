"""Calibración personal: estima los parámetros de ojo, cámara y rostro.

Ridge solo corrige relaciones suaves entre planos. El error al mover la cabeza
viene de una geometría equivocada (radio y profundidad del ojo, focal, escala
facial), y esos parámetros solo se distinguen con datos en varias poses. Aquí
se eligen los que mejor explican la fase con cabeza en movimiento. Un prior
anatómico evita valores absurdos, y el ancho real del monitor (EDID) da una
referencia física que ata la escala de esa geometría.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime

import cv2
import numpy as np
from scipy.optimize import minimize

from .config import Config
from .display import ScreenInfo
from .geometry import (EyeParams, GeometryEstimator, camera_matrix, eye_aspect_ratio, gaze_points,
                       lid_features, pose_features)
from .model import (GazeModel, _interp_matrix, blink_mask, fit_gaze_model, fit_residual,
                    poly_features)
from .validation import Report, Score, balance, score_points, validation_targets

TAG_STILL = 0
TAG_MOVING = 1
TAG_POINT = 10  # TAG_POINT + i: muestra del punto de validación i

_FOCALS = (0.55, 0.65, 0.8, 0.95, 1.15, 1.4, 1.7)
_BOUNDS = ((6.5, 13.0), (0.0, 14.0), (0.85, 1.2))  # radio, profundidad, escala
_PRIOR_MEAN = np.array([10.0, 5.0, 1.0, 0.95])
_PRIOR_SD = np.array([1.5, 3.0, 0.06, 0.40])
_PRIOR_WEIGHT = 1e-5
_ANCHOR_WEIGHT = 2e-3  # peso del ancho físico del monitor
_TRIM = 0.9
_MIN_HEAD_RANGE = 6.0
_MIN_GAIN = 0.95  # el corrector debe reducir el error al menos un 5 %


@dataclass(frozen=True, slots=True)
class Poses:
    """Pose por frame para una focal dada, con escala facial 1."""

    rots: np.ndarray
    trans: np.ndarray
    valid: np.ndarray


def solve_poses(pts: np.ndarray, cfg: Config, width: int, height: int, focal: float) -> Poses:
    """Resuelve PnP para un lote. OpenCV no ofrece versión por lotes."""
    estimator = GeometryEstimator(cfg, width, height, EyeParams(focal_scale=focal))
    n = pts.shape[0]
    rots, trans, valid = np.zeros((n, 3, 3)), np.zeros((n, 3)), np.zeros(n, bool)
    for i, frame in enumerate(pts):
        pose = estimator.solve_pose(frame)
        if pose is not None:
            rots[i], trans[i], valid[i] = cv2.Rodrigues(pose[0])[0], pose[1].ravel(), True
    return Poses(rots, trans, valid)


def head_range_deg(rots: np.ndarray) -> float:
    """Amplitud robusta del giro de cabeza (yaw o pitch, la mayor)."""
    forward = rots[:, :, 2]
    yaw = np.degrees(np.arctan2(forward[:, 0], forward[:, 2]))
    pitch = np.degrees(np.arcsin(np.clip(forward[:, 1], -1.0, 1.0)))
    low, high = np.percentile(np.column_stack((yaw, pitch)), [5, 95], axis=0)
    return float((high - low).max())


def _geometry_cost(
    values: np.ndarray,
    focal: float,
    pts: np.ndarray,
    poses: Poses,
    K: np.ndarray,
    target: np.ndarray,
    min_forward: float,
    width_mm: float | None,
    lids: np.ndarray,
) -> float:
    """Error tras un ajuste lineal, más el prior anatómico y el ancla física.

    Los párpados entran al ajuste lineal: si no, la geometría se deforma para
    explicar un sesgo vertical que no es geométrico.
    """
    radius, depth, scale = values
    params = EyeParams(radius, depth, focal, scale)
    por, valid = gaze_points(pts, poses.rots, poses.trans * scale, K, params, min_forward)
    valid &= poses.valid
    if valid.mean() < 0.85:
        return 1e3
    z = por[valid]
    std = z.std(0).clip(1e-9)
    z = (z - z.mean(0)) / std
    design = np.column_stack((np.ones(z.shape[0]), poly_features(z), lids[valid]))
    y = target[valid]
    beta = np.linalg.solve(design.T @ design + 1e-6 * np.eye(design.shape[1]), design.T @ y)
    squared = ((design @ beta - y) ** 2).sum(axis=1)
    kept = int(_TRIM * squared.size)
    cost = float(np.partition(squared, kept - 1)[:kept].mean())
    prior = ((np.array([radius, depth, scale, focal]) - _PRIOR_MEAN) / _PRIOR_SD) ** 2
    cost += _PRIOR_WEIGHT * float(prior.sum())
    if width_mm is not None:
        implied = float(std[0]) / max(abs(float(beta[1, 0])), 1e-9)
        cost += _ANCHOR_WEIGHT * float(np.log(implied / width_mm) ** 2)
    return cost


def fit_eye_params(
    pts: np.ndarray,
    target: np.ndarray,
    cfg: Config,
    width: int,
    height: int,
    width_mm: float | None = None,
) -> EyeParams:
    """Busca los parámetros: rejilla en focal y Powell acotado en el resto."""
    raw = lid_features(pts)
    lids = (raw - raw.mean(0)) / raw.std(0).clip(1e-6)
    best, best_cost = None, np.inf
    for focal in _FOCALS:
        poses = solve_poses(pts, cfg, width, height, focal)
        K = camera_matrix(width, height, focal)
        result = minimize(
            _geometry_cost, _PRIOR_MEAN[:3],
            args=(focal, pts, poses, K, target, cfg.min_gaze_forward, width_mm, lids),
            method="Powell", bounds=_BOUNDS,
            options={"xtol": 1e-2, "ftol": 1e-4, "maxfev": 150},
        )
        if result.fun < best_cost:
            best_cost = result.fun
            best = EyeParams(float(result.x[0]), float(result.x[1]), focal, float(result.x[2]))
    return best


@dataclass(frozen=True, slots=True)
class Batch:
    """Mediciones de un lote de frames con unos parámetros dados."""

    por: np.ndarray
    valid: np.ndarray
    pose: np.ndarray
    lids: np.ndarray
    distance: np.ndarray


def _observations(pts: np.ndarray, cfg: Config, size: tuple[int, int], params: EyeParams) -> Batch:
    """Mide un lote completo."""
    poses = solve_poses(pts, cfg, *size, params.focal_scale)
    K = camera_matrix(*size, params.focal_scale)
    scaled = poses.trans * params.face_scale
    por, valid = gaze_points(pts, poses.rots, scaled, K, params, cfg.min_gaze_forward)
    return Batch(
        por, valid & poses.valid, pose_features(poses.rots, poses.trans, params.face_scale),
        lid_features(pts), scaled[:, 2],
    )


def _deg_per_px(screen: tuple[int, int], width_mm: float | None, distance_mm: float) -> float | None:
    if width_mm is None or width_mm < 150.0 or distance_mm <= 0.0:
        return None
    return float(np.degrees(np.arctan2(width_mm / screen[0], distance_mm)))


def _width_of(model: GazeModel, info: ScreenInfo | None) -> float | None:
    return info.width_mm if info is not None else model.screen_width_mm()


def score_model(
    model: GazeModel,
    data: dict[str, np.ndarray],
    cfg: Config,
    size: tuple[int, int],
    info: ScreenInfo | None = None,
) -> Score | None:
    """Evalúa un modelo sobre las muestras de los puntos de validación."""
    chosen = data["tag"] >= TAG_POINT
    pts = data["pts"][chosen]
    if pts.shape[0] == 0:
        return None
    index = data["tag"][chosen] - TAG_POINT
    batch = _observations(pts, cfg, size, model.params)
    keep = batch.valid & (eye_aspect_ratio(pts) >= model.ear_closed)
    if not keep.any():
        return None
    targets, groups = validation_targets()
    return score_points(
        model.predict(batch.por[keep], batch.pose[keep], batch.lids[keep]),
        index[keep], targets, groups, model.screen,
        _deg_per_px(model.screen, _width_of(model, info), float(np.median(batch.distance[keep]))),
    )


def _training_mask(data: dict[str, np.ndarray], ear_ref: float) -> np.ndarray:
    ear = eye_aspect_ratio(data["pts"])
    train = data["tag"] < TAG_POINT
    train[train] = ~blink_mask(ear[train], data["t"][train], ear_ref)
    return train


def _refit(data, cfg, size, screen, params, ear_ref, use_lids: bool) -> GazeModel:
    """Ajusta Ridge sobre las dos fases de entrenamiento."""
    train = data["tag"] < TAG_POINT
    pts, t = data["pts"][train], data["t"][train]
    batch = _observations(pts, cfg, size, params)
    ok = batch.valid
    return fit_gaze_model(
        t[ok], batch.por[ok], eye_aspect_ratio(pts)[ok], data["t_tgt"], data["xy_tgt"],
        screen, cfg.max_lag_s, params, batch.lids[ok] if use_lids else None,
    )


def _add_correction(model: GazeModel, data, cfg, size) -> GazeModel:
    """Aprende el residuo de un modelo con los datos de entrenamiento."""
    mask = _training_mask(data, model.ear_ref)
    pts, t = data["pts"][mask], data["t"][mask]
    batch = _observations(pts, cfg, size, model.params)
    ok = batch.valid
    target = _interp_matrix((t[ok] - model.lag_s)[:, None], data["t_tgt"], data["xy_tgt"])[:, 0]
    xy = model.predict(batch.por[ok], None, batch.lids[ok])
    scaled = model.scaled_lids(batch.lids[ok], xy.shape[:-1])
    residual = fit_residual(xy, batch.pose[ok], target - xy, t[ok], scaled)
    return replace(model, residual=residual)


def _better(candidate: Score | None, current: Score) -> bool:
    return candidate is not None and balance(candidate) <= _MIN_GAIN * balance(current)


def fit_session(
    data: dict[str, np.ndarray],
    screen: tuple[int, int],
    cfg: Config,
    size: tuple[int, int],
    info: ScreenInfo | None = None,
) -> tuple[GazeModel, Report | None]:
    """Ajusta cada capa y solo conserva la que mejora los puntos fijos."""
    tag, pts, t = data["tag"], data["pts"], data["t"]
    still, moving = tag == TAG_STILL, tag == TAG_MOVING
    generic_params = EyeParams(cfg.eye_radius_mm, cfg.eye_depth_mm, cfg.focal_scale)

    # "Antes": cabeza quieta, parámetros por defecto y solo la mirada.
    batch = _observations(pts[still], cfg, size, generic_params)
    ok = batch.valid
    generic = fit_gaze_model(
        t[still][ok], batch.por[ok], eye_aspect_ratio(pts[still])[ok], data["t_tgt"],
        data["xy_tgt"], screen, cfg.max_lag_s, generic_params,
    )

    head = None
    if moving.sum() >= cfg.min_samples / 4:
        poses = solve_poses(pts[moving], cfg, *size, generic_params.focal_scale)
        head = head_range_deg(poses.rots[poses.valid])

    personal = None
    if head is not None and head >= _MIN_HEAD_RANGE:
        mask = _training_mask(data, generic.ear_ref)
        target = _interp_matrix(
            (t[mask] - generic.lag_s)[:, None], data["t_tgt"], data["xy_tgt"]
        )[:, 0]
        width = None if info is None else info.width_mm
        params = fit_eye_params(pts[mask], target, cfg, *size, width)
        personal = _refit(data, cfg, size, screen, params, generic.ear_ref, False)

    base = score_model(generic, data, cfg, size, info)
    if base is None:
        return generic, None

    # Se comparan las combinaciones completas: geometría y párpados interactúan.
    # Una opción más compleja debe superar a la actual por el margen exigido.
    options = [("generico", generic, False)]
    options.append(("generico", _refit(data, cfg, size, screen, generic_params, generic.ear_ref, True), True))
    if personal is not None:
        options.append(("personal", personal, False))
        options.append(("personal", _refit(data, cfg, size, screen, personal.params, generic.ear_ref, True), True))

    name, chosen, lids_active, current = options[0][0], generic, False, base
    for label, model, uses_lids in options[1:]:
        score = score_model(model, data, cfg, size, info)
        if _better(score, current):
            name, chosen, lids_active, current = label, model, uses_lids, score

    if personal is None:
        reason = "La cabeza se movió muy poco para estimar parámetros personales."
    elif name == "personal":
        reason = "El modelo personal redujo el error en la validación."
    else:
        reason = "El modelo personal no mejoró la validación; se mantuvo el genérico."

    before = current
    corrected = _add_correction(chosen, data, cfg, size)
    after = score_model(corrected, data, cfg, size, info)
    active = _better(after, before)
    if active:
        chosen, current = corrected, after

    chosen = replace(chosen, error=current.accuracy_px / float(np.hypot(*screen)))
    return chosen, _report(
        chosen, data, cfg, size, info, name, reason, current, base, head, before, active, lids_active
    )


def _report(model, data, cfg, size, info, name, reason, current, base, head, before, active, lids):
    """Reúne el resumen, incluidas las medidas físicas detectadas."""
    chosen = data["tag"] >= TAG_POINT
    distance = float(np.median(_observations(data["pts"][chosen], cfg, size, model.params).distance))
    width = _width_of(model, info)
    height = None if width is None else width * model.screen[1] / model.screen[0]
    return Report(
        datetime.now().isoformat(timespec="seconds"), model.screen, name, reason, current, base,
        asdict(model.params), head, validation_targets()[0].tolist(), before, active, lids,
        None if info is None and width is None else
        (info.width_mm, info.height_mm) if info else (width, height),
        "EDID" if info else ("estimado" if width else None), distance,
    )


def save_session(data, screen, size, info, path) -> None:
    """Guarda solo coordenadas de landmarks, nunca imágenes, para repetir el análisis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width = np.nan if info is None else info.width_mm
    height = np.nan if info is None else info.height_mm
    np.savez_compressed(
        path, **data, screen=np.array(screen), size=np.array(size),
        screen_mm=np.array([width, height]),
    )


def assess_model(
    model: GazeModel,
    data: dict[str, np.ndarray],
    cfg: Config,
    size: tuple[int, int],
    info: ScreenInfo | None = None,
) -> Report | None:
    """Mide un modelo ya guardado, sin reajustar nada."""
    score = score_model(model, data, cfg, size, info)
    if score is None:
        return None
    return _report(
        model, data, cfg, size, info, "actual", "Validación de la calibración guardada.",
        score, None, None, None, model.residual is not None, model.lid_mean is not None,
    )
