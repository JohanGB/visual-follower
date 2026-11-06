"""Modelo predictivo: del punto de mirada en mm a coordenadas de pantalla.

Salida normalizada: (0, 0) es la esquina superior izquierda y (1, 1) la
inferior derecha, de modo que valores fuera de [0, 1] significan fuera de la
pantalla.

Dos capas aprendidas de la calibración de cada persona:
    Ridge: mapeo global que extrapola. Una SVR-RBF volvería a la media fuera de
    los datos y enmascararía justo las miradas fuera de pantalla.
    Residuo kernel (KRR-RBF): corrige los sesgos no lineales que quedan. Lejos
    de los datos su corrección tiende a cero, así que nunca rompe la extrapolación.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold

from .geometry import EyeParams

_ALPHAS = np.logspace(-4, 2, 13)
_BLOCK_S = 2.0
_MIN_BLOCKS = 10  # 20 s de datos repartidos: por debajo no hay validación honesta
_VERSION = 4
_READABLE_VERSIONS = (2, 3, 4)
_BLINK_MAX_S = 0.5
_LID_CLIP = 3.0
_KERNEL_GRID = {"alpha": [0.03, 0.1, 0.3, 1.0, 3.0], "gamma": [0.01, 0.03, 0.1, 0.3]}
_KERNEL_POINTS = 450
_KERNEL_CLIP = 2.5
# Escalas mínimas de (x, y, yaw, pitch, distancia): una variable casi constante
# no debe convertirse en ruido al estandarizarla.
_KERNEL_MIN_STD = np.array([0.1, 0.1, 0.05, 0.05, 0.03, 0.02, 0.05])


def poly_features(z: np.ndarray) -> np.ndarray:
    """Expansión de grado 2 sin sesgo: x, y, x², xy, y²."""
    x, y = z[..., 0], z[..., 1]
    return np.stack((x, y, x * x, x * y, y * y), axis=-1)


def design_matrix(z: np.ndarray, lids: np.ndarray | None = None) -> np.ndarray:
    """Polinomio de la mirada y, si hay, términos verticales de los párpados."""
    base = poly_features(z)
    if lids is None:
        return base
    e, u, y = lids[..., 0], lids[..., 1], z[..., 1]
    return np.concatenate((base, np.stack((e, u, e * y, u * y), axis=-1)), axis=-1)


def blink_mask(ear: np.ndarray, t: np.ndarray, ear_ref: float) -> np.ndarray:
    """Marca parpadeos: caídas breves de la apertura o cierre casi total.

    La duración es lo que los separa de mirar abajo, que también baja el párpado
    pero se sostiene. Un umbral fijo descartaría justo la zona baja de la pantalla.
    """
    low = ear < 0.6 * ear_ref
    edges = np.flatnonzero(np.diff(np.concatenate(([False], low, [False])).astype(np.int8)))
    starts, stops = edges[::2], edges[1::2]
    short = t[stops - 1] - t[starts] < _BLINK_MAX_S
    run = np.cumsum(np.bincount(starts, minlength=low.size + 1)
                    - np.bincount(stops, minlength=low.size + 1))[:-1]
    run_id = np.cumsum(np.bincount(starts, minlength=low.size))[:low.size] - 1
    blink = np.zeros(low.size, bool)
    in_run = run > 0
    blink[in_run] = short[run_id[in_run]]
    return blink | (ear < 0.3 * ear_ref)


def _interp_matrix(query: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Interpolación lineal de `v(t)` en una matriz de tiempos `query`."""
    i = np.clip(np.searchsorted(t, query), 1, t.size - 1)
    w = np.clip((query - t[i - 1]) / (t[i] - t[i - 1]), 0.0, 1.0)
    return v[i - 1] + (v[i] - v[i - 1]) * w[..., None]


def estimate_lag(
    t_obs: np.ndarray,
    por: np.ndarray,
    t_tgt: np.ndarray,
    xy_tgt: np.ndarray,
    max_lag: float,
    step: float = 0.02,
) -> float:
    """Busca el retraso mirada+cámara que mejor explica el objetivo.

    Una sola pseudoinversa evalúa todos los candidatos a la vez: lo que cambia
    entre retrasos es la matriz objetivo, nunca el diseño.
    """
    lags = np.arange(0.0, max_lag + step / 2, step)
    z = (por - por.mean(0)) / por.std(0).clip(1e-9)
    design = np.column_stack((np.ones(z.shape[0]), poly_features(z)))

    # Y: (n_obs, 2 * n_lags). Cada par de columnas es un retraso candidato.
    query = t_obs[:, None] - lags[None, :]
    targets = _interp_matrix(query, t_tgt, xy_tgt).reshape(t_obs.size, -1)
    residual = targets - design @ (np.linalg.pinv(design) @ targets)
    sse = (residual**2).sum(axis=0).reshape(-1, 2).sum(axis=1)
    return float(lags[np.argmin(sse)])


def _fit_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, Ridge]:
    """Elige alpha con folds por bloque de tiempo y ajusta."""
    folds = list(GroupKFold(n_splits=5).split(x, groups=groups))
    alpha = float(RidgeCV(alphas=_ALPHAS, cv=folds).fit(x, y).alpha_)
    return alpha, Ridge(alpha=alpha).fit(x, y)


@dataclass(frozen=True, slots=True)
class ResidualKernel:
    """Corrector no lineal en forma dual: inferencia NumPy, sin pickle."""

    centers: np.ndarray  # (m, 5) estandarizados
    dual: np.ndarray  # (m, 2)
    gamma: float
    mean: np.ndarray
    std: np.ndarray

    def correction(
        self, xy: np.ndarray, pose: np.ndarray, lids: np.ndarray | None = None
    ) -> np.ndarray:
        """Corrección en coordenadas de pantalla; acepta un punto o un lote."""
        parts = (xy, pose) if self.mean.size == 5 else (xy, pose, lids)
        x = np.concatenate(parts, axis=-1)
        z = np.clip((x - self.mean) / self.std, -_KERNEL_CLIP, _KERNEL_CLIP)
        distance = ((z[..., None, :] - self.centers) ** 2).sum(axis=-1)
        return np.exp(-self.gamma * distance) @ self.dual


def fit_residual(
    xy: np.ndarray,
    pose: np.ndarray,
    residual: np.ndarray,
    t: np.ndarray,
    lids: np.ndarray | None = None,
) -> ResidualKernel:
    """Ajusta el corrector eligiendo alpha y gamma con bloques de tiempo."""
    error = np.linalg.norm(residual, axis=1)
    keep = error < 4.0 * np.median(error)
    x = np.column_stack((xy, pose) if lids is None else (xy, pose, lids))[keep]
    y = residual[keep]
    groups = ((t[keep] - t[keep][0]) / _BLOCK_S).astype(np.int64)

    mean, std = x.mean(0), np.maximum(x.std(0), _KERNEL_MIN_STD[: x.shape[1]])
    z = np.clip((x - mean) / std, -_KERNEL_CLIP, _KERNEL_CLIP)
    step = max(1, z.shape[0] // _KERNEL_POINTS)
    z, y, groups = z[::step], y[::step], groups[::step]

    folds = list(GroupKFold(n_splits=4).split(z, groups=groups))
    search = GridSearchCV(KernelRidge(kernel="rbf"), _KERNEL_GRID, cv=folds).fit(z, y)
    best = search.best_estimator_
    return ResidualKernel(z, best.dual_coef_, float(best.gamma), mean, std)


@dataclass(frozen=True, slots=True)
class GazeModel:
    """Parámetros mínimos para inferir sin pasar por sklearn."""

    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray  # (2, 5)
    intercept: np.ndarray  # (2,)
    lag_s: float
    error: float  # error mediano en validación, fracción de la diagonal
    ear_ref: float
    screen: tuple[int, int]
    params: EyeParams = EyeParams()
    residual: ResidualKernel | None = None
    ear_closed: float = 0.15  # apertura por debajo de la cual el ojo está cerrado
    lid_mean: np.ndarray | None = None
    lid_std: np.ndarray | None = None

    def scaled_lids(self, lids: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
        """Estandariza y acota los párpados; sin dato equivale a su valor medio."""
        if self.lid_mean is None:
            return None
        if lids is None:
            return np.zeros(shape + (2,))
        return np.clip((lids - self.lid_mean) / self.lid_std, -_LID_CLIP, _LID_CLIP)

    def predict(
        self, por: np.ndarray, pose: np.ndarray | None = None, lids: np.ndarray | None = None
    ) -> np.ndarray:
        """Predice (x, y) normalizado; acepta un punto o un lote (n, 2).

        Sin `pose` no se aplica el corrector kernel.
        """
        z = (por - self.mean) / self.std
        scaled = self.scaled_lids(lids, z.shape[:-1])
        xy = design_matrix(z, scaled) @ self.coef.T + self.intercept
        if self.residual is None or pose is None:
            return xy
        return xy + self.residual.correction(xy, pose, scaled)

    def screen_width_mm(self) -> float | None:
        """Ancho de pantalla implícito en el mapeo, en los mm del plano de la cámara."""
        gain = abs(float(self.coef[0, 0])) / float(self.std[0])
        return 1.0 / gain if gain > 1e-9 else None

    def save(self, path: Path) -> None:
        """Guarda en .npz, sin pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.residual is not None:
            r = self.residual
            extra = dict(r_centers=r.centers, r_dual=r.dual, r_gamma=r.gamma,
                         r_mean=r.mean, r_std=r.std)
        if self.lid_mean is not None:
            extra.update(lid_mean=self.lid_mean, lid_std=self.lid_std)
        np.savez(
            path, version=_VERSION, mean=self.mean, std=self.std, coef=self.coef,
            intercept=self.intercept, lag_s=self.lag_s, error=self.error,
            ear_ref=self.ear_ref, screen=np.array(self.screen),
            params=np.array(astuple(self.params)), ear_closed=self.ear_closed, **extra,
        )

    @classmethod
    def load(cls, path: Path) -> GazeModel | None:
        """Carga una calibración; None si no existe o no es compatible."""
        try:
            with np.load(path, allow_pickle=False) as d:
                if int(d["version"]) not in _READABLE_VERSIONS:
                    return None
                residual = None
                if "r_centers" in d:
                    residual = ResidualKernel(
                        d["r_centers"], d["r_dual"], float(d["r_gamma"]), d["r_mean"], d["r_std"]
                    )
                return cls(
                    d["mean"], d["std"], d["coef"], d["intercept"],
                    float(d["lag_s"]), float(d["error"]), float(d["ear_ref"]),
                    (int(d["screen"][0]), int(d["screen"][1])),
                    EyeParams(*(float(v) for v in d["params"])), residual,
                    float(d["ear_closed"]) if "ear_closed" in d else 0.55 * float(d["ear_ref"]),
                    d["lid_mean"] if "lid_mean" in d else None,
                    d["lid_std"] if "lid_std" in d else None,
                )
        except (OSError, KeyError, ValueError):
            return None


def fit_gaze_model(
    t_obs: np.ndarray,
    por: np.ndarray,
    ear: np.ndarray,
    t_tgt: np.ndarray,
    xy_tgt: np.ndarray,
    screen: tuple[int, int],
    max_lag: float = 0.5,
    params: EyeParams = EyeParams(),
    lids: np.ndarray | None = None,
) -> GazeModel:
    """Ajusta el modelo con muestras de calibración y valida por bloques de tiempo."""
    # Fuera los parpadeos: el iris se colapsa y contamina el ajuste.
    ear_ref = float(np.median(ear))
    keep = ~blink_mask(ear, t_obs, ear_ref)
    ear_closed = max(0.75 * float(np.percentile(ear[keep], 2)), 0.3 * ear_ref)
    t_obs, por = t_obs[keep], por[keep]
    lid_mean = lid_std = lid_z = None
    if lids is not None:
        lids = lids[keep]
        lid_mean, lid_std = lids.mean(0), lids.std(0).clip(1e-6)
        lid_z = np.clip((lids - lid_mean) / lid_std, -_LID_CLIP, _LID_CLIP)

    lag = estimate_lag(t_obs, por, t_tgt, xy_tgt, max_lag)
    y = _interp_matrix((t_obs - lag)[:, None], t_tgt, xy_tgt)[:, 0]

    mean, std = por.mean(0), por.std(0).clip(1e-9)
    x = design_matrix((por - mean) / std, lid_z)

    # Las muestras contiguas se parecen: mezclarlas daría un error optimista.
    # Por eso se valida con bloques de tiempo completos.
    blocks = ((t_obs - t_obs[0]) / _BLOCK_S).astype(np.int64)
    _, blocks = np.unique(blocks, return_inverse=True)
    if blocks.max() + 1 < _MIN_BLOCKS:
        raise ValueError("Muestras insuficientes o concentradas en pocos segundos.")
    holdout = blocks % 5 == 4
    train = ~holdout

    # Primera pasada robusta: descarta saltos sacádicos y distracciones.
    _, first = _fit_ridge(x[train], y[train], blocks[train])
    err = np.linalg.norm(first.predict(x[train]) - y[train], axis=1)
    med = np.median(err)
    inlier = err <= med + 3.0 * 1.4826 * np.median(np.abs(err - med))
    _, second = _fit_ridge(x[train][inlier], y[train][inlier], blocks[train][inlier])

    w, h = screen
    pixels = np.array([w, h], np.float64)
    diag = float(np.hypot(w, h))
    val_px = np.linalg.norm((second.predict(x[holdout]) - y[holdout]) * pixels, axis=1)

    # Modelo final con todos los datos limpios.
    all_err = np.linalg.norm(second.predict(x) - y, axis=1)
    med = np.median(all_err)
    clean = all_err <= med + 3.0 * 1.4826 * np.median(np.abs(all_err - med))
    _, final = _fit_ridge(x[clean], y[clean], blocks[clean])

    return GazeModel(
        mean, std, final.coef_, final.intercept_, lag,
        float(np.median(val_px)) / diag, ear_ref, screen, params, None, ear_closed,
        lid_mean, lid_std,
    )


class OneEuroFilter:
    """Filtro adaptativo: suave en fijaciones, rápido en sacádicos."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        """Olvida el estado, p. ej. tras perder el rostro."""
        self._x: np.ndarray | None = None
        self._dx = np.zeros(2)
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Filtra una muestra 2D con su marca de tiempo."""
        if self._x is None:
            self._x, self._t = x.copy(), t
            return x
        dt = max(t - self._t, 1e-3)
        self._t = t
        self._dx += self._alpha(self._d_cutoff, dt) * ((x - self._x) / dt - self._dx)
        cutoff = self._min_cutoff + self._beta * np.abs(self._dx)
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        return self._x
