"""Núcleo geométrico: de landmarks 2D a un punto de mirada invariante a la pose.

Flujo por frame, sin bucles de Python:
    landmarks -> solvePnP (R, t) -> centro del ojo 3D -> rayo cámara-iris
    -> intersección con el globo ocular -> vector de mirada
    -> corte con el plano de la cámara (z = 0), en milímetros.

Ese punto de corte descuenta rotación y traslación de la cabeza. Lo que queda
(posición de la pantalla, ángulo kappa de cada persona) es casi afín y lo
absorbe el regresor de `model.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config

# Puntos rígidos para PnP. Se excluyen boca y mentón, que se mueven al hablar.
_PNP_IDS = (33, 133, 362, 263, 168, 6, 1, 10, 234, 454)
_IRIS_IDS = (468, 469, 470, 471, 472, 473, 474, 475, 476, 477)
_LID_IDS = (159, 145, 386, 374)

#: Índices que `capture.py` extrae de los 478 landmarks, en este orden.
LANDMARK_IDS: tuple[int, ...] = _PNP_IDS + _IRIS_IDS + _LID_IDS

_N_PNP = len(_PNP_IDS)
_IRIS = slice(_N_PNP, _N_PNP + 10)
_LIDS = slice(_N_PNP + 10, _N_PNP + 14)

# canonical_face_model.obj de MediaPipe (cm), pasado a mm y a ejes de OpenCV
# (x derecha, y abajo, z hacia adelante): se invierten y, z.
_CANONICAL_CM = np.array(
    [
        [-4.445859, 2.663991, 3.173422],  # 33  comisura externa, ojo derecho
        [-1.856432, 2.585245, 3.757904],  # 133 comisura interna, ojo derecho
        [1.856432, 2.585245, 3.757904],  # 362 comisura interna, ojo izquierdo
        [4.445859, 2.663991, 3.173422],  # 263 comisura externa, ojo izquierdo
        [0.000000, 3.271027, 5.236015],  # 168 entrecejo
        [0.000000, 2.473255, 5.788627],  # 6   puente nasal
        [0.000000, -1.126865, 7.475604],  # 1   punta de la nariz
        [0.000000, 8.261778, 4.481535],  # 10  frente
        [-7.664182, 0.673132, -2.435867],  # 234 lateral derecho
        [7.664182, 0.673132, -2.435867],  # 454 lateral izquierdo
    ],
    dtype=np.float64,
)
FACE_MODEL_MM = _CANONICAL_CM * np.array([10.0, -10.0, -10.0])

# Punto medio de las comisuras de cada ojo: [derecho, izquierdo].
_EYE_MIDPOINTS_MM = FACE_MODEL_MM[[0, 2]] * 0.5 + FACE_MODEL_MM[[1, 3]] * 0.5


@dataclass(frozen=True, slots=True)
class EyeParams:
    """Parámetros que varían por persona y por cámara."""

    radius: float = 10.0  # centro del ojo -> centro del iris (mm)
    depth: float = 5.0  # centro del ojo detrás de las comisuras (mm)
    focal_scale: float = 0.95  # focal = focal_scale * ancho de imagen
    face_scale: float = 1.0  # tamaño del rostro frente al modelo canónico


@dataclass(frozen=True, slots=True)
class Observation:
    """Medición geométrica de un frame."""

    por: np.ndarray  # punto de mirada en el plano de la cámara (mm)
    pose: np.ndarray  # yaw y pitch (rad), log de la distancia relativa
    lids: np.ndarray  # apertura y posición del iris entre párpados
    ear: float  # apertura del ojo (párpados / ancho)
    rvec: np.ndarray
    tvec: np.ndarray


def camera_matrix(width: int, height: int, focal_scale: float) -> np.ndarray:
    """Intrínsecos aproximados para una webcam sin calibrar."""
    f = focal_scale * width
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float64)


def eye_aspect_ratio(pts: np.ndarray) -> np.ndarray:
    """Apertura media de ambos ojos. Acepta (24, 2) o un lote (n, 24, 2)."""
    lids = pts[..., _LIDS, :]
    gap = np.linalg.norm(
        np.stack((lids[..., 0, :] - lids[..., 1, :], lids[..., 2, :] - lids[..., 3, :]), -2),
        axis=-1,
    )
    width = np.linalg.norm(pts[..., [0, 2], :] - pts[..., [1, 3], :], axis=-1)
    return (gap / width).mean(axis=-1)


def lid_features(pts: np.ndarray) -> np.ndarray:
    """Señal vertical de los párpados, para corregir las miradas hacia abajo.

    Devuelve (apertura, altura del iris entre párpados), donde 0 es el párpado
    inferior y 1 el superior. Acepta (24, 2) o un lote (n, 24, 2).
    """
    lids = pts[..., _LIDS, :]
    upper, lower = lids[..., [0, 2], :], lids[..., [1, 3], :]
    axis = upper - lower
    iris = pts[..., _IRIS, :].reshape(pts.shape[:-2] + (2, 5, 2)).mean(axis=-2)
    gap2 = np.maximum((axis * axis).sum(axis=-1), 1e-6)
    height = ((iris - lower) * axis).sum(axis=-1) / gap2
    return np.stack((eye_aspect_ratio(pts), np.clip(height, -1.0, 2.0).mean(axis=-1)), axis=-1)


def pose_features(rots: np.ndarray, trans: np.ndarray, face_scale: float) -> np.ndarray:
    """Resume la pose para el corrector: yaw, pitch y log(distancia / 600 mm)."""
    forward = rots[:, :, 2]
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    pitch = np.arcsin(np.clip(forward[:, 1], -1.0, 1.0))
    distance = np.log(np.maximum(trans[:, 2] * face_scale, 1.0) / 600.0)
    return np.column_stack((yaw, pitch, distance))


def gaze_points(
    pts: np.ndarray,
    rots: np.ndarray,
    trans: np.ndarray,
    K: np.ndarray,
    params: EyeParams,
    min_forward: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Corta la mirada con z=0 en un lote; devuelve (puntos en mm, máscara válida).

    `pts` es (n, 24, 2), `rots` (n, 3, 3) y `trans` (n, 3).
    """
    f = K[[0, 1], [0, 1]]
    c = K[[0, 1], [2, 2]]

    # Centro del ojo. Se ancla a las comisuras medidas (no a la proyección de
    # PnP): así el temblor de la pose no se suma al del iris.
    corners_mid = 0.5 * (pts[:, [0, 2]] + pts[:, [1, 3]])
    z_mid = (rots[:, 2] @ (_EYE_MIDPOINTS_MM * params.face_scale).T + trans[:, 2:])[..., None]
    mid = np.concatenate(((corners_mid - c) / f * z_mid, z_mid), axis=-1)
    centers = mid + rots[:, None, :, 2] * params.depth

    # Rayo cámara -> iris y corte frontal con la esfera del ojo.
    iris = pts[:, _IRIS].reshape(-1, 2, 5, 2).mean(axis=2)
    rays = np.concatenate(((iris - c) / f, np.ones(iris.shape[:2] + (1,))), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    b = (rays * centers).sum(axis=-1)
    disc = b * b - (centers * centers).sum(axis=-1) + params.radius**2
    surface = rays * (b - np.sqrt(np.maximum(disc, 0.0)))[..., None]

    # Normal saliente de la esfera = dirección de la mirada; debe apuntar a la cámara.
    gaze = (surface - centers) / params.radius
    valid = (gaze[..., 2] < -min_forward).all(axis=1)
    gaze_z = np.minimum(gaze[..., 2], -1e-6)
    scale = -centers[..., 2] / gaze_z
    return (centers[..., :2] + gaze[..., :2] * scale[..., None]).mean(axis=1), valid


class GeometryEstimator:
    """Convierte landmarks en `Observation`, reutilizando la pose del frame previo."""

    def __init__(self, cfg: Config, width: int, height: int, params: EyeParams | None = None) -> None:
        self._min_forward = cfg.min_gaze_forward
        self.params = params or EyeParams(cfg.eye_radius_mm, cfg.eye_depth_mm, cfg.focal_scale)
        self._K = camera_matrix(width, height, self.params.focal_scale)
        self._face = FACE_MODEL_MM * self.params.face_scale
        self._dist = np.zeros(4)
        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None

    def reset(self) -> None:
        """Descarta la pose previa, p. ej. al perder el rostro."""
        self._rvec = self._tvec = None

    def solve_pose(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Pose de la cabeza; arranca desde la del frame previo si existe."""
        image_pts = pts[: _N_PNP]
        if self._rvec is None:
            ok, rvec, tvec = cv2.solvePnP(
                self._face, image_pts, self._K, self._dist, flags=cv2.SOLVEPNP_SQPNP
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                self._face, image_pts, self._K, self._dist,
                rvec=self._rvec.copy(), tvec=self._tvec.copy(),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not ok or tvec[2, 0] <= 0:
            self.reset()
            return None
        self._rvec, self._tvec = rvec, tvec
        return rvec, tvec

    def observe(self, pts: np.ndarray) -> Observation | None:
        """Procesa landmarks en píxeles, ordenados según `LANDMARK_IDS`."""
        pose = self.solve_pose(pts)
        if pose is None:
            return None
        rvec, tvec = pose
        rot, _ = cv2.Rodrigues(rvec)
        por, valid = gaze_points(
            pts[None], rot[None], tvec.T, self._K, self.params, self._min_forward
        )
        if not valid[0]:
            return None
        pose_vec = pose_features(rot[None], tvec.T, 1.0)[0]
        lids = lid_features(pts)
        return Observation(por[0], pose_vec, lids, float(lids[0]), rvec, tvec)
