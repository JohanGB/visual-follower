"""Integración con el modelo real. Se omite si no está descargado."""

import cv2
import numpy as np
import pytest

from gaze_tracker.capture import LandmarkDetector
from gaze_tracker.config import Config
from gaze_tracker.geometry import GeometryEstimator

skimage_data = pytest.importorskip("skimage.data")
MODEL = Config().model_path
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="modelo no descargado")


@pytest.fixture(scope="module")
def detector():
    det = LandmarkDetector(MODEL)
    yield det
    det.close()


def _frame():
    rgb = cv2.resize(skimage_data.astronaut(), (640, 640))[:480, :640]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def test_real_landmarks_produce_plausible_geometry(detector):
    pts = detector.detect(_frame(), 1.0)
    assert pts is not None and pts.shape == (24, 2)
    obs = GeometryEstimator(Config(), 640, 480).observe(pts)
    assert obs is not None
    assert 300 < obs.tvec[2, 0] < 1500  # distancia en mm
    assert 0.15 < obs.ear < 0.6
    assert np.isfinite(obs.por).all()


def test_no_face_returns_none(detector):
    assert detector.detect(np.full((480, 640, 3), 40, np.uint8), 2.0) is None
