"""Pruebas de la interfaz: hilos reales con Qt offscreen y una cámara simulada."""

import json
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QRect, Qt, QThread
from PyQt6.QtGui import QColor, QKeyEvent
from PyQt6.QtWidgets import QApplication

import simulate as sim
from gaze_tracker import worker as worker_mod
from gaze_tracker import ui as ui_mod
from gaze_tracker import calibration
from gaze_tracker.calibration import CalibrationWindow
from gaze_tracker.config import Config
from gaze_tracker.personal import TAG_MOVING, TAG_POINT, TAG_STILL, fit_session
from gaze_tracker.hotkey import GlobalHotkey
from gaze_tracker.model import GazeModel
from gaze_tracker.ui import AlertFrame, GazeDot, GazeApp
from gaze_tracker.worker import GazeWorker, Mode


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def pump(app, seconds, until=None):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.002)
    return until is None


@pytest.fixture(scope="module")
def trained():
    cfg = Config()
    model, *_ = sim.calibrate(sim.Truth(), cfg, np.random.default_rng(4))
    return model


class FakeCamera:
    """Sustituye a CameraStream: entrega un frame negro al ritmo pedido."""

    def __init__(self, cfg):
        self._frame = np.zeros((sim.H, sim.W, 3), np.uint8)

    def read(self, fps):
        time.sleep(1.0 / fps)
        return self._frame, time.perf_counter()

    def close(self):
        pass


class FakeDetector:
    """Entrega landmarks pregrabados; `None` simula ausencia de rostro."""

    source = None

    def __init__(self, path):
        pass

    def detect(self, frame, t):
        return type(self).source(t)

    def close(self):
        pass


@pytest.fixture
def fake_io(monkeypatch):
    monkeypatch.setattr(worker_mod, "CameraStream", FakeCamera)
    monkeypatch.setattr(worker_mod, "LandmarkDetector", FakeDetector)
    monkeypatch.setattr(worker_mod, "ensure_model", lambda p: p)


def looking_at(uv):
    rng = np.random.default_rng(0)
    r, t = sim.head_state(1, rng, sway=False)
    return sim.render(np.array([uv]), r, t, sim.Truth(), rng)[0]


def test_dot_is_click_through_and_small(app):
    dot = GazeDot(QRect(0, 0, 1920, 1080))
    assert dot.width() <= 32 and dot.height() <= 32
    flags = dot.windowFlags()
    assert flags & ui_mod.Qt.WindowType.WindowTransparentForInput
    assert flags & ui_mod.Qt.WindowType.WindowStaysOnTopHint
    dot.show()
    dot.place(1.4, -0.2, True)
    assert dot.x() == 1920 - dot.width() and dot.y() == 0
    assert dot.windowOpacity() == pytest.approx(0.3, abs=0.01)
    dot.close()


def test_alert_frame_paints_edges_only(app):
    frame = AlertFrame(QRect(0, 0, 800, 600))
    image = frame.grab().toImage()
    edge, middle = QColor(image.pixel(2, 300)), QColor(image.pixel(400, 300))
    assert edge.red() > 150 and edge.green() < 90
    assert middle.red() < 60
    frame.close()


def test_worker_emits_from_background_thread_and_ui_stays_responsive(app, fake_io, trained):
    FakeDetector.source = staticmethod(lambda t: looking_at((0.4, 0.5)))
    cfg = Config(fps_active=30)
    worker = GazeWorker(cfg)
    threads, points = set(), []

    def on_gaze(x, y, ok):
        threads.add(QThread.currentThread())
        points.append((x, y, ok))

    worker.gaze_moved.connect(on_gaze)
    worker.start()
    assert pump(app, 3, lambda: worker.pipeline is not None)
    worker.set_model(trained)
    worker.set_mode(Mode.TRACK)

    gaps, last = [], time.perf_counter()
    while time.perf_counter() - last < 1.0 and len(gaps) < 2000:
        app.processEvents()
        now = time.perf_counter()
        gaps.append(now)
        time.sleep(0.001)
    worker.stop()
    assert not worker.isRunning()
    assert len(points) > 10
    assert threads == {app.thread()}  # los slots corren en el hilo principal
    assert max(np.diff(gaps)) < 0.1
    x, y, ok = points[-1]
    assert ok and abs(x - 0.4) < 0.12 and abs(y - 0.5) < 0.12


def test_alert_signal_fires_once_after_three_seconds(app, fake_io, trained):
    start = [None]

    def source(t):
        start[0] = start[0] or t
        return None if t - start[0] > 0.5 else looking_at((0.5, 0.5))

    FakeDetector.source = staticmethod(source)
    worker = GazeWorker(Config(fps_active=30, fps_idle=30))
    events = []
    worker.alert_changed.connect(lambda a: events.append((a, time.perf_counter())))
    worker.start()
    assert pump(app, 3, lambda: worker.pipeline is not None)
    worker.set_model(trained)
    worker.set_mode(Mode.TRACK)
    t0 = time.perf_counter()
    pump(app, 5, lambda: bool(events))
    worker.stop()
    assert events and events[0][0] is True
    assert 3.0 <= events[0][1] - t0 <= 4.2
    assert len(events) == 1


def test_idle_rate_drops_without_face(app, fake_io):
    FakeDetector.source = staticmethod(lambda t: None)
    calls, base = [], FakeCamera.read

    def spy(self, fps):
        calls.append(fps)
        return base(self, min(fps, 200))

    FakeCamera.read = spy
    try:
        worker = GazeWorker(Config(idle_after_s=0.1, fps_active=15, fps_idle=5))
        worker.start()
        assert pump(app, 2, lambda: len(calls) > 4)
        worker.set_mode(Mode.TRACK)
        pump(app, 0.5)
        worker.stop()
    finally:
        FakeCamera.read = base
    assert calls[-1] == 5


def short_config(**kw):
    return Config(still_s=0.2, moving_s=0.2, point_s=0.03, point_settle_s=0.01, min_samples=5, **kw)


def run_window(app, window, on_sample=None, seconds=8):
    """Conduce la secuencia sin esperar la cuenta atrás ni las pausas."""
    results = []
    window.completed.connect(results.append)
    window.insufficient.connect(results.append)
    window.show()
    window.on_face(True)
    window._countdown = 1
    window._count()
    end = time.perf_counter() + seconds
    while not results and time.perf_counter() < end:
        app.processEvents()
        if on_sample is not None:
            on_sample(window)
        time.sleep(0.002)
    return results


@pytest.fixture
def fast_hints(monkeypatch):
    monkeypatch.setattr(calibration, "_HINT_S", 0.02)


def test_calibration_sequence_tags_every_stage(app, fast_hints):
    window = CalibrationWindow(short_config(), seed=1)
    window.setGeometry(0, 0, 640, 360)
    pts = np.ones((24, 2))
    results = run_window(app, window, lambda w: w.on_observed(time.perf_counter(), pts))
    assert isinstance(results[0], dict)
    tags = results[0]["tag"]
    assert TAG_STILL in tags and TAG_MOVING in tags
    assert (tags >= TAG_POINT).any()
    assert results[0]["pts"].shape[1:] == (24, 2)
    assert results[0]["t_tgt"].size == results[0]["xy_tgt"].shape[0] > 0
    window.close()


def test_validate_only_skips_training(app, fast_hints):
    window = CalibrationWindow(short_config(), seed=1, validate_only=True)
    pts = np.ones((24, 2))
    results = run_window(app, window, lambda w: w.on_observed(time.perf_counter(), pts))
    assert (results[0]["tag"] >= TAG_POINT).all()
    assert results[0]["t_tgt"].size == 0
    window.close()


def test_calibration_reports_too_few_samples(app, fast_hints):
    window = CalibrationWindow(replace(short_config(), min_samples=500), seed=2)
    results = run_window(app, window)
    assert results == [0]
    window.close()


def test_escape_is_ignored_while_fitting(app):
    window = CalibrationWindow(Config(), seed=3)
    window._phase = "fitting"
    seen = []
    window.cancelled.connect(lambda: seen.append(1))
    window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert not seen
    window.close()


def test_results_screen_decisions(app):
    window = CalibrationWindow(Config(), seed=3)
    window.resize(1280, 720)
    tr, cfg = sim.Truth(), Config()
    _, report = fit_session(sim.session(tr, cfg, np.random.default_rng(1)), sim.SCREEN_PX, cfg, (640, 480))
    window.show_report(report)
    window.grab()
    seen = []
    window.accepted.connect(lambda: seen.append("ok"))
    window.retry.connect(lambda: seen.append("retry"))
    for key in (Qt.Key.Key_R, Qt.Key.Key_Return):
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))
    assert seen == ["retry", "ok"]
    window.show_report(None)
    window.grab()
    window.close()


def test_full_calibration_flow_saves_and_tracks(app, fake_io, tmp_path, monkeypatch):
    """Recorre la app con muestras simuladas: ajustar, mostrar, aceptar, seguir."""
    FakeDetector.source = staticmethod(lambda t: looking_at((0.5, 0.5)))
    cfg = Config(calibration_path=tmp_path / "c.npz", history_path=tmp_path / "h.jsonl",
                 session_path=tmp_path / "s.npz", output_dir=tmp_path, fps_active=30)
    gaze = GazeApp(app, cfg)
    monkeypatch.setattr(gaze, "_screen_px", sim.SCREEN_PX)
    gaze.start()
    assert pump(app, 3, lambda: gaze._calibration is not None)
    window = gaze._calibration
    window._count_timer.stop()
    data = sim.session(sim.Truth(), cfg, np.random.default_rng(1))
    window.completed.emit(data)
    assert pump(app, 8, lambda: window._phase == "results" or window._report is not None)
    assert cfg.calibration_path.exists()
    entry = json.loads(cfg.history_path.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["chosen"] == "personal"
    window.accepted.emit()
    assert pump(app, 2, lambda: gaze._dot.isVisible())
    assert gaze._model.params.depth > 5.2
    assert cfg.session_path.exists()
    gaze.shutdown()
    gaze._dot.close()
    gaze._frame.close()


def test_calibration_waits_for_face(app):
    window = CalibrationWindow(Config(), seed=3)
    window.on_face(True)
    window.on_face(False)
    assert window._phase == "waiting"
    window.close()


def test_full_app_saves_heatmap_on_exit(app, fake_io, trained, tmp_path, monkeypatch):
    FakeDetector.source = staticmethod(lambda t: looking_at((0.3, 0.3)))
    screen = app.primaryScreen()
    ratio = screen.devicePixelRatio()
    size = (round(screen.geometry().width() * ratio), round(screen.geometry().height() * ratio))
    model = GazeModel(trained.mean, trained.std, trained.coef, trained.intercept,
                      trained.lag_s, trained.error, trained.ear_ref, size)
    cal = tmp_path / "cal.npz"
    model.save(cal)
    cfg = Config(calibration_path=cal, output_dir=tmp_path / "maps", fps_active=30)
    gaze = GazeApp(app, cfg)
    gaze.start()
    assert pump(app, 3, lambda: gaze._dot.isVisible())
    assert pump(app, 3, lambda: gaze._worker.pipeline.heatmap is not None
                and gaze._worker.pipeline.heatmap.fixation_s > 0.3)

    visible = gaze._dot.isVisible()
    gaze._toggle_requested.emit()
    assert pump(app, 1, lambda: gaze._dot.isVisible() != visible)
    gaze._toggle_requested.emit()

    path = gaze.shutdown()
    assert path is not None and path.exists()
    assert gaze.shutdown() is None
    assert not gaze._worker.isRunning()
    gaze._dot.close()
    gaze._frame.close()


def test_calibration_does_not_start_without_face(app, fake_io, tmp_path):
    FakeDetector.source = staticmethod(lambda t: None)
    cfg = Config(calibration_path=tmp_path / "none.npz", output_dir=tmp_path, fps_active=30)
    gaze = GazeApp(app, cfg)
    gaze.start()
    assert pump(app, 3, lambda: gaze._calibration is not None)
    pump(app, 0.3)
    assert gaze._calibration._phase == "waiting"
    FakeDetector.source = staticmethod(lambda t: looking_at((0.5, 0.5)))
    assert pump(app, 2, lambda: gaze._calibration._phase == "countdown")
    gaze._close_calibration()
    gaze.shutdown()
    gaze._dot.close()
    gaze._frame.close()


def test_hotkey_callback_from_foreign_thread_reaches_ui(app, fake_io, tmp_path):
    FakeDetector.source = staticmethod(lambda t: None)
    gaze = GazeApp(app, Config(calibration_path=tmp_path / "n.npz", output_dir=tmp_path))
    state = gaze._dot_enabled
    thread = threading.Thread(target=gaze._toggle_requested.emit)
    thread.start()
    thread.join()
    assert gaze._dot_enabled == state  # todavía no: va encolado al hilo de la UI
    assert pump(app, 1, lambda: gaze._dot_enabled != state)
    gaze._dot.close()
    gaze._frame.close()


def test_camera_failure_is_reported(app, monkeypatch):
    def broken(cfg):
        raise RuntimeError("sin cámara")

    monkeypatch.setattr(worker_mod, "CameraStream", broken)
    monkeypatch.setattr(worker_mod, "LandmarkDetector", FakeDetector)
    monkeypatch.setattr(worker_mod, "ensure_model", lambda p: p)
    worker = GazeWorker(Config())
    errors = []
    worker.failed.connect(errors.append)
    worker.start()
    assert pump(app, 2, lambda: bool(errors))
    worker.wait(1000)
    assert "sin cámara" in errors[0]


def test_win32_hotkey_loop(monkeypatch):
    """Ejercita el bucle Win32 con un user32 falso (no sustituye probar en Windows)."""
    import ctypes
    import ctypes.wintypes as wt
    import sys
    import types

    hits, unregistered = [], []

    class User32:
        def __init__(self):
            self.queue = [0x0312, 0x0100, 0x0312]

        def RegisterHotKey(self, hwnd, ident, mods, vk):
            assert mods == 0x0002 | 0x0004 | 0x4000 and vk == 0x48
            return 1

        def UnregisterHotKey(self, hwnd, ident):
            unregistered.append(ident)

        def GetMessageW(self, ref, *_):
            if not self.queue:
                return 0
            msg = ref._obj
            msg.message = self.queue.pop(0)
            msg.wParam = 0x4748
            return 1

        def PostThreadMessageW(self, *_):
            return 1

    fake = types.SimpleNamespace(
        user32=User32(), kernel32=types.SimpleNamespace(GetCurrentThreadId=lambda: 7)
    )
    monkeypatch.setattr(ctypes, "windll", fake, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    hotkey = GlobalHotkey(lambda: hits.append(1))
    assert hotkey.start()
    hotkey._thread.join(1)
    assert len(hits) == 2 and unregistered == [0x4748]


def test_win32_hotkey_conflict_is_reported(monkeypatch):
    import ctypes
    import sys
    import types

    fake = types.SimpleNamespace(
        user32=types.SimpleNamespace(RegisterHotKey=lambda *a: 0),
        kernel32=types.SimpleNamespace(GetCurrentThreadId=lambda: 7),
    )
    monkeypatch.setattr(ctypes, "windll", fake, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    hotkey = GlobalHotkey(lambda: None)
    assert hotkey.start() is False
    assert "en uso" in hotkey.error
