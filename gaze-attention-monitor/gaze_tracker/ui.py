"""Overlays del sistema y orquestación. Todo aquí corre en el hilo principal."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, QPoint, QPointF, QRect, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import (QAction, QColor, QDesktopServices, QIcon, QLinearGradient,
                         QPainter, QPaintEvent, QPixmap)
from PyQt6.QtWidgets import (QApplication, QMenu, QMessageBox, QSystemTrayIcon, QWidget)

from .calibration import CalibrationWindow
from .config import Config
from .display import detect_screen
from .hotkey import GlobalHotkey
from .model import GazeModel
from .validation import Report, append_history
from .worker import FitWorker, GazeWorker, Mode

log = logging.getLogger(__name__)

_OVERLAY_FLAGS = (
    Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowStaysOnTopHint
    | Qt.WindowType.Tool
    | Qt.WindowType.WindowTransparentForInput
    | Qt.WindowType.WindowDoesNotAcceptFocus
)
_DOT_SIZE = 26
_EDGE = 44


def _configure_overlay(widget: QWidget) -> None:
    """Ventana transparente que nunca roba foco ni clics."""
    widget.setWindowFlags(_OVERLAY_FLAGS)
    widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)


class GazeDot(QWidget):
    """Indicador de mirada: se pinta una vez y luego solo se mueve la ventana."""

    def __init__(self, area: QRect) -> None:
        super().__init__()
        _configure_overlay(self)
        self._area = area
        self.setFixedSize(_DOT_SIZE, _DOT_SIZE)

    def place(self, x: float, y: float, valid: bool) -> None:
        """Mueve el punto, ajustado al borde si la mirada sale de la pantalla."""
        area = self._area
        px = area.x() + round(min(max(x, 0.0), 1.0) * (area.width() - _DOT_SIZE))
        py = area.y() + round(min(max(y, 0.0), 1.0) * (area.height() - _DOT_SIZE))
        target = QPoint(px, py)
        # Ignorar saltos de 1-2 px evita recomponer el escritorio sin necesidad.
        if (target - self.pos()).manhattanLength() > 2:
            self.move(target)
        inside = valid and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        opacity = 0.85 if inside else 0.3
        if self.windowOpacity() != opacity:
            self.setWindowOpacity(opacity)

    def paintEvent(self, _: QPaintEvent) -> None:
        """Anillo fino con centro translúcido."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(_DOT_SIZE / 2, _DOT_SIZE / 2)
        painter.setPen(QColor(255, 255, 255, 230))
        painter.setBrush(QColor(245, 158, 11, 170))
        painter.drawEllipse(center, _DOT_SIZE / 2 - 3, _DOT_SIZE / 2 - 3)


class AlertFrame(QWidget):
    """Marco rojo translúcido. Estático: sin animación no consume CPU."""

    def __init__(self, area: QRect) -> None:
        super().__init__()
        _configure_overlay(self)
        self.setGeometry(area)

    def paintEvent(self, _: QPaintEvent) -> None:
        """Degradado hacia adentro en los cuatro bordes."""
        painter = QPainter(self)
        w, h = self.width(), self.height()
        strong, clear = QColor(220, 38, 38, 190), QColor(220, 38, 38, 0)
        edges = (
            (QRect(0, 0, w, _EDGE), QPointF(0, 0), QPointF(0, _EDGE)),
            (QRect(0, h - _EDGE, w, _EDGE), QPointF(0, h), QPointF(0, h - _EDGE)),
            (QRect(0, 0, _EDGE, h), QPointF(0, 0), QPointF(_EDGE, 0)),
            (QRect(w - _EDGE, 0, _EDGE, h), QPointF(w, 0), QPointF(w - _EDGE, 0)),
        )
        for rect, start, stop in edges:
            gradient = QLinearGradient(start, stop)
            gradient.setColorAt(0.0, strong)
            gradient.setColorAt(1.0, clear)
            painter.fillRect(rect, gradient)


def _tray_icon() -> QIcon:
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor("#E8E6E3"))
    painter.setBrush(QColor("#111214"))
    painter.drawEllipse(2, 6, 28, 20)
    painter.setBrush(QColor("#F59E0B"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(10, 10, 12, 12)
    painter.end()
    return QIcon(pixmap)


class GazeApp(QObject):
    """Conecta worker, calibración, overlays y salida sin bloquear la UI."""

    _toggle_requested = pyqtSignal()

    def __init__(
        self, app: QApplication, cfg: Config, recalibrate: bool = False, validate: bool = False
    ) -> None:
        super().__init__()
        self._app = app
        self._cfg = cfg
        self._force_calibration = recalibrate
        self._validate_on_start = validate
        screen = app.primaryScreen()
        self._area = screen.geometry()
        ratio = screen.devicePixelRatio()
        self._screen_px = (round(self._area.width() * ratio), round(self._area.height() * ratio))
        self._screen_info = detect_screen(self._screen_px)

        self._worker = GazeWorker(cfg)
        self._dot = GazeDot(self._area)
        self._frame = AlertFrame(self._area)
        self._calibration: CalibrationWindow | None = None
        self._model: GazeModel | None = None
        self._next_model: GazeModel | None = None
        self._fit: FitWorker | None = None
        self._dot_enabled = True
        self._face_visible = False
        self._finished = False
        self._heatmap_blur = 0.03

        self._hotkey = GlobalHotkey(self._toggle_requested.emit)
        self._tray = self._build_tray()

        self._toggle_requested.connect(self._toggle_dot)
        self._worker.face_changed.connect(self._on_face)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.gaze_moved.connect(self._on_gaze)
        self._worker.alert_changed.connect(self._on_alert)
        app.aboutToQuit.connect(self.shutdown)

    def start(self) -> None:
        """Arranca hilos y atajo."""
        if not self._hotkey.start():
            log.warning(self._hotkey.error)
            self._notify(self._hotkey.error or "Atajo global no disponible.")
        self._worker.start(QThread.Priority.LowPriority)

    def _build_tray(self) -> QSystemTrayIcon:
        tray = QSystemTrayIcon(_tray_icon(), self)
        tray.setToolTip("Gaze Tracker")
        menu = QMenu()
        actions = (
            ("Mostrar/ocultar indicador (Ctrl+Shift+H)", self._toggle_dot),
            ("Recalibrar", lambda: self._begin_calibration()),
            ("Medir precisión", lambda: self._begin_calibration(self._model is not None)),
            ("Abrir carpeta de mapas", self._open_output),
            ("Salir", self._app.quit),
        )
        for text, slot in actions:
            action = QAction(text, menu)
            action.triggered.connect(slot)
            menu.addAction(action)
        tray.setContextMenu(menu)
        self._menu = menu  # Qt no toma posesión del menú
        tray.show()
        return tray

    def _notify(self, text: str) -> None:
        if QSystemTrayIcon.isSystemTrayAvailable():
            self._tray.showMessage("Gaze Tracker", text, QSystemTrayIcon.MessageIcon.Information, 4000)
        log.info(text)

    def _on_ready(self) -> None:
        model = None if self._force_calibration else GazeModel.load(self._cfg.calibration_path)
        if model is None or model.screen != self._screen_px:
            self._begin_calibration()
        elif self._validate_on_start:
            self._model = model
            self._worker.set_model(model)
            self._begin_calibration(validate_only=True)
        else:
            self._start_tracking(model)

    def _start_tracking(self, model: GazeModel) -> None:
        self._model = model
        self._heatmap_blur = model.error
        self._worker.set_model(model)
        self._worker.set_mode(Mode.TRACK)
        if self._dot_enabled:
            self._dot.show()

    def _begin_calibration(self, validate_only: bool = False) -> None:
        if self._calibration is not None:
            return
        self._worker.set_mode(Mode.PAUSED)
        self._dot.hide()
        self._frame.hide()
        window = CalibrationWindow(self._cfg, validate_only=validate_only)
        window.completed.connect(lambda data: self._on_samples(data, validate_only))
        window.cancelled.connect(self._on_cancelled)
        window.insufficient.connect(self._on_insufficient)
        window.accepted.connect(self._on_accepted)
        window.retry.connect(lambda: self._on_retry(validate_only))
        self._worker.observed.connect(window.on_observed)
        self._worker.face_changed.connect(window.on_face)
        self._calibration = window
        window.setGeometry(self._area)
        window.showFullScreen()
        window.activateWindow()
        window.on_face(self._face_visible)
        self._worker.set_mode(Mode.CALIBRATE)

    def _close_calibration(self) -> None:
        window, self._calibration = self._calibration, None
        if window is not None:
            self._worker.observed.disconnect(window.on_observed)
            self._worker.face_changed.disconnect(window.on_face)
            window.close()
            window.deleteLater()

    def _on_samples(self, data: dict, validate_only: bool) -> None:
        self._worker.set_mode(Mode.PAUSED)
        self._fit = FitWorker(
            data, self._screen_px, self._cfg, self._worker.frame_size, self._screen_info,
            self._model if validate_only else None,
        )
        self._fit.done.connect(self._on_fitted)
        self._fit.failed.connect(self._on_fit_failed)
        self._fit.start()

    def _on_fitted(self, model: GazeModel, report: Report | None) -> None:
        self._next_model = model
        model.save(self._cfg.calibration_path)
        if report is not None:
            append_history(report, self._cfg.history_path)
        if self._calibration is not None:
            self._calibration.show_report(report)

    def _on_accepted(self) -> None:
        self._close_calibration()
        model, self._next_model = self._next_model, None
        if model is not None:
            self._start_tracking(model)

    def _on_retry(self, validate_only: bool) -> None:
        self._close_calibration()
        self._model, self._next_model = self._next_model, None
        self._begin_calibration(validate_only)

    def _on_cancelled(self) -> None:
        self._close_calibration()
        if self._model is None:
            self._app.quit()
        else:
            self._start_tracking(self._model)

    def _on_face(self, visible: bool) -> None:
        self._face_visible = visible

    def _on_insufficient(self, count: int) -> None:
        self._on_fit_failed(f"Solo se capturaron {count} muestras válidas.")

    def _on_fit_failed(self, reason: str) -> None:
        self._close_calibration()
        answer = QMessageBox.question(
            None, "Gaze Tracker", f"{reason}\n¿Repetir la calibración?"
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._begin_calibration()
        else:
            self._on_cancelled()

    def _on_failed(self, message: str) -> None:
        log.error(message)
        QMessageBox.critical(None, "Gaze Tracker", message)
        self._app.quit()

    def _on_gaze(self, x: float, y: float, valid: bool) -> None:
        if self._dot.isVisible():
            self._dot.place(x, y, valid)

    def _on_alert(self, active: bool) -> None:
        self._frame.setVisible(active)

    def _toggle_dot(self) -> None:
        self._dot_enabled = not self._dot_enabled
        self._dot.setVisible(self._dot_enabled and self._calibration is None)

    def _open_output(self) -> None:
        self._cfg.output_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._cfg.output_dir)))

    def shutdown(self) -> Path | None:
        """Detiene los hilos y guarda el mapa de calor. Es idempotente."""
        if self._finished:
            return None
        self._finished = True
        self._hotkey.stop()
        self._worker.stop()
        if self._fit is not None:
            self._fit.wait(3000)
        self._tray.hide()
        pipeline = self._worker.pipeline
        if pipeline is None or pipeline.heatmap is None:
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = pipeline.heatmap.render(self._cfg.output_dir / f"heatmap_{stamp}.png", self._heatmap_blur)
        log.info("Mapa de calor: %s", path)
        return path
