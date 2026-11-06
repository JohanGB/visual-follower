"""Ventana de calibración y validación: sin clics, solo seguir lo que aparece.

Secuencia completa: seguir el círculo con la cabeza quieta, seguirlo moviendo la
cabeza, mirar puntos fijos y mirar más puntos desde otra posición. Los puntos
fijos no entran al ajuste: sirven para medir el error real.
"""

from __future__ import annotations

import time

import numpy as np
from PyQt6.QtCore import QPointF, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QKeyEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QWidget

from .config import Config
from .personal import TAG_MOVING, TAG_POINT, TAG_STILL
from .trajectory import generate_trajectory
from .validation import Report, validation_targets

_BG = QColor("#111214")
_RING = QColor("#E8E6E3")
_MUTED = QColor("#A09C98")
_DOT = QColor("#F59E0B")
_ERROR = QColor("#F87171")
_BASE_RADIUS = 22.0
_POINT_RADIUS = 11.0
_RATE = 60.0
_HINT_S = 3.5

_MOVE_HINT = "Ahora sigue el círculo moviendo la cabeza despacio:\ngira, inclina, acércate y aléjate"
_POINTS_HINT = "Vuelve a tu postura normal\ny mira fijamente cada punto"
_SHIFT_HINT = "Cambia de postura: muévete un poco\na un lado y quédate ahí"


class CalibrationWindow(QWidget):
    """Conduce la secuencia y reúne muestras; el ajuste ocurre fuera de aquí."""

    completed = pyqtSignal(dict)
    cancelled = pyqtSignal()
    insufficient = pyqtSignal(int)
    accepted = pyqtSignal()
    retry = pyqtSignal()

    def __init__(self, cfg: Config, seed: int | None = None, validate_only: bool = False) -> None:
        super().__init__()
        self._cfg = cfg
        self._validate_only = validate_only
        self._targets, groups = validation_targets()
        self._steps: list[tuple] = []
        if not validate_only:
            self._steps += [
                ("pursuit", generate_trajectory(cfg.still_s, _RATE, seed), TAG_STILL),
                ("hint", _MOVE_HINT),
                ("pursuit", generate_trajectory(cfg.moving_s, _RATE, _next_seed(seed)), TAG_MOVING),
            ]
        self._steps += [
            ("hint", _POINTS_HINT),
            ("points", np.flatnonzero(groups == 0)),
            ("hint", _SHIFT_HINT),
            ("points", np.flatnonzero(groups == 1)),
        ]

        self._phase = "waiting"
        self._countdown = 3
        self._step: tuple | None = None
        self._step_start = 0.0
        self._tag = -1
        self._index = 0
        self._point = -1
        self._dirty = QRect()
        self._report: Report | None = None

        self._t: list[float] = []
        self._pts: list[np.ndarray] = []
        self._tags: list[int] = []
        self._log_t: list[float] = []
        self._log_xy: list[tuple[float, float]] = []

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._count_timer = QTimer(self)
        self._count_timer.setInterval(1000)
        self._count_timer.timeout.connect(self._count)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def on_face(self, visible: bool) -> None:
        """Arranca la cuenta atrás cuando hay rostro; la cancela si se pierde antes."""
        if self._phase == "waiting" and visible:
            self._phase = "countdown"
            self._countdown = 3
            self._count_timer.start()
        elif self._phase == "countdown" and not visible:
            self._phase = "waiting"
            self._count_timer.stop()
        self.update()

    def on_observed(self, t: float, pts: np.ndarray) -> None:
        """Guarda los landmarks crudos: el ajuste recalcula la geometría con ellos."""
        if self._tag >= 0:
            self._t.append(t)
            self._pts.append(pts)
            self._tags.append(self._tag)

    def _count(self) -> None:
        self._countdown -= 1
        if self._countdown <= 0:
            self._count_timer.stop()
            self._phase = "running"
            self._advance()
            self._timer.start()
        self.update()

    def _advance(self) -> None:
        """Pasa al siguiente paso de la secuencia."""
        self._tag = -1
        if not self._steps:
            self._step = None
            self._timer.stop()
            self._finish()
            return
        self._step = self._steps.pop(0)
        self._step_start = time.perf_counter()
        self._index = 0
        self._point = -1
        self._dirty = QRect()
        if self._step[0] == "pursuit":
            self._tag = self._step[2]
        self.update()

    def _tick(self) -> None:
        kind = self._step[0]
        now = time.perf_counter()
        elapsed = now - self._step_start
        if kind == "hint":
            if elapsed >= _HINT_S:
                self._advance()
        elif kind == "pursuit":
            self._tick_pursuit(now, elapsed)
        else:
            self._tick_points(elapsed)

    def _tick_pursuit(self, now: float, elapsed: float) -> None:
        trajectory = self._step[1]
        index = int(elapsed * _RATE)
        if index >= trajectory.t.size:
            self._advance()
            return
        self._index = index
        x, y = trajectory.xy[index]
        self._log_t.append(now)
        self._log_xy.append((float(x), float(y)))
        # Se repinta solo el área del círculo, no toda la pantalla.
        r = int(_BASE_RADIUS * trajectory.radius[index]) + 4
        rect = QRect(int(x * self.width()) - r, int(y * self.height()) - r, 2 * r, 2 * r)
        self.update(self._dirty.united(rect))
        self._dirty = rect

    def _tick_points(self, elapsed: float) -> None:
        ids = self._step[1]
        slot = int(elapsed / self._cfg.point_s)
        if slot >= ids.size:
            self._advance()
            return
        if slot != self._point:
            self._point = slot
            self._tag = -1
            self.update()
        # Se ignora el inicio de cada punto: el ojo aún viaja hacia él.
        if elapsed - slot * self._cfg.point_s >= self._cfg.point_settle_s:
            self._tag = TAG_POINT + int(ids[slot])

    def _finish(self) -> None:
        self._phase = "fitting"
        self.update()
        training = sum(tag < TAG_POINT for tag in self._tags)
        if not self._validate_only and training < self._cfg.min_samples:
            self.insufficient.emit(training)
            return
        self.completed.emit(
            {
                "t": np.array(self._t),
                "pts": np.stack(self._pts) if self._pts else np.zeros((0, 24, 2)),
                "tag": np.array(self._tags, np.int64),
                "t_tgt": np.array(self._log_t),
                "xy_tgt": np.array(self._log_xy).reshape(-1, 2),
            }
        )

    def show_report(self, report: Report | None) -> None:
        """Muestra el resultado de la validación y espera una decisión."""
        self._report = report
        self._phase = "results"
        self.unsetCursor()
        self.setFocus()
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Esc cancela; en resultados, Enter continúa y R repite."""
        key = event.key()
        if self._phase == "results":
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
                self.accepted.emit()
            elif key == Qt.Key.Key_R:
                self.retry.emit()
        elif key == Qt.Key.Key_Escape and self._phase != "fitting":
            self._timer.stop()
            self._count_timer.stop()
            self.cancelled.emit()
        else:
            super().keyPressEvent(event)

    def _paint_text(self, painter: QPainter, text: str, size: int = 22) -> None:
        font = QFont()
        font.setPointSize(size)
        painter.setFont(font)
        painter.setPen(_RING)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)

    def _paint_pursuit(self, painter: QPainter) -> None:
        trajectory = self._step[1]
        if self._step[2] == TAG_MOVING:
            font = QFont()
            font.setPointSize(12)
            painter.setFont(font)
            painter.setPen(_MUTED)
            painter.drawText(
                QRect(0, 18, self.width(), 30), Qt.AlignmentFlag.AlignCenter,
                "Mueve la cabeza despacio sin dejar de seguir el círculo",
            )
        x, y = trajectory.xy[self._index]
        center = QPointF(x * self.width(), y * self.height())
        radius = _BASE_RADIUS * float(trajectory.radius[self._index])
        painter.setPen(_RING)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, radius, radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_DOT)
        painter.drawEllipse(center, 4.0, 4.0)

    def _paint_point(self, painter: QPainter) -> None:
        if self._point < 0:
            return
        x, y = self._targets[self._step[1][self._point]]
        center = QPointF(x * self.width(), y * self.height())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_DOT)
        painter.drawEllipse(center, _POINT_RADIUS, _POINT_RADIUS)
        painter.setBrush(_BG)
        painter.drawEllipse(center, 3.0, 3.0)

    def _paint_results(self, painter: QPainter) -> None:
        report = self._report
        if report is None:
            self._paint_text(
                painter,
                "No hubo suficientes puntos para medir el error.\n\n"
                "Enter para continuar · R para repetir", 18,
            )
            return
        w, h = self.width(), self.height()
        for target, estimate in zip(report.targets, report.current.estimates):
            tx, ty = target[0] * w, target[1] * h
            if estimate[0] is not None:
                painter.setPen(QPen(_ERROR, 2))
                painter.drawLine(QPointF(tx, ty), QPointF(estimate[0] * w, estimate[1] * h))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_DOT if estimate[0] is not None else _MUTED)
            painter.drawEllipse(QPointF(tx, ty), 5.0, 5.0)
        text = report_text(report)
        font = QFont()
        font.setPointSize(16)
        box = QFontMetrics(font).boundingRect(
            self.rect(), int(Qt.AlignmentFlag.AlignCenter), text
        ).adjusted(-28, -22, 28, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_BG)
        painter.drawRoundedRect(box, 10, 10)
        self._paint_text(painter, text, 16)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Dibuja según la fase actual."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(event.rect(), _BG)
        if self._phase == "waiting":
            self._paint_text(painter, "Colócate frente a la cámara")
        elif self._phase == "countdown":
            first = "Mira los puntos" if self._validate_only else "Sigue el círculo con la mirada, sin mover la cabeza"
            self._paint_text(painter, f"{first}\n{self._countdown}")
        elif self._phase == "fitting":
            self._paint_text(painter, "Ajustando el modelo…")
        elif self._phase == "results":
            self._paint_results(painter)
        elif self._step[0] == "hint":
            self._paint_text(painter, self._step[1])
        elif self._step[0] == "pursuit":
            self._paint_pursuit(painter)
        else:
            self._paint_point(painter)


def _next_seed(seed: int | None) -> int | None:
    return None if seed is None else seed + 1


def report_text(report: Report) -> str:
    """Resumen legible de un reporte."""
    score = report.current
    degrees = "" if score.accuracy_deg is None else f" (≈{score.accuracy_deg:.1f}°)"
    lines = [
        f"Exactitud: {score.accuracy_px:.0f} px{degrees}",
        f"Precisión: {score.precision_px:.0f} px de temblor, antes del filtro",
    ]
    if score.vertical_r2 is not None:
        low = "" if score.low_zone_px is None else f", zona baja {score.low_zone_px:.0f} px"
        lines.append(f"Vertical: R² {score.vertical_r2:.2f}, pendiente {score.vertical_slope:.2f}{low}")
    base = report.baseline
    rows = (("Postura normal", "natural_px"), ("Cabeza movida", "moved_px"))
    for label, field in rows:
        value = getattr(score, field)
        if value is None:
            continue
        before = "" if base is None else f"{getattr(base, field):.0f} → "
        lines.append(f"{label}: {before}{value:.0f} px")
    if report.chosen != "actual":
        layers = []
        if report.lids_active:
            layers.append("párpados")
        if report.correction_active:
            layers.append("corrector kernel")
        lines.append("Capas activas: " + (", ".join(layers) if layers else "ninguna adicional"))
    if report.screen_mm is not None:
        inches = float(np.hypot(*report.screen_mm)) / 25.4
        place = f"Monitor {inches:.1f}\" ({report.screen_source})"
        if report.distance_mm is not None:
            place += f" · distancia ≈ {report.distance_mm / 10:.0f} cm"
        lines.append(place)
    lines += ["", report.reason, "", "Enter para continuar · R para repetir"]
    return "\n".join(lines)
