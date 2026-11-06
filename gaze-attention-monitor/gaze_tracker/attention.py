"""Regla de atención: alerta tras N segundos seguidos fuera de pantalla."""

from __future__ import annotations

from enum import Enum


class Gaze(Enum):
    """Estado de la mirada en un instante."""

    ON = "on"
    OFF = "off"
    UNKNOWN = "unknown"  # ojos cerrados: no hay medición válida


class AttentionMonitor:
    """Máquina de estados con antirrebote; recibe el tiempo para ser determinista.

    Un parpadeo no cuenta en ninguna dirección. Si los ojos siguen cerrados más
    allá de `blink_grace`, deja de ser parpadeo y cuenta como fuera de pantalla
    desde que empezó: mirar el teléfono también baja los párpados.
    """

    def __init__(self, limit_s: float, recover_s: float, blink_grace_s: float) -> None:
        self._limit = limit_s
        self._recover = recover_s
        self._grace = blink_grace_s
        self.reset()

    def reset(self) -> None:
        """Vuelve al estado inicial, sin alerta."""
        self.alert = False
        self._off_since: float | None = None
        self._on_since: float | None = None
        self._unknown_since: float | None = None

    def update(self, gaze: Gaze, t: float) -> bool:
        """Incorpora una muestra y devuelve si la alerta está activa."""
        off_start = t
        if gaze is Gaze.UNKNOWN:
            if self._unknown_since is None:
                self._unknown_since = t
            if t - self._unknown_since < self._grace:
                return self.alert
            off_start = self._unknown_since
            gaze = Gaze.OFF
        else:
            self._unknown_since = None

        if gaze is Gaze.ON:
            if self._on_since is None:
                self._on_since = t
            if t - self._on_since >= self._recover:
                self._off_since = None
                self.alert = False
        else:
            self._on_since = None
            if self._off_since is None:
                self._off_since = off_start
            if t - self._off_since >= self._limit:
                self.alert = True
        return self.alert
