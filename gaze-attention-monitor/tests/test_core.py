"""Pruebas de geometría, modelo, trayectoria, atención y mapa de calor."""

import json

import numpy as np
import pytest

import simulate as sim
from gaze_tracker.attention import AttentionMonitor, Gaze
from gaze_tracker import display
from gaze_tracker.calibration import report_text
from gaze_tracker.config import Config
from gaze_tracker.display import ScreenInfo, fit_orientation, parse_edid, registry_key
from gaze_tracker.geometry import (EyeParams, FACE_MODEL_MM, GeometryEstimator, eye_aspect_ratio,
                                   lid_features)
from gaze_tracker.heatmap import FixationHeatmap
from gaze_tracker.model import (GazeModel, OneEuroFilter, ResidualKernel, blink_mask, design_matrix,
                                estimate_lag, fit_gaze_model, fit_residual, poly_features)
from gaze_tracker.personal import TAG_POINT, assess_model, fit_session, save_session
from gaze_tracker.pipeline import GazePipeline
from gaze_tracker.validation import append_history, balance, score_points, validation_targets
from gaze_tracker.trajectory import generate_trajectory


@pytest.fixture(scope="module")
def calibrated():
    tr, cfg = sim.Truth(), Config()
    rng = np.random.default_rng(11)
    model, naive, lag = sim.calibrate(tr, cfg, rng)
    return tr, cfg, model, naive, lag


def test_face_model_uses_opencv_axes():
    assert FACE_MODEL_MM[7, 1] < 0  # frente arriba: y negativa
    assert FACE_MODEL_MM[6, 2] < FACE_MODEL_MM[8, 2]  # nariz más cerca de la cámara
    assert np.linalg.norm(FACE_MODEL_MM[0] - FACE_MODEL_MM[3]) == pytest.approx(88.9, abs=0.1)


def test_pnp_recovers_pose():
    rng = np.random.default_rng(0)
    r = np.radians([[-5.0, 18.0, 4.0]])
    t = np.array([[30.0, 60.0, 600.0]])
    tr = sim.Truth(face_scale=1.0, noise_px=0.0, focal=0.95 * sim.W)
    px = sim.render(np.array([[0.5, 0.5]]), r, t, tr, rng)
    obs = GeometryEstimator(Config(), sim.W, sim.H).observe(px[0])
    assert np.allclose(obs.rvec.ravel(), r[0], atol=1e-3)
    assert np.allclose(obs.tvec.ravel(), t[0], atol=0.5)


def test_lag_is_recovered(calibrated):
    _, _, model, _, lag = calibrated
    assert model.lag_s == pytest.approx(lag, abs=0.02)


def test_lag_search_on_plain_signals():
    t = np.linspace(0, 40, 800)
    xy = np.column_stack((np.sin(t * 0.7) * 0.5 + 0.5, np.cos(t * 1.1) * 0.5 + 0.5))
    tq = np.arange(1, 39, 0.05)
    delayed = np.column_stack([np.interp(tq - 0.24, t, xy[:, i]) for i in (0, 1)])
    assert estimate_lag(tq, delayed * 300, t, xy, 0.5) == pytest.approx(0.24, abs=0.02)


def test_geometry_beats_naive_when_head_moves(calibrated):
    tr, cfg, model, naive, _ = calibrated
    rows = sim.evaluate(model, naive, tr, cfg, np.random.default_rng(5), n=300)
    moved = [r for r in rows if r[0] != "calibración"]
    assert np.median([r[2] for r in moved]) * 2 < np.median([r[1] for r in moved])
    assert max(r[2] for r in moved if r[0] != "combinado") < 170


def test_predict_supports_batch_and_single(calibrated):
    _, _, model, _, _ = calibrated
    batch = np.array([[10.0, 40.0], [-30.0, 120.0]])
    assert np.allclose(model.predict(batch)[1], model.predict(batch[1]))


def test_ridge_extrapolates_off_screen(calibrated):
    tr, cfg, model, _, _ = calibrated
    rng = np.random.default_rng(1)
    uv = np.column_stack((np.full(200, 1.3), rng.uniform(0.2, 0.8, 200)))
    r, t = sim.head_state(200, rng)
    por, ok = sim.geometric_feature(sim.render(uv, r, t, tr, rng), cfg)
    assert np.mean(model.predict(por[ok])[:, 0] > 1.0 + cfg.off_margin) > 0.95


def test_model_round_trip(tmp_path, calibrated):
    _, _, model, _, _ = calibrated
    path = tmp_path / "sub" / "cal.npz"
    model.save(path)
    loaded = GazeModel.load(path)
    assert loaded.screen == model.screen
    assert np.allclose(loaded.predict(model.mean), model.predict(model.mean))
    assert GazeModel.load(tmp_path / "missing.npz") is None


def test_corrupt_calibration_is_ignored(tmp_path):
    path = tmp_path / "cal.npz"
    path.write_bytes(b"not a zip")
    assert GazeModel.load(path) is None


def test_too_few_time_blocks_is_a_clear_error():
    t = np.arange(0.0, 8.0, 0.05)
    xy = np.column_stack((np.sin(t), np.cos(t)))
    with pytest.raises(ValueError, match="insuficientes"):
        fit_gaze_model(t, xy * 50, np.full(t.size, 0.3), t, xy, (1920, 1080))


def test_gap_in_samples_does_not_break_fit():
    t = np.concatenate((np.arange(0.0, 12.0, 0.05), np.arange(30.0, 45.0, 0.05)))
    tt = np.arange(0.0, 45.0, 1 / 60)
    xy = np.column_stack((np.sin(tt) * 0.5 + 0.5, np.cos(tt * 0.7) * 0.5 + 0.5))
    por = np.column_stack([np.interp(t, tt, xy[:, i]) for i in (0, 1)]) * 200
    model = fit_gaze_model(t, por, np.full(t.size, 0.3), tt, xy, (1920, 1080))
    assert model.error < 0.02


def test_poly_features_shape():
    assert poly_features(np.zeros((7, 2))).shape == (7, 5)


def test_trajectory_properties():
    tr = generate_trajectory(45.0, seed=1)
    assert tr.xy.min() >= 0.0 and tr.xy.max() <= 1.0
    speed = np.hypot(*(np.diff(tr.xy, axis=0) * (1.0, 9 / 16)).T) * 60.0
    assert np.percentile(speed, 95) < 0.9
    cells = {(int(x * 3.999), int(y * 2.999)) for x, y in tr.xy}
    assert len(cells) >= 11  # recorre casi toda la rejilla 4×3
    assert tr.radius.min() > 0.6 and tr.radius.max() < 1.4
    assert np.max(np.abs(np.diff(tr.radius))) < 0.02  # cambio sutil


def test_trajectories_differ_between_sessions():
    a, b = generate_trajectory(10.0, seed=1), generate_trajectory(10.0, seed=2)
    assert not np.allclose(a.xy, b.xy)


def drive(monitor, seq, dt=1 / 15, t0=100.0):
    """Aplica una lista de (estado, segundos) y devuelve el historial."""
    t, out = t0, []
    for gaze, secs in seq:
        for _ in range(round(secs / dt)):
            out.append((t, monitor.update(gaze, t)))
            t += dt
    return out


def make_monitor():
    return AttentionMonitor(3.0, 0.3, 0.5)


def test_alert_fires_after_three_seconds_only():
    hist = drive(make_monitor(), [(Gaze.OFF, 2.9)])
    assert not hist[-1][1]
    hist = drive(make_monitor(), [(Gaze.OFF, 3.2)])
    assert hist[-1][1]
    first = next(t for t, a in hist if a)
    assert first - hist[0][0] == pytest.approx(3.0, abs=0.08)


def test_returning_resets_timer():
    hist = drive(make_monitor(), [(Gaze.OFF, 2.5), (Gaze.ON, 0.5), (Gaze.OFF, 2.5)])
    assert not any(a for _, a in hist)


def test_noise_blip_does_not_reset_timer():
    hist = drive(make_monitor(), [(Gaze.OFF, 2.0), (Gaze.ON, 0.13), (Gaze.OFF, 1.2)])
    assert hist[-1][1]


def test_blink_is_ignored_both_ways():
    hist = drive(make_monitor(), [(Gaze.ON, 1), (Gaze.UNKNOWN, 0.3), (Gaze.ON, 1)])
    assert not any(a for _, a in hist)
    hist = drive(make_monitor(), [(Gaze.OFF, 2.0), (Gaze.UNKNOWN, 0.3), (Gaze.OFF, 1.0)])
    assert hist[-1][1]


def test_long_closed_eyes_count_as_away():
    hist = drive(make_monitor(), [(Gaze.UNKNOWN, 3.3)])
    assert hist[-1][1]


def test_alert_clears_on_return():
    hist = drive(make_monitor(), [(Gaze.OFF, 3.5), (Gaze.ON, 0.5)])
    assert hist[-1][1] is False


def test_heatmap_is_constant_memory_and_filters(tmp_path):
    hm = FixationHeatmap(192, (1920, 1080), 0.45)
    size = hm._grid.nbytes
    t = 0.0
    for _ in range(3000):
        hm.add(0.3, 0.4, t)
        t += 1 / 15
    assert hm._grid.nbytes == size
    assert hm.fixation_s == pytest.approx(200.0, abs=0.2)
    before = hm.fixation_s
    hm.add(0.9, 0.9, t)  # sacádico: no suma
    assert hm.fixation_s == before
    hm.add(1.4, 0.5, t + 0.07)  # fuera de pantalla: no suma
    assert hm.fixation_s == before

    import cv2
    path = hm.render(tmp_path / "ñandú" / "h.png")
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (1080, 1920, 3)
    hot = image[int(0.4 * 1080), int(0.3 * 1920)].astype(int)
    cold = image[900, 1700].astype(int)
    assert hot.sum() > cold.sum() + 150


def test_empty_heatmap_writes_nothing(tmp_path):
    hm = FixationHeatmap(192, (1920, 1080), 0.45)
    assert hm.render(tmp_path / "h.png") is None


def _pipeline_with(model, cfg):
    pipe = GazePipeline(cfg, sim.W, sim.H)
    pipe.set_model(model)
    return pipe


def test_pipeline_no_face_triggers_alert(calibrated):
    _, cfg, model, _, _ = calibrated
    pipe = _pipeline_with(model, cfg)
    t, sample = 0.0, None
    for _ in range(50):
        sample = pipe.track(None, t)
        t += 1 / 15
    assert sample.alert and not sample.has_point


def test_pipeline_on_screen_stays_quiet_and_fills_heatmap(calibrated, tmp_path):
    tr, cfg, model, _, _ = calibrated
    pipe = _pipeline_with(model, cfg)
    rng = np.random.default_rng(2)
    uv = np.tile([0.35, 0.6], (150, 1))
    r, t = sim.head_state(150, rng)
    px = sim.render(uv, r, t, tr, rng)
    samples = [pipe.track(p, i / 15) for i, p in enumerate(px)]
    assert not any(s.alert for s in samples)
    assert np.hypot(samples[-1].x - 0.35, samples[-1].y - 0.6) < 0.12
    assert pipe.heatmap.fixation_s > 8.0


def test_pipeline_blink_keeps_last_point(calibrated):
    tr, cfg, model, _, _ = calibrated
    pipe = _pipeline_with(model, cfg)
    rng = np.random.default_rng(3)
    r, t = sim.head_state(1, rng)
    px = sim.render(np.array([[0.5, 0.5]]), r, t, tr, rng)[0]
    closed = px.copy()
    closed[20:24:2] = closed[21:24:2]  # párpados juntos
    assert eye_aspect_ratio(closed) < 0.01
    open_sample = pipe.track(px, 0.0)
    blink = pipe.track(closed, 0.066)
    assert blink.gaze is Gaze.UNKNOWN and not blink.has_point
    assert (blink.x, blink.y) == (open_sample.x, open_sample.y)


def test_one_euro_reduces_jitter_but_follows_jumps():
    f = OneEuroFilter(0.7, 1.2)
    rng = np.random.default_rng(0)
    raw = 0.5 + rng.normal(0, 0.02, (300, 2))
    out = np.array([f(x, i / 15).copy() for i, x in enumerate(raw)])
    assert out[50:].std() < raw[50:].std() * 0.6
    for i in range(300, 310):
        y = f(np.array([0.9, 0.9]), i / 15).copy()
    assert y[0] > 0.8


@pytest.fixture(scope="module")
def session_fit():
    tr, cfg = sim.Truth(eye_radius=11.2, eye_depth=8.0, focal=1.05 * sim.W), Config()
    data = sim.session(tr, cfg, np.random.default_rng(1))
    model, report = fit_session(data, sim.SCREEN_PX, cfg, (sim.W, sim.H))
    return tr, cfg, data, model, report


def test_personal_params_beat_generic_on_held_out_points(session_fit):
    _, _, _, model, report = session_fit
    assert report.chosen == "personal"
    assert report.current.moved_px < report.baseline.moved_px * 0.3
    assert model.params.depth == pytest.approx(8.0, abs=1.5)


def test_validation_error_replaces_training_error(session_fit):
    _, _, _, model, report = session_fit
    diagonal = np.hypot(*sim.SCREEN_PX)
    assert model.error == pytest.approx(report.current.accuracy_px / diagonal)


def test_personal_params_survive_save_and_reach_pipeline(tmp_path, session_fit):
    _, cfg, _, model, _ = session_fit
    path = tmp_path / "c.npz"
    model.save(path)
    loaded = GazeModel.load(path)
    assert loaded.params == model.params
    pipe = GazePipeline(cfg, sim.W, sim.H)
    pipe.set_model(loaded)
    assert pipe._geometry.params == model.params


def test_unknown_calibration_version_is_rejected(tmp_path, session_fit):
    path = tmp_path / "old.npz"
    np.savez(path, version=1)
    assert GazeModel.load(path) is None


def test_still_head_keeps_generic_model():
    tr, cfg = sim.Truth(), Config()
    data = sim.session(tr, cfg, np.random.default_rng(2), head_moves=False)
    model, report = fit_session(data, sim.SCREEN_PX, cfg, (sim.W, sim.H))
    assert report.chosen == "generico"
    assert "muy poco" in report.reason
    assert model.params == EyeParams()


def test_missing_validation_keeps_generic():
    tr, cfg = sim.Truth(), Config()
    data = sim.session(tr, cfg, np.random.default_rng(3))
    keep = data["tag"] < TAG_POINT
    data = {**data, "t": data["t"][keep], "pts": data["pts"][keep], "tag": data["tag"][keep]}
    model, report = fit_session(data, sim.SCREEN_PX, cfg, (sim.W, sim.H))
    assert report is None and model.params == EyeParams()


def test_assess_reuses_saved_model_without_refitting(session_fit):
    _, cfg, data, model, report = session_fit
    again = assess_model(model, data, cfg, (sim.W, sim.H))
    assert again.chosen == "actual" and again.baseline is None
    assert again.current.accuracy_px == pytest.approx(report.current.accuracy_px)


def test_score_separates_bias_from_jitter():
    targets, groups = validation_targets()
    index = np.repeat(np.arange(targets.shape[0]), 20)
    rng = np.random.default_rng(0)
    biased = targets[index] + (0.05, 0.0)
    noisy = targets[index] + rng.normal(0, 0.03, (index.size, 2))
    a = score_points(biased, index, targets, groups, (1920, 1080))
    b = score_points(noisy, index, targets, groups, (1920, 1080), deg_per_px=0.02)
    assert a.accuracy_px == pytest.approx(96.0, abs=0.1) and a.precision_px < 0.01
    assert b.precision_px > 25 and b.accuracy_px < 25
    assert b.accuracy_deg == pytest.approx(b.accuracy_px * 0.02)


def test_score_needs_enough_points():
    targets, groups = validation_targets()
    index = np.repeat(np.arange(3), 20)
    assert score_points(targets[index], index, targets, groups, (1920, 1080)) is None


def test_history_appends_json_without_images(tmp_path, session_fit):
    _, _, _, _, report = session_fit
    path = tmp_path / "x" / "h.jsonl"
    append_history(report, path)
    append_history(report, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["chosen"] == "personal"
    assert "pts" not in lines[0]


def fit_biased(bias, seed=1):
    tr = sim.Truth(pursuit_error=0.012, bias_mm=bias, bias_shape="wave")
    cfg = Config()
    data = sim.session(tr, cfg, np.random.default_rng(seed))
    return (*fit_session(data, sim.SCREEN_PX, cfg, (sim.W, sim.H)), tr, cfg)


@pytest.fixture(scope="module")
def biased():
    return fit_biased(1.5)


def test_correction_activates_and_reduces_held_out_error(biased):
    model, report, *_ = biased
    assert report.correction_active and model.residual is not None
    assert balance(report.current) <= 0.95 * balance(report.before_correction)


def alert_after(model, cfg, tr, uv, seed=8):
    """Sostiene la mirada 5 s en un punto y devuelve si la alerta quedó activa."""
    rng = np.random.default_rng(seed)
    pipe = GazePipeline(cfg, sim.W, sim.H)
    pipe.set_model(model)
    q = np.tile(uv, (75, 1)).astype(float) + rng.normal(0, 0.02, (75, 2))
    r, t = sim.head_state(75, rng)
    px = sim.render(q, r, t, tr, rng)
    return [pipe.track(p, i / 15) for i, p in enumerate(px)][-1].alert


@pytest.fixture(scope="module")
def vertical_symptom():
    """Reproduce lo medido en Windows: pendiente vertical cercana a 0.5."""
    tr = sim.Truth(pursuit_error=0.012, lid_coupling=2.0, down_bias=9.0)
    cfg = Config()
    data = sim.session(tr, cfg, np.random.default_rng(1))
    info = ScreenInfo(530.0, 300.0, "EDID")
    return (*fit_session(data, sim.SCREEN_PX, cfg, (sim.W, sim.H), info), tr, cfg)


def test_lids_fix_the_vertical_symptom(vertical_symptom):
    model, report, *_ = vertical_symptom
    assert report.baseline.vertical_slope < 0.6
    assert report.lids_active and model.lid_mean is not None
    assert report.current.vertical_r2 > 0.9
    assert report.current.accuracy_px < 0.3 * report.baseline.accuracy_px


@pytest.mark.parametrize("uv, expected", [
    ((0.5, 0.5), False), ((0.5, 0.95), False), ((1.25, 0.5), True),
    ((0.5, -0.3), True), ((0.5, 1.3), True), ((0.5, 2.0), True),
])
def test_alert_works_in_every_direction_with_symptom(vertical_symptom, uv, expected):
    model, _, tr, cfg = vertical_symptom
    assert alert_after(model, cfg, tr, uv) is expected


def test_correction_keeps_off_screen_detection(biased):
    model, _, tr, cfg = biased
    assert alert_after(model, cfg, tr, (1.5, 0.5))


def test_blink_mask_keeps_sustained_low_lids():
    t = np.arange(200) / 20.0
    ear = np.full(200, 0.30)
    ear[40:44] = 0.05  # parpadeo de 0.2 s
    ear[100:150] = 0.16  # mirar abajo 2.5 s
    ear[170:190] = 0.05  # ojos cerrados, aunque dure
    mask = blink_mask(ear, t, 0.30)
    assert mask[40:44].all() and not mask[100:150].any() and mask[170:190].all()
    assert not mask[:40].any() and mask.sum() == 24


def test_blink_mask_without_blinks():
    t = np.arange(50) / 20.0
    assert not blink_mask(np.full(50, 0.3), t, 0.3).any()


def test_lid_features_single_and_batch_agree():
    rng = np.random.default_rng(0)
    r, t = sim.head_state(5, rng)
    uv = rng.uniform(0.1, 0.9, (5, 2))
    px = sim.render(uv, r, t, sim.Truth(lid_coupling=1.0), rng)
    assert np.allclose(lid_features(px)[2], lid_features(px[2]))


def test_lid_height_tracks_vertical_gaze():
    rng = np.random.default_rng(0)
    tr = sim.Truth(noise_px=0.0)
    r, t = sim.head_state(2, rng, sway=False)
    px = sim.render(np.array([[0.5, 0.1], [0.5, 0.9]]), r, t, tr, rng)
    height = lid_features(px)[:, 1]
    assert height[0] > height[1]  # mirar abajo baja el iris entre los párpados


def test_lids_model_round_trip(tmp_path, vertical_symptom):
    model, *_ = vertical_symptom
    path = tmp_path / "c.npz"
    model.save(path)
    loaded = GazeModel.load(path)
    por, pose, lids = model.mean[None], np.zeros((1, 3)), np.array([[0.2, 0.5]])
    assert np.allclose(loaded.predict(por, pose, lids), model.predict(por, pose, lids))
    assert loaded.ear_closed == pytest.approx(model.ear_closed)


def test_model_requires_no_lids_to_predict(vertical_symptom):
    model, *_ = vertical_symptom
    assert model.predict(model.mean[None]).shape == (1, 2)


def test_report_shows_vertical_monitor_and_layers(vertical_symptom):
    _, report, *_ = vertical_symptom
    text = report_text(report)
    assert "Vertical: R²" in text and "EDID" in text and "distancia" in text
    assert "párpados" in text


def test_screen_width_is_estimated_without_edid(session_fit):
    _, _, _, model, report = session_fit
    assert report.screen_source == "estimado"
    assert 200 < report.screen_mm[0] < 1200


def test_session_replays_identically(tmp_path, vertical_symptom):
    _, report, tr, cfg = vertical_symptom
    data = sim.session(tr, cfg, np.random.default_rng(1))
    path = tmp_path / "s" / "last.npz"
    save_session(data, sim.SCREEN_PX, (sim.W, sim.H), ScreenInfo(530.0, 300.0, "EDID"), path)
    with np.load(path, allow_pickle=False) as saved:
        assert np.array_equal(saved["pts"], data["pts"])
        assert tuple(saved["screen"]) == sim.SCREEN_PX
        assert "frame" not in saved


def edid(width=527, height=296, basic=(53, 30)):
    block = bytearray(128)
    block[:8] = bytes((0, 255, 255, 255, 255, 255, 255, 0))
    block[21], block[22] = basic
    block[66], block[67] = width & 0xFF, height & 0xFF
    block[68] = ((width >> 8) << 4) | (height >> 8)
    block[127] = (256 - sum(block[:127]) % 256) % 256
    return bytes(block)


def test_parse_edid_reads_millimeters():
    assert parse_edid(edid()) == (527, 296)
    assert parse_edid(edid(1600, 900)) == (1600, 900)


def test_parse_edid_falls_back_to_centimeters():
    assert parse_edid(edid(0, 0)) == (530, 300)


def test_parse_edid_rejects_corrupt_data():
    broken = bytearray(edid())
    broken[30] ^= 0xFF
    assert parse_edid(bytes(broken)) is None
    assert parse_edid(b"short") is None
    assert parse_edid(edid(0, 0, (0, 0))) is None  # proyector: sin tamaño


def test_registry_key_from_device_id():
    device = r"\\?\DISPLAY#GSM5BBF#5&3a1f&0&UID4352#{e6f07b5f-ee97-4a90-b076-33f57bf4eaa7}"
    assert registry_key(device) == (
        r"SYSTEM\CurrentControlSet\Enum\DISPLAY\GSM5BBF\5&3a1f&0&UID4352\Device Parameters"
    )
    assert registry_key("nada") is None


def test_orientation_and_coherence():
    assert fit_orientation((527, 296), (1920, 1080)).width_mm == 527
    assert fit_orientation((527, 296), (1080, 1920)).width_mm == 296
    assert fit_orientation((500, 500), (1920, 1080)) is None


def test_detect_screen_reads_fake_windows(monkeypatch):
    monkeypatch.setattr(display.sys, "platform", "win32")
    monkeypatch.setattr(display, "_primary_device_id", lambda: "x#GSM#123#y")
    monkeypatch.setattr(display, "_read_edid", lambda key: edid())
    info = display.detect_screen((1920, 1080))
    assert info.source == "EDID" and info.inches == pytest.approx(23.8, abs=0.1)


def test_detect_screen_survives_registry_errors(monkeypatch):
    def broken(key):
        raise OSError("sin acceso")

    monkeypatch.setattr(display.sys, "platform", "win32")
    monkeypatch.setattr(display, "_primary_device_id", lambda: "x#GSM#123#y")
    monkeypatch.setattr(display, "_read_edid", broken)
    assert display.detect_screen((1920, 1080)) is None


def test_correction_is_bounded_far_from_data():
    x = np.random.default_rng(0).normal(0, 0.1, (300, 2))
    pose = np.zeros((300, 3))
    t = np.arange(300) / 15.0
    residual = 0.02 * np.sin(8 * x)
    kernel = fit_residual(x, pose, residual, t)
    far = kernel.correction(np.array([9.0, -9.0]), np.array([1.5, -1.5, 2.0]))
    assert np.abs(far).max() < 0.1


def test_predict_without_pose_uses_ridge_only(biased):
    model, *_ = biased
    por = model.mean[None]
    plain = design_matrix(np.zeros((1, 2)), model.scaled_lids(None, (1,))) @ model.coef.T + model.intercept
    assert np.allclose(model.predict(por), plain)
    assert not np.allclose(model.predict(por, np.zeros((1, 3))), plain)


def test_kernel_inference_matches_sklearn_form():
    x = np.random.default_rng(1).uniform(-0.3, 0.3, (200, 2))
    pose = np.column_stack((x, np.zeros(200)))
    kernel = fit_residual(x, pose, 0.03 * np.sin(6 * x), np.arange(200) / 15.0)
    one = kernel.correction(x[3], pose[3])
    batch = kernel.correction(x, pose)[3]
    assert np.allclose(one, batch)


def test_model_with_correction_round_trip_without_pickle(tmp_path, biased):
    model, *_ = biased
    path = tmp_path / "c.npz"
    model.save(path)
    loaded = GazeModel.load(path)
    por, pose = model.mean[None], np.array([[0.1, -0.05, 0.1]])
    assert np.allclose(loaded.predict(por, pose), model.predict(por, pose))
    with np.load(path, allow_pickle=False) as d:
        assert "r_dual" in d


def test_previous_calibration_version_still_loads(tmp_path, session_fit):
    _, _, _, model, _ = session_fit
    path = tmp_path / "v2.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as d:
        content = {k: d[k] for k in d.files if not k.startswith("r_")}
    content["version"] = 2
    np.savez(path, **content)
    loaded = GazeModel.load(path)
    assert loaded is not None and loaded.residual is None


def test_pipeline_uses_pose_for_prediction(biased):
    model, _, tr, cfg = biased
    pipe = GazePipeline(cfg, sim.W, sim.H)
    pipe.set_model(model)
    rng = np.random.default_rng(6)
    r, t = sim.head_state(1, rng, yaw=12)
    px = sim.render(np.array([[0.5, 0.5]]), r, t, tr, rng)[0]
    obs = pipe.observe(px)
    expected = model.predict(obs.por, obs.pose, obs.lids)
    pipe2 = GazePipeline(cfg, sim.W, sim.H)
    pipe2.set_model(model)
    sample = pipe2.track(px, 0.0)
    assert (sample.x, sample.y) == pytest.approx(tuple(expected), abs=1e-6)
