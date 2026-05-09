"""Your submission entry point — replace the baseline model here.

Contract (do NOT change the signature):

    predict(request: dict) -> dict

Input: one request dict (keys below). Output: a dict with intent probability
and 4 future bounding boxes.

Request keys (all required):
    ped_id                str       opaque token, stable within the dataset
    frame_w, frame_h      int
    time_of_day, weather, location   str  (may be empty strings)
    ego_available         bool      True when ego speed/yaw history is valid
    bbox_history          list[16][4]  past bboxes at 15 Hz, oldest → current
                                       each bbox is [x1, y1, x2, y2] in pixels
    ego_speed_history     list[16]  past ego speeds (m/s); zeros if unavailable
    ego_yaw_history       list[16]  past yaw rates (rad/s); zeros if unavailable
    requested_at_frame    int       native-30fps frame id of current observation

Output keys (all required):
    intent                float     P(crossing within next 2s), in [0, 1]
    bbox_500ms            list[4]   predicted bbox at +0.5 s
    bbox_1000ms           list[4]   predicted bbox at +1.0 s
    bbox_1500ms           list[4]   predicted bbox at +1.5 s
    bbox_2000ms           list[4]   predicted bbox at +2.0 s

The grader calls predict() once per request, row-by-row. Prediction order
must match input order.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "model.pkl"
HORIZONS_FRAMES = [8, 15, 23, 30]   # at 15 Hz → 0.5, 1.0, 1.5, 2.0 s
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

_cached_model = None


def _load_model():
    global _cached_model
    if _cached_model is None:
        with open(MODEL_PATH, "rb") as f:
            _cached_model = pickle.load(f)
    return _cached_model


def _as_2d(x) -> np.ndarray:
    """Coerce list-of-lists / object-array-of-arrays to (N, 4) float64."""
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _engineered_features(req: dict) -> np.ndarray:
    """Mirrors baseline.py's feature builder so the XGBoost model sees the same
    layout at inference as at training. Keep this in lock-step with baseline.py.
    """
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    
    vx = np.diff(cx)
    vy = np.diff(cy)
    
    # New features: Acceleration and Speed Trends
    # Short-term (last 4 frames) vs Long-term (last 12 frames)
    vx_short = vx[-4:].mean()
    vx_long = vx[-12:].mean()
    accel_x = vx[-4:].mean() - vx[-8:-4].mean() if len(vx) >= 8 else 0.0
    
    # Speed (magnitude of velocity)
    speed = np.sqrt(vx**2 + vy**2)
    speed_last = speed[-1]
    speed_mean = speed.mean()
    
    # Stopping signal: is the pedestrian currently "still"?
    is_stopping = 1.0 if speed_last < 1.0 else 0.0 # 1px per frame threshold
    
    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    feats = [
        cx[-1] / fw,
        cy[-1] / fh,
        w[-1] / fw,
        h[-1] / fh,
        vx_short / fw,
        vy[-4:].mean() / fh,
        vx.std() / fw,
        vy.std() / fh,
        (h / (w + 1e-6)).mean(),
        float(req["ego_available"]),
        ego_s.mean(), ego_s[-1], ego_s.max(),
        ego_y.mean(), ego_y[-1], np.abs(ego_y).max(),
        # New intent-focused features
        vx_long / fw,
        accel_x / fw,
        speed_last / fw,
        speed_mean / fw,
        is_stopping,
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


def _constant_velocity_trajectory(req: dict) -> dict[str, list[float]]:
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    
    # EMA smoothing of velocity, acceleration, and size changes
    vx_raw = np.diff(cx)
    vy_raw = np.diff(cy)
    vw_raw = np.diff(w)
    vh_raw = np.diff(h)
    
    alpha_v = 0.3
    alpha_a = 0.1
    vx, vy = vx_raw[0], vy_raw[0]
    vw, vh = vw_raw[0], vh_raw[0]
    ax, ay = 0.0, 0.0
    
    for i in range(1, len(vx_raw)):
        prev_vx, prev_vy = vx, vy
        vx = alpha_v * vx_raw[i] + (1 - alpha_v) * vx
        vy = alpha_v * vy_raw[i] + (1 - alpha_v) * vy
        vw = alpha_v * vw_raw[i] + (1 - alpha_v) * vw
        vh = alpha_v * vh_raw[i] + (1 - alpha_v) * vh
        
        cur_ax = vx - prev_vx
        cur_ay = vy - prev_vy
        ax = alpha_a * cur_ax + (1 - alpha_a) * ax
        ay = alpha_a * cur_ay + (1 - alpha_a) * ay
        
    cur_cx, cur_cy = float(cx[-1]), float(cy[-1])
    cur_w, cur_h = float(w[-1]), float(h[-1])
    vx, vy, vw, vh = float(vx), float(vy), float(vw), float(vh)
    ax, ay = float(ax), float(ay)

    out: dict[str, list[float]] = {}
    for h_frames, key in zip(HORIZONS_FRAMES, HORIZON_KEYS):
        # Dampen acceleration over time
        nx = cur_cx + vx * h_frames + 0.5 * ax * (h_frames ** 2) * 0.5
        ny = cur_cy + vy * h_frames + 0.5 * ay * (h_frames ** 2) * 0.5
        
        # Adaptive size with floor
        nw = max(10.0, cur_w + vw * h_frames)
        nh = max(10.0, cur_h + vh * h_frames)
        
        out[key] = [nx - nw / 2, ny - nh / 2, nx + nw / 2, ny + nh / 2]
    return out


def predict(request: dict) -> dict:
    intent_model = _load_model()["intent"]
    feats = _engineered_features(request).reshape(1, -1)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    intent_prob = float(intent_model.predict_proba(feats)[0, 1])
    if not np.isfinite(intent_prob):
        intent_prob = 0.5

    out = _constant_velocity_trajectory(request)
    for k in HORIZON_KEYS:
        out[k] = [float(v) if np.isfinite(v) else 0.0 for v in out[k]]
    out["intent"] = intent_prob
    return out
