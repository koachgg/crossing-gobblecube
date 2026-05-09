import pandas as pd
import numpy as np

dev = pd.read_parquet('data/dev.parquet')

def as_2d(x):
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])

def evaluate_traj_fn(fn_name, fn):
    ades = []
    for h, steps in [('bbox_500ms', 8), ('bbox_1000ms', 15), ('bbox_1500ms', 23), ('bbox_2000ms', 30)]:
        tb = np.stack([as_2d(x) for x in dev[h]])
        tcx = (tb[:, 0] + tb[:, 2]) * 0.5
        tcy = (tb[:, 1] + tb[:, 3]) * 0.5
        
        preds_cx, preds_cy = [], []
        preds_w, preds_h = [], []
        for _, row in dev.iterrows():
            bh = as_2d(row['bbox_history'])
            nx, ny, nw, nh = fn(bh, steps, row)
            preds_cx.append(nx)
            preds_cy.append(ny)
            preds_w.append(nw)
            preds_h.append(nh)
            
        preds_cx = np.array(preds_cx)
        preds_cy = np.array(preds_cy)
        ade = float(np.hypot(preds_cx - tcx, preds_cy - tcy).mean())
        ades.append(ade)

    mean_ade = np.mean(ades)
    print(f'[{fn_name}] Mean ADE: {mean_ade:.2f} px')
    return mean_ade

def ema_accel_adaptive_fn(bh, steps, row):
    cx = (bh[:, 0] + bh[:, 2]) * 0.5
    cy = (bh[:, 1] + bh[:, 3]) * 0.5
    w = bh[:, 2] - bh[:, 0]
    h = bh[:, 3] - bh[:, 1]
    
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
        
    nx = cx[-1] + vx * steps + 0.5 * ax * (steps ** 2) * 0.5
    ny = cy[-1] + vy * steps + 0.5 * ay * (steps ** 2) * 0.5
    nw = max(10, w[-1] + vw * steps)
    nh = max(10, h[-1] + vh * steps)
    return nx, ny, nw, nh

def baseline_fn(bh, steps, row):
    cx = (bh[:, 0] + bh[:, 2]) * 0.5
    cy = (bh[:, 1] + bh[:, 3]) * 0.5
    w = bh[:, 2] - bh[:, 0]
    h = bh[:, 3] - bh[:, 1]
    vx = float(np.diff(cx[-5:]).mean())
    vy = float(np.diff(cy[-5:]).mean())
    return cx[-1] + vx * steps, cy[-1] + vy * steps, w[-1], h[-1]

evaluate_traj_fn('Baseline', baseline_fn)
evaluate_traj_fn('EMA + Accel + Adaptive Size', ema_accel_adaptive_fn)
