"""Phát hiện calibration drift không cần label, bằng edge-alignment score (topic A, Advanced).

Khi chạy thật, hệ thống chỉ có MỘT calib (có thể đã lệch) và không biết score "đúng" của frame.
Score tuyệt đối khác nhau rất nhiều giữa các scene (KITTI: 30-80%), nên không đặt ngưỡng tuyệt đối được.
Thay vào đó dùng kiểm tra cực đại cục bộ:
  - S0      = score với calib hiện tại
  - S(c)    = score khi xoay thêm một góc hiệu chỉnh c quanh trục đang kiểm tra, c thuộc lưới ±GRID
  - gain    = max_c S(c) - S0
  Calib đúng thì S0 gần như là cực đại -> gain nhỏ. Calib lệch thì có c làm score tăng -> gain lớn.
  Ngưỡng = gain lớn nhất trên các frame chưa perturb (0 báo động giả trên tập này).
  Drift ước lượng = -argmax_c S(c).

Ví dụ:
    python -m src.drift_detect --data-root data/kitti_mini --out results/drift_detect_kitti.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.calib_sweep import EDGE_WINDOW_PX, edge_alignment, image_edge_distance, make_calib
from starter.datasets import list_frames, load_frame
from starter.projection import cam_to_image, velo_to_cam

GRID = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
DRIFTS = [0.0, 0.5, 1.0, 2.0, 3.0]


def score(xyz, calib, image, edge_dist, window):
    pts_cam = velo_to_cam(xyz, calib)
    uv, depth, _ = cam_to_image(pts_cam, calib.P2, image.shape)
    return edge_alignment(uv, depth, image.shape, edge_dist, window)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phát hiện calibration drift bằng kiểm tra cực đại cục bộ của edge-alignment score")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame-step", type=int, default=1, help="lấy 1 frame mỗi N frame")
    ap.add_argument("--axes", nargs="*", default=["yaw_deg", "pitch_deg"], help="trục góc cần kiểm tra")
    ap.add_argument("--edge-window", type=int, default=EDGE_WINDOW_PX)
    ap.add_argument("--out", default="results/drift_detect.csv", help="CSV theo frame; CSV tổng hợp lưu với hậu tố _summary")
    args = ap.parse_args()

    rows = []
    for fid in list_frames(args.data_root)[:: args.frame_step]:
        fr = load_frame(args.data_root, fid)
        xyz = fr["points"][:, :3].astype(np.float64)
        edge_dist = image_edge_distance(fr["image"])
        for axis in args.axes:
            for drift in DRIFTS:
                # calib "hiện tại" = calib gốc bị lệch `drift`; hiệu chỉnh c cộng thêm trên cùng trục
                s = {c: score(xyz, make_calib(fr["calib"], axis, drift + c), fr["image"], edge_dist, args.edge_window)
                     for c in GRID}
                best = max(GRID, key=lambda c: (s[c], -abs(c)))
                rows.append({"frame": fid, "axis": axis, "drift": drift, "score_current": round(s[0.0], 2),
                             "score_best": round(s[best], 2), "gain": round(s[best] - s[0.0], 2),
                             "est_drift": -best})
    df = pd.DataFrame(rows)

    summary = []
    for axis, g in df.groupby("axis"):
        thr = g.loc[g.drift == 0, "gain"].max()
        for drift, gd in g.groupby("drift"):
            summary.append({"axis": axis, "drift": drift, "threshold_gain": thr, "n_frames": len(gd),
                            "detect_rate": round(float((gd.gain > thr).mean()), 3),
                            "median_gain": float(gd.gain.median()),
                            "est_drift_median": float(gd.est_drift.median())})
    sm = pd.DataFrame(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    sm.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(sm.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
