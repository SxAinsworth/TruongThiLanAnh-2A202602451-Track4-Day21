"""[B3] Đo latency các bước của pipeline QA calibration trên CPU: bỏ lần chạy đầu, lặp N lần, báo p50/p95.

Các bước đo (cùng 1 frame, dữ liệu đã nạp sẵn trong RAM, không tính thời gian đọc file):
  projection    velo_to_cam + cam_to_image cho toàn bộ point cloud
  edge_score    projection + edge-alignment score (đã có sẵn bản đồ cạnh Canny của ảnh)
  canny         bản đồ khoảng cách tới cạnh Canny của ảnh (1 lần mỗi ảnh)
  drift_check   kiểm tra cực đại cục bộ cho 1 trục = 9 lần edge_score (như src/drift_detect.py)

    python -m src.latency --data-root data/kitti_mini --frame 000011 --runs 21
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.calib_sweep import edge_alignment, image_edge_distance, make_calib
from src.drift_detect import GRID
from starter.datasets import load_frame
from starter.projection import cam_to_image, velo_to_cam


def main() -> None:
    ap = argparse.ArgumentParser(description="[B3] Đo latency p50/p95 của projection, edge score và drift check")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", default="000011")
    ap.add_argument("--runs", type=int, default=21, help="tổng số lần chạy, lần đầu bị bỏ (warm-up)")
    ap.add_argument("--hardware", default="Intel Core i7-1185G7 @ 3.00GHz, RAM 16 GB, chỉ dùng CPU (máy có GTX 1650 nhưng không dùng)",
                    help="mô tả phần cứng ghi vào CSV")
    ap.add_argument("--out", default="results/latency_kitti.csv")
    args = ap.parse_args()

    fr = load_frame(args.data_root, args.frame)
    xyz, image, calib = fr["points"][:, :3].astype(np.float64), fr["image"], fr["calib"]
    edist = image_edge_distance(image)

    def projection():
        return cam_to_image(velo_to_cam(xyz, calib), calib.P2, image.shape)

    def edge_score(c=calib):
        uv, depth, _ = cam_to_image(velo_to_cam(xyz, c), c.P2, image.shape)
        return edge_alignment(uv, depth, image.shape, edist)

    def drift_check():
        return max(edge_score(make_calib(calib, "yaw_deg", g))[0] for g in GRID)

    steps = {"projection": projection, "edge_score": edge_score,
             "canny": lambda: image_edge_distance(image), "drift_check": drift_check}
    rows = []
    for name, fn in steps.items():
        for i in range(args.runs):
            t0 = time.perf_counter()
            fn()
            rows.append({"step": name, "run": i, "warmup": i == 0, "ms": round((time.perf_counter() - t0) * 1000, 3)})
    df = pd.DataFrame(rows)
    df["frame"], df["n_points"], df["hardware"], df["python"] = args.frame, len(xyz), args.hardware, platform.python_version()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    m = df[~df.warmup].groupby("step", sort=False).ms
    summary = pd.DataFrame({"p50_ms": m.median(), "p95_ms": m.quantile(0.95), "n_runs": m.size(),
                            "warmup_ms": df[df.warmup].set_index("step").ms})
    print(f"{args.hardware}, frame {args.frame}, {len(xyz)} điểm")
    print(summary.round(1).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
