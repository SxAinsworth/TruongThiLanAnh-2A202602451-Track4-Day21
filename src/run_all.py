"""Chạy lại toàn bộ bài lab theo từng phần (checkpoint), hoặc chỉ một vài phần.

    python -m src.run_all                 # chạy tất cả
    python -m src.run_all --list          # xem danh sách phần
    python -m src.run_all --steps cp3_sweep_kitti cp4_figures
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PY = sys.executable

# (tên phần, mô tả, danh sách lệnh). Thứ tự quan trọng: cp4_figures đọc CSV của cp3.
STEPS = [
    ("cp0_data", "Kiểm tra dữ liệu + thống kê point cloud", [
        [PY, "tools/verify_data.py", "--data-root", "data/kitti_mini"],
        [PY, "tools/verify_data.py", "--data-root", "data/nuscenes_mini_subset"],
        [PY, "-m", "starter.data_health", "--data-root", "data/synthetic"],
        [PY, "-m", "starter.data_health", "--data-root", "data/kitti_mini", "--out", "results/data_health_kitti.csv"],
        [PY, "-m", "starter.data_health", "--data-root", "data/nuscenes_mini_subset", "--out", "results/data_health_nusc.csv"],
    ]),
    ("cp2_projection", "Self-test 2 hàm TODO + overlay calib đúng: synthetic, KITTI gần/vừa/xa, nuScenes", [
        [PY, "-m", "src.test_projection"],
        [PY, "-m", "starter.projection", "--data-root", "data/synthetic", "--frame", "000000"],
        [PY, "-m", "starter.projection", "--data-root", "data/kitti_mini", "--frame", "000025"],
        [PY, "-m", "starter.projection", "--data-root", "data/kitti_mini", "--frame", "000011"],
        [PY, "-m", "starter.projection", "--data-root", "data/kitti_mini", "--frame", "000009"],
        [PY, "-m", "starter.projection", "--data-root", "data/nuscenes_mini_subset", "--frame", "scene-0103_010"],
    ]),
    ("cp3_sample_yaw", "Script mẫu codelab: yaw sweep 3 frame + biểu đồ (đối chiếu bảng kỳ vọng)", [
        [PY, "-m", "src.exp_yaw_sweep", "--data-root", "data/kitti_mini", "--frames", "000008", "000011", "000049"],
        [PY, "-m", "src.plot_yaw_sweep"],
    ]),
    ("cp3_sweep_kitti","Sweep drift yaw/pitch/roll/t trên KITTI", [
        [PY, "-m", "src.calib_sweep", "--data-root", "data/kitti_mini", "--out", "results/calib_sweep_kitti.csv"],
    ]),
    ("cp3_sweep_nusc", "Sweep drift trên nuScenes (bonus B5)", [
        [PY, "-m", "src.calib_sweep", "--data-root", "data/nuscenes_mini_subset", "--frame-step", "4",
         "--edge-window", "15", "--out", "results/calib_sweep_nusc.csv"],
    ]),
    ("cp3_drift_detect", "Phát hiện drift bằng edge score (Advanced)", [
        [PY, "-m", "src.drift_detect", "--data-root", "data/kitti_mini", "--out", "results/drift_detect_kitti.csv"],
    ]),
    ("cp4_figures", "Biểu đồ, ảnh demo và ảnh fail_*", [
        [PY, "-m", "src.make_figures"],
    ]),
    ("cp5_check", "Kiểm tra trước khi nộp", [
        [PY, "tools/check_submission.py"],
    ]),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # console Windows mặc định cp1252, không in được tiếng Việt
    names = [s[0] for s in STEPS]
    ap = argparse.ArgumentParser(description="Chạy lại toàn bộ bài lab theo từng phần")
    ap.add_argument("--steps", nargs="*", choices=names, default=names, help="các phần cần chạy (mặc định: tất cả)")
    ap.add_argument("--list", action="store_true", help="in danh sách phần rồi thoát")
    args = ap.parse_args()

    if args.list:
        for name, desc, cmds in STEPS:
            print(f"{name:<18} {desc} ({len(cmds)} lệnh)")
        return

    env = {**os.environ, "PYTHONUTF8": "1"}
    summary = []
    for name, desc, cmds in STEPS:
        if name not in args.steps:
            continue
        print(f"\n===== {name}: {desc}", flush=True)
        t0 = time.perf_counter()
        for cmd in cmds:
            print("$ " + " ".join(["python"] + cmd[1:]), flush=True)
            if subprocess.run(cmd, env=env).returncode != 0:
                print(f"[FAIL] {name}: lệnh lỗi, dừng lại")
                sys.exit(1)
        summary.append((name, time.perf_counter() - t0))

    print("\n===== Tóm tắt")
    for name, dt in summary:
        print(f"[OK] {name:<18} {dt:6.1f}s")


if __name__ == "__main__":
    main()
