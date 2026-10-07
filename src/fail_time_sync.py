"""fail_04 [Time]: chiếu LiDAR nuScenes lên camera mà KHÔNG bù chuyển động xe giữa 2 thời điểm chụp.

Camera và LiDAR nuScenes chụp lệch nhau vài chục ms. Bản có bù (use_ego_motion=True) được coi là đúng:
label 2D và điểm của object lấy từ bản này. Sau đó chỉ chiếu điểm bằng calib KHÔNG bù để đo sai lệch.
(Nếu lấy label từ bản không bù thì box cũng lệch theo điểm, metric sẽ không thấy lỗi.)

Metric cho mỗi frame:
  dt_ms          t_camera - t_lidar
  ego_shift_m    quãng xe đi được giữa 2 thời điểm (độ dời của calib không bù so với calib đúng)
  shift_px_near / shift_px_far   median độ dịch pixel của điểm < 15 m / >= 30 m
  inbox_ok / inbox_noego         % điểm object rơi vào 2D box khi có / không bù chuyển động

    python -m src.fail_time_sync
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.calib_sweep import DEFAULT_CLASSES, points_in_box3d
from starter.datasets import list_frames, load_frame
from starter.projection import cam_to_image, draw_box2d, overlay_points, velo_to_cam


def project_full(xyz, calib, shape):
    """uv (N, 2) cho mọi điểm, NaN với điểm không chiếu được."""
    uv, _, mask = cam_to_image(velo_to_cam(xyz, calib), calib.P2, shape)
    full = np.full((len(xyz), 2), np.nan)
    full[mask] = uv
    return full


def analyse(data_root: str, fid: str) -> dict:
    ok = load_frame(data_root, fid, use_ego_motion=True)
    bad = load_frame(data_root, fid, use_ego_motion=False)
    xyz = ok["points"][:, :3].astype(np.float64)
    shape = ok["image"].shape
    cam_ok = velo_to_cam(xyz, ok["calib"])
    uv_ok, uv_bad = project_full(xyz, ok["calib"], shape), project_full(xyz, bad["calib"], shape)

    both = np.isfinite(uv_ok).all(1) & np.isfinite(uv_bad).all(1)
    shift = np.linalg.norm(uv_ok - uv_bad, axis=1)
    depth = cam_ok[:, 2]

    hit_ok = hit_bad = n = 0
    for obj in ok["labels"]:
        if obj.type not in DEFAULT_CLASSES:
            continue
        idx = points_in_box3d(cam_ok, obj) & np.isfinite(uv_ok).all(1)
        if idx.sum() < 10:
            continue
        x1, y1, x2, y2 = obj.bbox
        for uv, acc in ((uv_ok, "ok"), (uv_bad, "bad")):
            p = uv[idx]
            h = int(((p[:, 0] >= x1) & (p[:, 0] <= x2) & (p[:, 1] >= y1) & (p[:, 1] <= y2)).sum())
            if acc == "ok":
                hit_ok += h
            else:
                hit_bad += h
        n += int(idx.sum())

    T_ok, T_bad = np.eye(4), np.eye(4)
    T_ok[:3], T_bad[:3] = ok["calib"].Tr_velo_to_cam, bad["calib"].Tr_velo_to_cam
    ego_shift = float(np.linalg.norm((T_ok @ np.linalg.inv(T_bad))[:3, 3]))
    dt_ms = (ok["timestamp_camera_us"] - ok["timestamp_lidar_us"]) / 1000
    return {"frame": fid, "dt_ms": round(dt_ms, 1), "ego_shift_m": round(ego_shift, 3),
            "speed_mps": round(ego_shift / abs(dt_ms) * 1000, 2) if dt_ms else np.nan,
            "n_in_image_ok": int(np.isfinite(uv_ok).all(1).sum()), "n_in_image_noego": int(np.isfinite(uv_bad).all(1).sum()),
            "shift_px_near": round(float(np.median(shift[both & (depth < 15)])), 2) if (both & (depth < 15)).any() else np.nan,
            "shift_px_far": round(float(np.median(shift[both & (depth >= 30)])), 2) if (both & (depth >= 30)).any() else np.nan,
            "obj_points": n,
            "inbox_ok": round(100 * hit_ok / n, 2) if n else np.nan,
            "inbox_noego": round(100 * hit_bad / n, 2) if n else np.nan}


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def figure(data_root: str, row: pd.Series, out: Path) -> None:
    """Ảnh đúng (có bù) / sai (không bù) cạnh nhau, cùng 2D box label đúng; khoanh object lệch nhiều nhất."""
    fid = row.frame
    ok = load_frame(data_root, fid, use_ego_motion=True)
    bad = load_frame(data_root, fid, use_ego_motion=False)
    xyz = ok["points"][:, :3]
    cam_ok = velo_to_cam(xyz, ok["calib"])
    uv_ok = project_full(xyz, ok["calib"], ok["image"].shape)
    uv_bad = project_full(xyz, bad["calib"], ok["image"].shape)

    # object mất nhiều điểm khỏi box nhất khi không bù -> khoanh đỏ trên cả 2 ảnh.
    # Cùng định nghĩa với analyse(): điểm bị đẩy ra ngoài ảnh (uv_bad = NaN) tính là rơi khỏi box.
    worst, worst_drop = None, -1.0
    for obj in ok["labels"]:
        idx = points_in_box3d(cam_ok, obj) & np.isfinite(uv_ok).all(1)
        if obj.type not in DEFAULT_CLASSES or idx.sum() < 10:
            continue
        x1, y1, x2, y2 = obj.bbox
        p = uv_bad[idx]
        drop = 1 - ((p[:, 0] >= x1) & (p[:, 0] <= x2) & (p[:, 1] >= y1) & (p[:, 1] <= y2)).mean()
        if drop > worst_drop:
            worst, worst_drop = obj, drop
    panels = []
    for fr, tag in ((ok, "OK: ego-motion compensated"), (bad, "FAIL: --ignore-ego-motion")):
        uv, depth, _ = cam_to_image(velo_to_cam(xyz, fr["calib"]), fr["calib"].P2, ok["image"].shape)
        vis = overlay_points(ok["image"], uv, depth, radius=3)
        for obj in ok["labels"]:
            if obj.type in DEFAULT_CLASSES and points_in_box3d(cam_ok, obj).sum() >= 10:
                vis = draw_box2d(vis, obj.bbox, label=obj.type)
        if worst is not None:
            x1, y1, x2, y2 = worst.bbox
            dist = float(np.hypot(worst.location[0], worst.location[2]))
            cv2.rectangle(vis, (int(x1) - 15, int(y1) - 15), (int(x2) + 15, int(y2) + 15), (0, 0, 255), 4)
            cv2.putText(vis, f"{worst.type} {dist:.1f} m: {100 * worst_drop:.0f}% pts out of box w/o comp.",
                        (max(int(x1), 5), int(y2) + 50 if y2 + 60 < vis.shape[0] else max(int(y1) - 30, 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        txt =(f"{fid} {tag} | dt={row.dt_ms:.0f} ms, ego moved {row.ego_shift_m:.2f} m | "
               f"inbox {row.inbox_ok if fr is ok else row.inbox_noego:.1f}%")
        panels.append(label(cv2.resize(vis, (1200, 675)), txt))
    cv2.imwrite(str(out), np.vstack(panels))


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo lỗi Time: chiếu nuScenes có / không bù chuyển động xe")
    ap.add_argument("--data-root", default="data/nuscenes_mini_subset")
    ap.add_argument("--out", default="results/time_sync_nusc.csv")
    ap.add_argument("--fig", default="results/figures/fail_04_nusc_no_ego_motion.png")
    args = ap.parse_args()

    df = pd.DataFrame([analyse(args.data_root, f) for f in list_frames(args.data_root)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    df["scene"] = df.frame.str[:10]
    print(df.groupby("scene")[["dt_ms", "speed_mps", "shift_px_near", "shift_px_far", "inbox_ok", "inbox_noego"]]
          .median().round(2).to_string())
    worst = df.loc[(df.inbox_ok - df.inbox_noego).idxmax()]
    print("frame tệ nhất:\n" + worst.to_string())
    figure(args.data_root, worst, Path(args.fig))
    print(f"-> {args.out}, {args.fig}")


if __name__ == "__main__":
    main()
