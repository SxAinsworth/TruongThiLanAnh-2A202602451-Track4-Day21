"""Topic A: đo độ nhạy của LiDAR-camera projection với calibration drift.

Mỗi cấu hình chỉ thay đổi MỘT tham số extrinsic (yaw / pitch / roll / tx / ty / tz),
mọi thứ khác (frame, class, ngưỡng) giữ nguyên. Không có phép ngẫu nhiên nào,
nên chạy lại luôn ra cùng số liệu.

Metric cho mỗi cấu hình:
  fov_ratio        số điểm chiếu vào ảnh / số điểm chiếu vào ảnh khi chưa perturb
  inbox_<bin>      % điểm LiDAR của object (nằm trong 3D box GT và trong ảnh ở baseline) rơi vào 2D box label,
                   gộp theo khoảng cách object: near < 15 m, mid 15-30 m, far >= 30 m
  edge_score       % điểm "biên độ sâu" nằm cách cạnh Canny của ảnh <= EDGE_TOL_PX pixel
  edge_ratio       edge_score / edge_score khi chưa perturb (trung bình theo frame)

Ví dụ:
    python -m src.calib_sweep --data-root data/kitti_mini --out results/calib_sweep_kitti.csv
    python -m src.calib_sweep --data-root data/nuscenes_mini_subset --frame-step 4 \
        --out results/calib_sweep_nusc.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_frame
from starter.projection import cam_to_image, perturb_extrinsic, velo_to_cam

DIST_BINS = [("near", 0.0, 15.0), ("mid", 15.0, 30.0), ("far", 30.0, np.inf)]
DEFAULT_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Cyclist", "Bicycle"]
EDGE_TOL_PX = 3.0
DEPTH_JUMP_M = 1.0
MIN_OBJ_POINTS = 10
EDGE_WINDOW_PX = 7    # 64 beam (KITTI). LiDAR thưa hơn cần cửa sổ lớn hơn, xem --edge-window

# (tham số, các mức). Mức 0 là baseline, chung cho mọi tham số.
DEFAULT_SWEEP = {
    "yaw_deg": [0.5, 1.0, 2.0, 3.0],
    "pitch_deg": [0.5, 1.0, 2.0, 3.0],
    "roll_deg": [0.5, 1.0, 2.0, 3.0],
    "tx_m": [0.02, 0.05, 0.10],
    "ty_m": [0.02, 0.05, 0.10],
    "tz_m": [0.02, 0.05, 0.10],
}


def make_calib(calib, param: str, value: float):
    kw = {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}
    t = [0.0, 0.0, 0.0]
    if param in kw:
        kw[param] = value
    elif param in ("tx_m", "ty_m", "tz_m"):
        t["xyz".index(param[1])] = value
    elif param != "none":
        raise ValueError(param)
    return perturb_extrinsic(calib, t_xyz_m=tuple(t), **kw)


def points_in_box3d(pts_cam: np.ndarray, obj) -> np.ndarray:
    """Mask điểm (camera frame) nằm trong 3D box KITTI (location = tâm đáy, quay quanh trục y)."""
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (pts_cam - obj.location) @ R          # = R^T (p - loc) cho từng hàng
    return (np.abs(local[:, 0]) <= l / 2) & (local[:, 1] <= 0) & (local[:, 1] >= -h) \
        & (np.abs(local[:, 2]) <= w / 2)


def image_edge_distance(image: np.ndarray) -> np.ndarray:
    """Bản đồ khoảng cách (pixel) tới cạnh Canny gần nhất."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)


def edge_alignment(uv: np.ndarray, depth: np.ndarray, shape, edge_dist: np.ndarray,
                   window: int = EDGE_WINDOW_PX) -> tuple[float, int]:
    """Điểm biên độ sâu: điểm gần mà trong lân cận window x window pixel có điểm xa hơn >= DEPTH_JUMP_M.
    Trả về (% điểm biên nằm sát cạnh ảnh, số điểm biên). Không cần label."""
    H, W = shape[:2]
    u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
    dmap = np.zeros((H, W), np.float32)
    order = np.argsort(-depth)                     # ghi điểm xa trước, điểm gần đè lên
    dmap[v[order], u[order]] = depth[order]
    far_nb = cv2.dilate(dmap, np.ones((window, window), np.uint8))
    is_edge = (far_nb[v, u] - depth) >= DEPTH_JUMP_M
    n = int(is_edge.sum())
    if n == 0:
        return float("nan"), 0
    return float((edge_dist[v[is_edge], u[is_edge]] <= EDGE_TOL_PX).mean() * 100), n


def evaluate(frames: list[dict], param: str, value: float, edge_window: int = EDGE_WINDOW_PX):
    """Trả về (dict metric tổng hợp, list dòng theo object, list dòng theo frame)."""
    fov_ratio, edge_scores, edge_ratios, frame_rows = [], [], [], []
    inbox = {b: [0, 0] for b, _, _ in DIST_BINS}   # [số điểm trong 2D box, tổng điểm object]
    obj_rows = []
    for f in frames:
        calib = make_calib(f["calib"], param, value)
        pts_cam = velo_to_cam(f["xyz"], calib)
        uv, depth, mask = cam_to_image(pts_cam, calib.P2, f["image"].shape)
        fov_ratio.append(mask.sum() / max(f["n_fov0"], 1))
        es, n_edge = edge_alignment(uv, depth, f["image"].shape, f["edge_dist"], edge_window)
        edge_scores.append(es)
        edge_ratios.append(es / f["edge0"] if f["edge0"] else np.nan)
        frame_rows.append({"frame": f["frame_id"], "param": param, "value": value,
                           "fov_ratio": round(fov_ratio[-1], 4), "edge_score": round(es, 2), "n_edge": n_edge})

        # chiếu riêng điểm của từng object (đã xác định bằng calib gốc)
        uv_full = np.full((len(mask), 2), np.nan)
        uv_full[mask] = uv
        for k, (obj, idx) in enumerate(f["objects"]):
            p = uv_full[idx]
            x1, y1, x2, y2 = obj.bbox
            hit = int(((p[:, 0] >= x1) & (p[:, 0] <= x2) & (p[:, 1] >= y1) & (p[:, 1] <= y2)).sum())
            dist = float(np.hypot(obj.location[0], obj.location[2]))
            b = next(name for name, lo, hi in DIST_BINS if lo <= dist < hi)
            inbox[b][0] += hit
            inbox[b][1] += len(idx)
            obj_rows.append({"frame": f["frame_id"], "obj": k, "type": obj.type, "dist_m": round(dist, 2),
                             "bin": b, "n_pts": len(idx), "param": param, "value": value,
                             "inbox_pct": round(100 * hit / len(idx), 2)})
    row = {"param": param, "value": value, "n_frames": len(frames),
           "fov_ratio": round(float(np.mean(fov_ratio)), 4),
           "edge_score": round(float(np.nanmean(edge_scores)), 2),
           "edge_ratio": round(float(np.nanmean(edge_ratios)), 4)}
    for b, (hit, tot) in inbox.items():
        row[f"inbox_{b}"] = round(100 * hit / tot, 2) if tot else np.nan
        row[f"npts_{b}"] = tot
    return row, obj_rows, frame_rows


def prepare(data_root: str, frame_ids: list[str], classes: list[str], fov_only: bool = True,
            edge_window: int = EDGE_WINDOW_PX) -> list[dict]:
    """Đọc frame và tính sẵn những thứ không phụ thuộc perturb (điểm của object, cạnh ảnh)."""
    frames = []
    for fid in frame_ids:
        fr = load_frame(data_root, fid)
        xyz = fr["points"][:, :3].astype(np.float64)
        calib = fr["calib"]
        pts_cam = velo_to_cam(xyz, calib)
        uv, depth, mask = cam_to_image(pts_cam, calib.P2, fr["image"].shape)
        objects = []
        for obj in fr["labels"]:
            if obj.type not in classes:
                continue
            in_box = points_in_box3d(pts_cam, obj) & np.isfinite(pts_cam).all(axis=1)
            # Chỉ giữ điểm object đã nằm trong ảnh ở baseline: object bị cắt ở rìa ảnh (truncated)
            # có phần lớn điểm ngoài FOV, nếu tính vào mẫu số thì inbox thấp dù calib đúng
            # (xem src/fail_truncation.py).
            idx = np.flatnonzero(in_box & mask) if fov_only else np.flatnonzero(in_box)
            if len(idx) >= MIN_OBJ_POINTS:
                objects.append((obj, idx))
        edge_dist = image_edge_distance(fr["image"])
        edge0, _ = edge_alignment(uv, depth, fr["image"].shape, edge_dist, edge_window)
        frames.append({"frame_id": fid, "xyz": xyz, "calib": calib, "image": fr["image"],
                       "objects": objects, "n_fov0": int(mask.sum()),
                       "edge_dist": edge_dist, "edge0": edge0})
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep calibration drift (mỗi lần 1 tham số) và đo mismatch của projection")
    ap.add_argument("--data-root", default="data/kitti_mini", help="data/kitti_mini, data/nuscenes_mini_subset, data/synthetic")
    ap.add_argument("--frames", nargs="*", help="danh sách frame id; mặc định dùng tất cả")
    ap.add_argument("--frame-step", type=int, default=1, help="lấy 1 frame mỗi N frame (nuScenes: 4 -> 20 frame)")
    ap.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES, help="class object dùng để đo inbox")
    ap.add_argument("--params", nargs="*", default=list(DEFAULT_SWEEP), choices=list(DEFAULT_SWEEP),
                    help="các tham số cần sweep")
    ap.add_argument("--edge-window", type=int, default=EDGE_WINDOW_PX,
                    help="cửa sổ (pixel) tìm điểm xa hơn khi xác định biên độ sâu; KITTI 64 beam: 7")
    ap.add_argument("--all-object-points", action="store_true",
                    help="tính cả điểm object nằm ngoài ảnh ở baseline vào mẫu số (cách đo cũ, bị sai với object truncated)")
    ap.add_argument("--out", default="results/calib_sweep.csv",
                    help="CSV tổng hợp; CSV theo object / theo frame lưu cạnh với hậu tố _objects / _frames")
    args = ap.parse_args()

    frame_ids = args.frames or list_frames(args.data_root)[:: args.frame_step]
    t0 = time.perf_counter()
    frames = prepare(args.data_root, frame_ids, args.classes, fov_only=not args.all_object_points,
                     edge_window=args.edge_window)
    n_obj = sum(len(f["objects"]) for f in frames)
    print(f"{args.data_root}: {len(frames)} frame, {n_obj} object có >= {MIN_OBJ_POINTS} điểm")

    rows, obj_rows, frame_rows = [], [], []
    configs = [("none", 0.0)] + [(p, v) for p in args.params for v in DEFAULT_SWEEP[p]]
    for param, value in configs:
        row, objs, frs = evaluate(frames, param, value, args.edge_window)
        rows.append(row)
        obj_rows += objs
        frame_rows += frs
        print(f"{param:>9}={value:<5} fov={row['fov_ratio']:.3f} inbox near/mid/far="
              f"{row['inbox_near']:.1f}/{row['inbox_mid']:.1f}/{row['inbox_far']:.1f}% "
              f"edge={row['edge_score']:.1f} ({row['edge_ratio']:.3f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(obj_rows).to_csv(out.with_name(out.stem + "_objects.csv"), index=False)
    pd.DataFrame(frame_rows).to_csv(out.with_name(out.stem + "_frames.csv"), index=False)
    print(f"-> {out} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
