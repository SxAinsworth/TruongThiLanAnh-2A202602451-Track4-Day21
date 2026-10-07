"""Vẽ toàn bộ hình cho REPORT từ các CSV trong results/ (chạy sau calib_sweep và drift_detect).

    python -m src.make_figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calib_sweep import EDGE_TOL_PX, DEPTH_JUMP_M, EDGE_WINDOW_PX, image_edge_distance, make_calib, points_in_box3d
from starter.datasets import load_frame
from starter.projection import cam_to_image, draw_box2d, overlay_points, velo_to_cam

BIN_STYLE = {"near": ("tab:green", "near < 15 m"), "mid": ("tab:orange", "mid 15-30 m"), "far": ("tab:red", "far ≥ 30 m")}
PARAM_LABEL = {"yaw_deg": "yaw (°)", "pitch_deg": "pitch (°)", "roll_deg": "roll (°)",
               "tx_m": "tx (m)", "ty_m": "ty (m)", "tz_m": "tz (m)"}


def sweep_curve(df: pd.DataFrame, param: str) -> pd.DataFrame:
    base = df[df.param == "none"].assign(param=param)
    return pd.concat([base, df[df.param == param]]).sort_values("value")


def plot_inbox(res: Path, fig_dir: Path) -> None:
    k = pd.read_csv(res / "calib_sweep_kitti.csv")
    n = pd.read_csv(res / "calib_sweep_nusc.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, param in zip(axes.flat, PARAM_LABEL):
        ck, cn = sweep_curve(k, param), sweep_curve(n, param)
        for b, (color, label) in BIN_STYLE.items():
            ax.plot(ck.value, ck[f"inbox_{b}"], "o-", color=color, label=f"KITTI {label}")
            ax.plot(cn.value, cn[f"inbox_{b}"], "x--", color=color, alpha=0.6, label=f"nuScenes {label}")
        ax.set_xlabel(PARAM_LABEL[param])
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("% điểm object nằm trong 2D box")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Calibration drift → % điểm LiDAR của object rơi vào 2D box label (1 tham số/lần)\n"
                 "Lưu ý: perturb_extrinsic giả định trục LiDAR của KITTI; với nuScenes pitch/roll và tx/ty bị đảo (xem fail_02)")
    fig.tight_layout()
    fig.savefig(fig_dir / "sweep_inbox_kitti_nusc.png", dpi=110)
    plt.close(fig)


def plot_class(res: Path, fig_dir: Path) -> pd.DataFrame:
    """Bảng + hình: theo class (Car/Van/Truck vs Pedestrian/Cyclist) với yaw."""
    o = pd.read_csv(res / "calib_sweep_kitti_objects.csv")
    o = o[o.param.isin(["none", "yaw_deg"])].copy()
    o["group"] = np.where(o.type.isin(["Pedestrian", "Cyclist"]), "Pedestrian/Cyclist", "Car/Van/Truck")
    o["hit"] = o.inbox_pct * o.n_pts / 100
    t = o.groupby(["group", "bin", "value"]).agg(hit=("hit", "sum"), n=("n_pts", "sum"), n_obj=("obj", "size"))
    t["inbox"] = (100 * t.hit / t.n).round(1)
    tab = t.inbox.unstack("value")
    tab["n_obj"] = t.n_obj.unstack("value").iloc[:, 0]
    tab.to_csv(res / "yaw_inbox_by_class_kitti.csv")
    return tab


def plot_edge(res: Path, fig_dir: Path) -> None:
    k = pd.read_csv(res / "calib_sweep_kitti.csv")
    n = pd.read_csv(res / "calib_sweep_nusc.csv")
    det = pd.read_csv(res / "drift_detect_kitti_summary.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for param, marker in [("yaw_deg", "o"), ("pitch_deg", "s"), ("roll_deg", "^")]:
        ck, cn = sweep_curve(k, param), sweep_curve(n, param)
        a1.plot(ck.value, ck.edge_ratio, marker + "-", label=f"KITTI {param}")
        a1.plot(cn.value, cn.edge_ratio, marker + "--", alpha=0.6, label=f"nuScenes {param}")
    a1.axhline(1, color="gray", lw=0.8)
    a1.set_xlabel("mức drift (°)")
    a1.set_ylabel("edge score / edge score baseline (TB 20 frame)")
    a1.set_title("Edge-alignment score theo drift")
    a1.legend(fontsize=7)
    a1.grid(alpha=0.3)
    for axis, g in det.groupby("axis"):
        a2.plot(g.drift, 100 * g.detect_rate, "o-", label=f"{axis} (ngưỡng gain > {g.threshold_gain.iloc[0]:.2f})")
    a2.set_xlabel("drift thật (°)")
    a2.set_ylabel("% frame phát hiện drift")
    a2.set_title("Phát hiện drift bằng kiểm tra cực đại cục bộ (KITTI, 20 frame)")
    a2.set_ylim(-5, 105)
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "edge_score_drift_detect.png", dpi=110)
    plt.close(fig)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 12 * len(text) + 20), 32), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def render(fr: dict, calib, boxes: bool = True, max_depth: float = 50.0) -> np.ndarray:
    uv, depth, _ = cam_to_image(velo_to_cam(fr["points"][:, :3], calib), calib.P2, fr["image"].shape)
    vis = overlay_points(fr["image"], uv, depth, max_depth=max_depth)
    if boxes:
        for obj in fr["labels"]:
            vis = draw_box2d(vis, obj.bbox, label=obj.type)
    return vis


def demo_yaw(fig_dir: Path) -> None:
    """Demo: cùng frame 000011, yaw 0 / 1 / 2 độ."""
    fr = load_frame("data/kitti_mini", "000011")
    rows = [label(render(fr, make_calib(fr["calib"], "yaw_deg", v)), f"KITTI 000011  yaw drift {v:g} deg")
            for v in [0.0, 1.0, 2.0]]
    cv2.imwrite(str(fig_dir / "demo_yaw_drift_000011.png"), np.vstack(rows))


def fail_truncation(fig_dir: Path) -> None:
    """fail_01: xe sát camera bị cắt ở rìa ảnh -> phần lớn điểm của 3D box nằm ngoài FOV."""
    fr = load_frame("data/kitti_mini", "000011")
    calib = fr["calib"]
    pts_cam = velo_to_cam(fr["points"][:, :3], calib)
    _, _, mask = cam_to_image(pts_cam, calib.P2, fr["image"].shape)
    obj = min((o for o in fr["labels"] if o.type == "Car"), key=lambda o: o.bbox[0])   # xe sát mép trái
    inb = points_in_box3d(pts_cam, obj) & np.isfinite(pts_cam).all(axis=1)
    n_in, n_tot = int((inb & mask).sum()), int(inb.sum())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
    a1.imshow(cv2.cvtColor(render(fr, calib), cv2.COLOR_BGR2RGB))
    a1.set_title(f"Ảnh: chỉ thấy {n_in}/{n_tot} điểm ({100 * n_in / n_tot:.1f}%) của xe trong box (mép trái)")
    a1.axis("off")
    near = np.isfinite(pts_cam).all(axis=1) & (pts_cam[:, 2] > -5) & (pts_cam[:, 2] < 20) & (np.abs(pts_cam[:, 0]) < 15)
    a2.scatter(pts_cam[near, 0], pts_cam[near, 2], s=0.3, c="lightgray")
    a2.scatter(pts_cam[inb & ~mask, 0], pts_cam[inb & ~mask, 2], s=2, c="red", label="điểm xe NGOÀI ảnh")
    a2.scatter(pts_cam[inb & mask, 0], pts_cam[inb & mask, 2], s=2, c="green", label="điểm xe trong ảnh")
    fx, cx, W = calib.P2[0, 0], calib.P2[0, 2], fr["image"].shape[1]
    z = np.array([0, 20])
    a2.plot(z * (0 - cx) / fx, z, "b--", lw=1, label="biên FOV camera")
    a2.plot(z * (W - cx) / fx, z, "b--", lw=1)
    a2.set_xlabel("x_cam (m, phải)")
    a2.set_ylabel("z_cam (m, trước)")
    a2.set_aspect("equal")
    a2.set_xlim(-15, 15)
    a2.set_ylim(-5, 20)
    a2.legend(fontsize=8, loc="upper right")
    a2.set_title(f"BEV: xe {obj.location[2]:.1f} m, truncated={obj.truncated:.2f}")
    fig.suptitle("fail_01 [Metric]: đếm mọi điểm trong 3D box làm mẫu số -> inbox = "
                 f"{100 * n_in / n_tot:.1f}% dù calibration ĐÚNG")
    fig.tight_layout()
    fig.savefig(fig_dir / "fail_01_truncated_near_car_metric.png", dpi=110)
    plt.close(fig)


def fail_axes(fig_dir: Path) -> None:
    """fail_02: perturb_extrinsic giả định trục KITTI; với nuScenes 'pitch' thực ra là roll và ngược lại."""
    k = load_frame("data/kitti_mini", "000011")
    n = load_frame("data/nuscenes_mini_subset", "scene-0103_010")
    tiles = []
    for fr, name in [(k, "KITTI 000011"), (n, "nuScenes scene-0103_010")]:
        row = []
        for param in ["pitch_deg", "roll_deg"]:
            img = render(fr, make_calib(fr["calib"], param, 2.0), boxes=False)
            img = cv2.resize(img, (900, int(900 * img.shape[0] / img.shape[1])))
            row.append(label(img, f"{name}  {param}=2"))
        h = max(r.shape[0] for r in row)
        tiles.append(np.hstack([cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 4, cv2.BORDER_CONSTANT) for r in row]))
    w = max(t.shape[1] for t in tiles)
    grid = np.vstack([cv2.copyMakeBorder(t, 0, 4, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles])
    cv2.imwrite(str(fig_dir / "fail_02_nuscenes_pitch_roll_axes_swapped.png"), grid)


def fail_pitch_false_max(fig_dir: Path, window: int = EDGE_WINDOW_PX) -> None:
    """fail_03: frame 000011, calib ĐÚNG nhưng score lại cao hơn khi xoay pitch -3 độ."""
    fr = load_frame("data/kitti_mini", "000011")
    edist = image_edge_distance(fr["image"])
    gray = cv2.cvtColor(fr["image"], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    panels = []
    for v in [0.0, -3.0]:
        calib = make_calib(fr["calib"], "pitch_deg", v)
        uv, depth, _ = cam_to_image(velo_to_cam(fr["points"][:, :3], calib), calib.P2, fr["image"].shape)
        H, W = gray.shape
        u, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        dmap = np.zeros((H, W), np.float32)
        o = np.argsort(-depth)
        dmap[vv[o], u[o]] = depth[o]
        is_edge = (cv2.dilate(dmap, np.ones((window, window), np.uint8))[vv, u] - depth) >= DEPTH_JUMP_M
        hit = edist[vv, u] <= EDGE_TOL_PX
        vis = cv2.cvtColor((gray * 0.5).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        vis[edges > 0] = (200, 200, 200)
        for good, color in [(False, (0, 0, 255)), (True, (0, 255, 0))]:
            sel = is_edge & (hit == good)
            for x, y in zip(u[sel], vv[sel]):
                cv2.circle(vis, (int(x), int(y)), 2, color, -1)
        sc = 100 * (is_edge & hit).sum() / max(is_edge.sum(), 1)
        tag = "calib DUNG" if v == 0 else "calib SAI (pitch -3 deg)"
        panels.append(label(vis, f"{tag}: edge score = {sc:.1f}%  (xanh = diem bien sat canh Canny, do = lech)"))
    cv2.imwrite(str(fig_dir / "fail_03_pitch_edge_score_false_max_000011.png"), np.vstack(panels))


def main() -> None:
    ap = argparse.ArgumentParser(description="Vẽ hình cho REPORT từ results/*.csv")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_inbox(res, fig_dir)
    print(plot_class(res, fig_dir).to_string())
    plot_edge(res, fig_dir)
    demo_yaw(fig_dir)
    fail_truncation(fig_dir)
    fail_axes(fig_dir)
    fail_pitch_false_max(fig_dir)
    print(f"-> {fig_dir}")


if __name__ == "__main__":
    main()
