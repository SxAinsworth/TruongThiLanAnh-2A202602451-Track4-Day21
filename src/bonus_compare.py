"""[B1] So sánh 2 cách phát hiện calibration drift trên cùng 20 frame KITTI, cùng metric "% frame phát hiện":

  (1) inbox: % điểm LiDAR của object rơi vào 2D box label (cần label -> chỉ dùng được cho QA offline)
  (2) edge : kiểm tra cực đại cục bộ của edge-alignment score (không cần label -> dùng được khi xe chạy)

Ngưỡng của cả hai đặt sao cho 0 báo động giả trên 20 frame chưa perturb:
  inbox: báo drift nếu inbox của frame < min inbox của các frame baseline
  edge : báo drift nếu gain > max gain của các frame baseline (đã tính trong src/drift_detect.py)

Đọc CSV của src.calib_sweep và src.drift_detect, nên phải chạy 2 script đó trước.
    python -m src.bonus_compare
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def inbox_detection(objects_csv: Path, axes: list[str]) -> pd.DataFrame:
    o = pd.read_csv(objects_csv)
    o["hit"] = o.inbox_pct * o.n_pts / 100
    per_frame = o.groupby(["frame", "param", "value"]).agg(hit=("hit", "sum"), n=("n_pts", "sum")).reset_index()
    per_frame["inbox"] = 100 * per_frame.hit / per_frame.n
    base = per_frame[per_frame.param == "none"]
    thr = base.inbox.min()
    rows = []
    for axis in axes:
        rows.append({"axis": axis, "drift": 0.0, "inbox_detect_rate": float((base.inbox < thr).mean())})
        for v, g in per_frame[per_frame.param == axis].groupby("value"):
            rows.append({"axis": axis, "drift": v, "inbox_detect_rate": float((g.inbox < thr).mean())})
    out = pd.DataFrame(rows)
    out["inbox_threshold_pct"] = round(thr, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="[B1] So sánh phát hiện drift: inbox (cần label) và edge score (không cần label)")
    ap.add_argument("--objects-csv", default="results/calib_sweep_kitti_objects.csv", help="output theo object của src.calib_sweep")
    ap.add_argument("--edge-csv", default="results/drift_detect_kitti_summary.csv", help="output tổng hợp của src.drift_detect")
    ap.add_argument("--out", default="results/bonus_b1_inbox_vs_edge.csv")
    ap.add_argument("--fig", default="results/figures/bonus_b1_inbox_vs_edge.png")
    args = ap.parse_args()

    edge = pd.read_csv(args.edge_csv).rename(columns={"detect_rate": "edge_detect_rate", "threshold_gain": "edge_threshold_gain"})
    axes = sorted(edge.axis.unique())
    df = inbox_detection(Path(args.objects_csv), axes).merge(
        edge[["axis", "drift", "edge_detect_rate", "edge_threshold_gain"]], on=["axis", "drift"])
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))

    fig, axs = plt.subplots(1, len(axes), figsize=(5.5 * len(axes), 4), sharey=True)
    for ax, axis in zip(axs, axes):
        g = df[df.axis == axis]
        ax.plot(g.drift, 100 * g.inbox_detect_rate, "o-", label="inbox (cần label)")
        ax.plot(g.drift, 100 * g.edge_detect_rate, "s--", label="edge score (không cần label)")
        ax.set_title(f"Drift {axis}")
        ax.set_xlabel("drift thật (°)")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("% frame phát hiện (KITTI, 20 frame)")
    axs[0].legend()
    fig.suptitle("[B1] 2 cách phát hiện calibration drift, ngưỡng = 0 báo động giả trên frame chưa lệch")
    fig.tight_layout()
    fig.savefig(args.fig, dpi=120)
    print(f"-> {args.out}, {args.fig}")


if __name__ == "__main__":
    main()
