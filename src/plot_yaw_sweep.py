"""Vẽ kết quả quét yaw (script mẫu codelab). Chạy từ gốc repo: python -m src.plot_yaw_sweep"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/yaw_perturb_sweep.csv", dtype={"frame": str})   # giữ "000011", không đổi thành 11

fig, ax = plt.subplots(figsize=(6, 4))
for frame, g in df.groupby("frame"):
    ax.plot(g["yaw_deg"], 100 * g["hit_ratio"], marker="o", label=f"frame {frame}")
ax.set_xlabel("Lệch yaw (độ)")
ax.set_ylabel("% điểm của vật thể nằm trong 2D box")
ax.set_ylim(0, 105)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

out = Path("results/figures/yaw_sweep.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"-> {out}")
