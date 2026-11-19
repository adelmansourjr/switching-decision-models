"""Plot the exact counterexample and held-out policy gaps."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
witness = json.loads((HERE / "exact_witness.json").read_text())
report = json.loads((HERE / "confirmation.json").read_text())

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (w, p) = plt.subplots(1, 2, figsize=(7.1, 2.05))
w.set_title("Same mean thresholds, opposite choices", loc="left", fontsize=9, weight="bold")
w.axvline(0, color=".5", linewidth=.8)
effects = np.array(witness["normalized_action_gaps"]) * np.array([1, -1]) * 100
w.scatter(effects, [1, 0], color=["#007a78", "#b15e3b"], s=30, zorder=3)
w.set_yticks([1, 0], ["Prior A", "Prior B"])
w.set_ylim(-.6, 1.6)
w.set_xlim(-2.2, 1.6)
w.set_xlabel("Gain from action 1 over action 0 (points)", fontsize=8)
p.set_title("Held-out gap to the exact optimum", loc="left", fontsize=9, weight="bold")
names = ("mean", "full", next(key for key in report["payoff"] if key not in ("mean", "full", "exact")))
labels = ["Mean thresholds", "Full posterior", "Depth-8 planner"]
gaps = [100 * (report["payoff"]["exact"] - report["payoff"][name]) for name in names]
p.barh([2, 1, 0], gaps, color=["#889aa7", "#007a78", "#38536c"], height=.48)
p.set_yticks([2, 1, 0], labels)
p.set_xlabel("Average payoff loss (points, lower is better)", fontsize=8)
p.set_xlim(0, max(gaps) * 1.32)
for y, gap in zip([2, 1, 0], gaps):
    p.text(gap + max(gaps) * .025, y, f"{gap:.3f}", va="center", fontsize=8)
for ax in (w, p):
    ax.tick_params(labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=".92")
fig.tight_layout(w_pad=1.6)
fig.savefig(HERE / "equilibrium_result.png", dpi=260, bbox_inches="tight", facecolor="white")
