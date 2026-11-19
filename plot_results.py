"""Render a standalone figure for the prospective-supervisor proposal."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
results = json.loads((HERE / "confirmation.json").read_text())["endpoints"]

fig, (joint, marginal) = plt.subplots(1, 2, figsize=(7.1, 2.0), gridspec_kw={"width_ratios": [1, 1.28]})
plt.rcParams.update({"font.family": "DejaVu Sans"})
for ax in (joint, marginal):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, length=0, colors="#263544")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e7ecf0")
    ax.set_xlabel("Change in time high (percentage points)", fontsize=8, color="#263544")


def plot(ax, key, y, color):
    effect = results[key]
    value = 100 * effect["mean"]
    lo, hi = (100 * bound for bound in effect["interval"])
    ax.errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt="o", color=color,
                markersize=4.6, linewidth=1.3, capsize=3)


joint.set_title("Joint occupation", loc="left", fontsize=10, color="#142735", weight="bold")
joint.set_yticks([0], ["Both high"])
joint.set_ylim(-.48, .48)
joint.set_xlim(0, 36)
joint.set_xticks([0, 10, 20, 30])
plot(joint, "joint_high_change", 0, "#007a78")

marginal.set_title("Individual duty fractions", loc="left", fontsize=10, color="#142735", weight="bold")
marginal.axvspan(-.5, .5, color="#eef6f4", zorder=0)
marginal.axvline(0, color="#9aaab3", linewidth=.8)
marginal.set_yticks([1, 0], ["Relay 1", "Relay 2"])
marginal.set_ylim(-.55, 1.55)
marginal.set_xlim(-.59, .59)
marginal.set_xticks([-.5, -.25, 0, .25, .5])
plot(marginal, "relay1_duty_change", 1, "#007a78")
plot(marginal, "relay2_duty_change", 0, "#007a78")

fig.tight_layout(pad=.85, w_pad=2.0)
fig.savefig(HERE / "phase_effect.png", dpi=250, facecolor="white")
