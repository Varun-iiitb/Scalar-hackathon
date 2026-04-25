"""
DubGuard result visualiser.

Reads training/logs/training_log.csv and produces:
  training/plots/reward_curves.png — all reward components over training steps

Run standalone:
  python plot_results.py
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH  = PROJECT_ROOT / "training" / "logs" / "training_log.csv"
PLOTS_DIR = PROJECT_ROOT / "training" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    print(f"ERROR: {CSV_PATH} not found. Run training first.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} log entries.")
print(f"Steps: {df['global_step'].min()} to {df['global_step'].max()}")
print(f"Columns: {list(df.columns)}")

WINDOW = 20

def smooth(series):
    return series.rolling(window=WINDOW, min_periods=1).mean()

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle("DubGuard — Full Training Report", fontsize=16, fontweight="bold", y=0.98)

# ── PLOT 1: All reward components ──
ax1 = axes[0]
ax1.plot(df["global_step"], smooth(df["detection_score"]),
         color="#2196F3", label="Detection (40%)", linewidth=2)
ax1.plot(df["global_step"], smooth(df["severity_score"]),
         color="#4CAF50", label="Severity (20%)", linewidth=2)
ax1.plot(df["global_step"], smooth(df["correction_score"]),
         color="#FF9800", label="Correction (25%)", linewidth=2)
ax1.plot(df["global_step"], smooth(df["cultural_score"]),
         color="#9C27B0", label="Cultural (15%)", linewidth=2)

# Mark curriculum transitions by difficulty change
if "difficulty" in df.columns:
    prev = None
    for _, row in df.iterrows():
        if row["difficulty"] != prev and prev is not None:
            ax1.axvline(x=row["global_step"], color="gray",
                        linestyle="--", alpha=0.6, linewidth=1)
            ax1.text(row["global_step"] + 3, 0.97, row["difficulty"],
                     fontsize=8, color="gray", va="top",
                     transform=ax1.get_xaxis_transform())
        prev = row["difficulty"]

ax1.set_xlabel("Training Step")
ax1.set_ylabel("Score (0 to 1)")
ax1.set_title("Reward Components over Training")
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# ── PLOT 2: False-positive penalty ──
ax2 = axes[1]
ax2.fill_between(df["global_step"], smooth(df["false_positive_penalty"]),
                  alpha=0.25, color="#F44336")
ax2.plot(df["global_step"], smooth(df["false_positive_penalty"]),
         color="#F44336", label="False-positive penalty", linewidth=2)
ax2.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)

ax2.set_xlabel("Training Step")
ax2.set_ylabel("Penalty (≤ 0)")
ax2.set_title("False-Positive Penalty over Training\n(Should trend toward 0 as agent stops over-flagging)")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# ── PLOT 3: Combined reward ──
ax3 = axes[2]
ax3.plot(df["global_step"], smooth(df["combined_reward"]),
         color="#607D8B", label="Combined reward", linewidth=2)
ax3.axhline(y=0, color="black", linewidth=0.8, alpha=0.4)
ax3.axhline(y=0.5, color="#4CAF50", linestyle=":", linewidth=1.2,
            alpha=0.7, label="Target (0.5)")

ax3.set_xlabel("Training Step")
ax3.set_ylabel("Combined Reward")
ax3.set_title("Overall Combined Reward over Training")
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = PLOTS_DIR / "reward_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# Summary stats
print("\n=== FINAL TRAINING STATS (last 50 steps) ===")
last = df.tail(50)
print(f"  Detection:   {last['detection_score'].mean():.3f}")
print(f"  Severity:    {last['severity_score'].mean():.3f}")
print(f"  Correction:  {last['correction_score'].mean():.3f}")
print(f"  Cultural:    {last['cultural_score'].mean():.3f}")
print(f"  FP penalty:  {last['false_positive_penalty'].mean():.3f}")
print(f"  Combined:    {last['combined_reward'].mean():.3f}")
if "parse_failed" in df.columns:
    parse_rate = last["parse_failed"].mean() * 100
    print(f"  Parse fail:  {parse_rate:.1f}%")
