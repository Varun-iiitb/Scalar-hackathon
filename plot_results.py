"""
IsoSync result visualiser.

Reads logs/rewards_log.csv and produces two plots:
  plots/reward_curves.png  — all 4 reward components over episodes (smoothed)
  plots/budget_deficit.png — cumulative budget deficit over episodes (smoothed)

Run standalone:
  python plot_results.py
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH     = PROJECT_ROOT / "logs" / "rewards_log.csv"
PLOTS_DIR    = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 20


def main():
    if not CSV_PATH.exists():
        print(f"No log file found at {CSV_PATH} — run train.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("Log file is empty.")
        return

    df["timing_smooth"]   = df["timing_reward"].rolling(WINDOW, min_periods=1).mean()
    df["semantic_smooth"] = df["semantic_reward"].rolling(WINDOW, min_periods=1).mean()
    df["budget_smooth"]   = df["budget_reward"].rolling(WINDOW, min_periods=1).mean()
    df["locale_smooth"]   = df["locale_reward"].rolling(WINDOW, min_periods=1).mean()
    df["deficit_smooth"]  = df["budget_deficit"].rolling(WINDOW, min_periods=1).mean()

    level_transitions = df.groupby("level")["episode"].min().to_dict()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # ── Plot 1: Reward components ────────────────────────────────────────────
    ax1.plot(df["episode"], df["timing_smooth"],   color="#2196F3", label="Timing",   linewidth=2)
    ax1.plot(df["episode"], df["semantic_smooth"], color="#4CAF50", label="Semantic", linewidth=2)
    ax1.plot(df["episode"], df["budget_smooth"],   color="#FF9800", label="Budget",   linewidth=2)
    ax1.plot(df["episode"], df["locale_smooth"],   color="#9C27B0", label="Locale",   linewidth=2)

    for level, ep in level_transitions.items():
        if level > 1:
            ax1.axvline(x=ep, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
            ax1.text(ep + 3, 0.95, f"Level {level}", color="gray", fontsize=9, va="top")

    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Reward (0 – 1)", fontsize=12)
    ax1.set_title("IsoSync — Reward Components over Training", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Budget deficit ───────────────────────────────────────────────
    ax2.fill_between(df["episode"], df["deficit_smooth"], alpha=0.2, color="#F44336")
    ax2.plot(df["episode"], df["deficit_smooth"], color="#F44336", label="Budget deficit (s)", linewidth=2)
    ax2.axhline(y=0.2, color="#4CAF50", linestyle=":", linewidth=1.5, label="Target threshold (0.2s)")
    ax2.axhline(y=2.0, color="#FF9800", linestyle=":", linewidth=1.5, label="Fail threshold (2.0s)")

    for level, ep in level_transitions.items():
        if level > 1:
            ax2.axvline(x=ep, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
            ax2.text(ep + 3, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] > 0 else 0.5,
                     f"Level {level}", color="gray", fontsize=9, va="top")

    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Cumulative deficit (seconds)", fontsize=12)
    ax2.set_title(
        "IsoSync — Budget Deficit over Training\n"
        "(Should trend toward 0 as agent learns tighter translations)",
        fontsize=13, fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    out1 = PLOTS_DIR / "reward_curves.png"
    out2 = PLOTS_DIR / "budget_deficit.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
