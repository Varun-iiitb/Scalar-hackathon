"""
IsoSync result visualiser.

Reads logs/rewards_log.csv and produces two plots:
  plots/reward_curves.png  — all 4 reward components over episodes
  plots/budget_deficit.png — cumulative budget deficit over episodes

Run standalone:
  python plot_results.py
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display needed on HF Spaces
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH     = PROJECT_ROOT / "logs" / "rewards_log.csv"
PLOTS_DIR    = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Curriculum transition episode numbers
LEVEL2_START = 200
LEVEL3_START = 500


def _smooth(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling mean for readability."""
    return series.rolling(window=window, min_periods=1).mean()


def main():
    if not CSV_PATH.exists():
        print(f"No log file found at {CSV_PATH} — run train.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("Log file is empty.")
        return

    ep = df["episode"]

    # ── Plot 1: All 4 reward components ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    components = [
        ("timing_reward",   "Timing",   "#2196F3"),
        ("semantic_reward", "Semantic", "#4CAF50"),
        ("budget_reward",   "Budget",   "#FF9800"),
        ("locale_reward",   "Locale",   "#9C27B0"),
    ]

    for col, label, color in components:
        if col in df.columns:
            ax.plot(ep, _smooth(df[col]), label=label, color=color, linewidth=2)

    # Curriculum transition lines
    ax.axvline(LEVEL2_START, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(LEVEL3_START, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(LEVEL2_START + 5, 0.02, "Level 2", color="gray", fontsize=9)
    ax.text(LEVEL3_START + 5, 0.02, "Level 3", color="gray", fontsize=9)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward (0 – 1)", fontsize=12)
    ax.set_title("IsoSync — Reward Components over Training", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out1 = PLOTS_DIR / "reward_curves.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved {out1}")

    # ── Plot 2: Budget deficit ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))

    if "budget_deficit" in df.columns:
        ax.plot(ep, _smooth(df["budget_deficit"]), color="#F44336", linewidth=2,
                label="Budget deficit (s)")
        ax.fill_between(ep, 0, _smooth(df["budget_deficit"]),
                        color="#F44336", alpha=0.15)

    ax.axhline(0.2, color="#4CAF50", linestyle=":", linewidth=1.5, label="Target threshold (0.2s)")
    ax.axhline(2.0, color="#FF9800", linestyle=":", linewidth=1.5, label="Fail threshold (2.0s)")

    ax.axvline(LEVEL2_START, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(LEVEL3_START, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(LEVEL2_START + 5, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 0.05,
            "Level 2", color="gray", fontsize=9)
    ax.text(LEVEL3_START + 5, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 0.05,
            "Level 3", color="gray", fontsize=9)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative deficit (seconds)", fontsize=12)
    ax.set_title("IsoSync — Budget Deficit over Training\n"
                 "(Should trend toward 0 as agent learns tighter translations)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out2 = PLOTS_DIR / "budget_deficit.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
