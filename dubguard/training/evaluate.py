"""
DubGuard evaluation script.

Loads the untrained base model and the trained LoRA checkpoint, runs both on
10 fixed evaluation episodes, and produces three plots:

  training/plots/reward_curves.png       — 6-subplot training curves
  training/plots/before_after.png        — before/after comparison table
  training/plots/false_positive_rate.png — FP rate over training

Results are also saved to training/logs/eval_results.json.

Usage:
  cd dubguard
  python -X utf8 training/evaluate.py
"""

import sys
import os
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_PATH = PROJECT_ROOT / "training" / "checkpoints" / "final"
LOGS_DIR        = PROJECT_ROOT / "training" / "logs"
PLOTS_DIR       = PROJECT_ROOT / "training" / "plots"
CSV_PATH        = LOGS_DIR / "training_log.csv"
EVAL_JSON_PATH  = LOGS_DIR / "eval_results.json"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

import torch
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from unsloth import FastLanguageModel
from peft import PeftModel

from data.loader import load_all
from rewards.combined import compute_reward
from training.prompts import SYSTEM_PROMPT, format_observation
from training.agent import DubGuardAgent

# ══════════════════════════════════════════════════════════════════════════════
#  FIXED EVALUATION SET — 10 episodes, 2 per error type, deterministic
# ══════════════════════════════════════════════════════════════════════════════

def _select_eval_episodes(n_per_type: int = 2) -> list[dict]:
    all_eps = load_all(shuffle=False)  # no shuffle = deterministic
    error_types = [
        "timing_collision", "translation_error",
        "tone_mismatch", "cultural_mismatch", None,
    ]
    selected = []
    for et in error_types:
        pool = [ep for ep in all_eps if ep["ground_truth"]["error_type"] == et]
        selected.extend(pool[:n_per_type])
    return selected[:10]


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _run_inference(model, tokenizer, observation: dict) -> tuple[dict, bool]:
    """Run a single inference pass and return (action_dict, parse_failed)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": format_observation(observation)},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,            # greedy for eval
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Parse
    parse_failed = False
    action = None
    for attempt in [
        lambda: json.loads(raw),
        lambda: json.loads(raw[raw.index("{"):raw.rindex("}") + 1]),
        lambda: json.loads(re.search(r"\{[\s\S]*\}", raw).group()),
    ]:
        try:
            action = attempt()
            break
        except Exception:
            continue

    if action is None:
        parse_failed = True
        seg_id = observation.get("segment_id", "seg_0000")
        action = {
            "segment_id": int("".join(filter(str.isdigit, str(seg_id))) or "0"),
            "error_type": None, "severity": "PASS",
            "reason": "parse_error", "suggested_fix": "",
            "estimated_fix_duration": 0.0,
        }

    return action, parse_failed


def _score(action: dict, ep: dict) -> dict:
    obs = ep["observation"]
    gt  = ep["ground_truth"]
    return compute_reward(
        action=action,
        ground_truth=gt,
        max_allowed_duration=obs["max_allowed_dubbed_duration_seconds"],
        language_code=obs["dubbed"]["language_code"],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"

def _load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    return model, tokenizer


def _load_trained_model(tokenizer):
    base, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )
    trained = PeftModel.from_pretrained(base, str(CHECKPOINT_PATH))
    FastLanguageModel.for_inference(trained)
    return trained


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — reward curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_reward_curves(df: pd.DataFrame):
    metrics = [
        ("detection_score",        "Detection Score"),
        ("severity_score",         "Severity Score"),
        ("correction_score",       "Correction Score"),
        ("cultural_score",         "Cultural Score"),
        ("false_positive_penalty", "False Positive Penalty"),
        ("combined_reward",        "Combined Reward"),
    ]
    window = 20

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("DubGuard Training Results", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        smoothed = df[col].rolling(window, min_periods=1).mean()
        ax.plot(df["global_step"], smoothed, linewidth=1.8)
        ax.axvline(x=200, color="orange", linestyle="--", linewidth=1, label="→ medium")
        ax.axvline(x=500, color="red",    linestyle="--", linewidth=1, label="→ hard")
        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = PLOTS_DIR / "reward_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — before / after comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_before_after(eval_results: list[dict]):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    col_labels = ["Error type", "Before", "After", "Δ"]
    table_data = []
    cell_colors = [["#f0f0f0"] * 4]   # header row colour

    for r in eval_results:
        delta = r["trained_reward"] - r["base_reward"]
        table_data.append([
            r["error_type"] or "none (PASS)",
            f"{r['base_reward']:.3f}",
            f"{r['trained_reward']:.3f}",
            f"{delta:+.3f}",
        ])
        cell_colors.append([
            "#ffffff", "#ffffff", "#ffffff",
            "#d4edda" if delta >= 0 else "#f8d7da",
        ])

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors[1:],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)

    ax.set_title(
        "Before vs After Training — Combined Reward on 10 Fixed Eval Episodes",
        fontsize=13, fontweight="bold", pad=20,
    )

    green_patch = mpatches.Patch(color="#d4edda", label="Improvement")
    red_patch   = mpatches.Patch(color="#f8d7da", label="Regression")
    ax.legend(handles=[green_patch, red_patch], loc="lower right")

    out = PLOTS_DIR / "before_after.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — false positive rate over training
# ══════════════════════════════════════════════════════════════════════════════

def plot_false_positive_rate(df: pd.DataFrame):
    clean = df[df["planted_error_type"] == "none"].copy()
    if clean.empty:
        print("  No clean episodes in log — skipping FP rate plot.")
        return

    # FP = clean episode where agent predicted an error (severity != PASS)
    clean["is_fp"] = (clean["predicted_severity"] != "PASS").astype(float)
    window = 50
    fp_rate = clean.set_index("global_step")["is_fp"].rolling(window, min_periods=1).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fp_rate.index, fp_rate.values, color="steelblue", linewidth=1.8)
    ax.axvline(x=200, color="orange", linestyle="--", linewidth=1, label="→ medium")
    ax.axvline(x=500, color="red",    linestyle="--", linewidth=1, label="→ hard")
    ax.set_title("False Positive Rate Over Training (rolling 50-episode window)", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = PLOTS_DIR / "false_positive_rate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  DubGuard Evaluation")
    print("=" * 60)

    eval_episodes = _select_eval_episodes(n_per_type=2)
    print(f"\nEval set: {len(eval_episodes)} episodes")
    for ep in eval_episodes:
        gt = ep["ground_truth"]
        print(f"  {ep['observation']['episode_id']:20s}  error={gt['error_type']}")

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading base model …")
    base_model, tokenizer = _load_base_model()

    trained_model = None
    if CHECKPOINT_PATH.exists():
        print("Loading trained checkpoint …")
        trained_model = _load_trained_model(tokenizer)
    else:
        print(f"WARNING: checkpoint not found at {CHECKPOINT_PATH}")
        print("Skipping trained-model inference — before/after plot will be omitted.")

    # ── Run inference ──────────────────────────────────────────────────────────
    print("\nRunning inference …")
    eval_results = []

    for ep in eval_episodes:
        obs = ep["observation"]
        gt  = ep["ground_truth"]

        # Base model
        base_action, base_parse_fail = _run_inference(base_model, tokenizer, obs)
        base_reward_dict = _score(base_action, ep)
        base_combined    = float(base_reward_dict["combined_score"])

        # Trained model
        if trained_model is not None:
            tr_action, tr_parse_fail = _run_inference(trained_model, tokenizer, obs)
            tr_reward_dict = _score(tr_action, ep)
            tr_combined    = float(tr_reward_dict["combined_score"])
        else:
            tr_action, tr_parse_fail = base_action, True
            tr_reward_dict = base_reward_dict
            tr_combined    = base_combined

        result = {
            "episode_id":       obs["episode_id"],
            "error_type":       gt["error_type"],
            "difficulty":       obs["difficulty_level"],
            # base model
            "base_action":      base_action,
            "base_rewards":     base_reward_dict,
            "base_reward":      base_combined,
            "base_parse_fail":  base_parse_fail,
            # trained model
            "trained_action":   tr_action,
            "trained_rewards":  tr_reward_dict,
            "trained_reward":   tr_combined,
            "trained_parse_fail": tr_parse_fail,
        }
        eval_results.append(result)

        delta = tr_combined - base_combined
        symbol = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(
            f"  {obs['episode_id']:22s}  "
            f"base={base_combined:+.3f}  trained={tr_combined:+.3f}  "
            f"Δ={delta:+.3f} {symbol}"
        )

    # ── Save eval JSON ─────────────────────────────────────────────────────────
    with open(EVAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"\nEval results saved → {EVAL_JSON_PATH}")

    # ── Generate plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print(f"  Training log: {len(df)} rows")
        plot_reward_curves(df)
        plot_false_positive_rate(df)
    else:
        print(f"  Training CSV not found ({CSV_PATH}) — skipping curve plots.")

    if trained_model is not None:
        plot_before_after(eval_results)
    else:
        print("  No trained model — skipping before/after plot.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
