"""
DubGuard GRPO training script.
Trains Qwen2.5-3B-Instruct to act as a QC reviewer for AI-dubbed video segments
using Group Relative Policy Optimization (GRPO) via Unsloth + TRL.

Curriculum:
  Episodes   0 – 199  →  easy   segments only
  Episodes 200 – 499  →  easy + medium segments
  Episodes 500 – 999  →  all segments (easy + medium + hard)

Each curriculum phase is a separate GRPOTrainer.train() call so that the LoRA
weights accumulate across phases on the same model object.
"""

import sys
import os
import json
import csv
import collections
from pathlib import Path
from datetime import datetime

# ── Project root on sys.path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — change these before a run
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAME          = "unsloth/Qwen2.5-3B-Instruct"
NUM_EPISODES        = 1000          # total training steps (= "episodes")
EASY_THRESHOLD      = 200           # steps 0-199   → easy only
MEDIUM_THRESHOLD    = 500           # steps 200-499 → easy + medium
                                    # steps 500-999 → all difficulties
LOG_INTERVAL        = 10            # console summary every N steps
CHECKPOINT_INTERVAL = 100           # save checkpoint every N steps
NUM_GENERATIONS     = 4             # GRPO: completions generated per prompt
MAX_SEQ_LEN         = 1024
LORA_RANK           = 16
LEARNING_RATE       = 5e-6
OUTPUT_DIR          = PROJECT_ROOT / "training" / "checkpoints"
LOGS_DIR            = PROJECT_ROOT / "training" / "logs"
HF_REPO_NAME        = "dubguard-qwen2.5-3b-instruct"   # set your HF username/repo
# ══════════════════════════════════════════════════════════════════════════════

# Create all output directories up front
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "training" / "plots").mkdir(parents=True, exist_ok=True)

CSV_PATH = LOGS_DIR / "training_log.csv"

# ── Imports after path setup ───────────────────────────────────────────────────
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback, TrainerState, TrainerControl

from unsloth import FastLanguageModel

from data.loader import load_all
from rewards.combined import compute_reward
from training.prompts import SYSTEM_PROMPT, format_observation
from training.agent import DubGuardAgent


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL REWARD LOG BUFFER
#  The reward function (called inside TRL's training loop) writes here;
#  the callback flushes this to CSV after each step.
# ══════════════════════════════════════════════════════════════════════════════
_reward_log_buffer: list[dict] = []


def _parse_completion(text: str) -> dict:
    """Parse the model's raw text output into an action dict."""
    import re
    # Strategy 1: direct
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Strategy 2: first { to last }
    try:
        s, e = text.index("{"), text.rindex("}") + 1
        return json.loads(text[s:e])
    except (ValueError, json.JSONDecodeError):
        pass
    # Strategy 3: regex
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group())
    except (json.JSONDecodeError, ValueError):
        pass
    return {
        "segment_id": 0, "error_type": None, "severity": "PASS",
        "reason": "parse_error", "suggested_fix": "",
        "estimated_fix_duration": 0.0, "_parse_failed": True,
    }


def grpo_reward_fn(
    completions: list[str],
    ground_truth_json: list[str],
    max_allowed_duration: list[float],
    language_code: list[str],
    episode_id: list[str],
    difficulty: list[str],
    planted_error_type: list[str],
    **kwargs,
) -> list[float]:
    """
    Reward function passed to TRL's GRPOTrainer.

    Called with `num_generations` completions per prompt in the batch.
    Each kwarg list is length = batch_size × num_generations (TRL broadcasts
    dataset columns to match the number of completions).

    Writes per-completion stats to _reward_log_buffer for CSV logging.
    Imports compute_reward from rewards/combined.py — reward logic lives there.
    """
    rewards = []

    for completion, gt_json, max_dur, lang, ep_id, diff, planted in zip(
        completions, ground_truth_json, max_allowed_duration,
        language_code, episode_id, difficulty, planted_error_type,
    ):
        parse_failed = False
        try:
            action = _parse_completion(completion)
            parse_failed = action.pop("_parse_failed", False)
        except Exception:
            action = {
                "segment_id": 0, "error_type": None, "severity": "PASS",
                "reason": "exception", "suggested_fix": "",
                "estimated_fix_duration": 0.0,
            }
            parse_failed = True

        try:
            gt = json.loads(gt_json)
            reward_dict = compute_reward(action, gt, float(max_dur), str(lang))
        except Exception:
            reward_dict = {
                "detection_score": 0.0, "severity_score": 0.0,
                "correction_score": 0.0, "cultural_score": 0.0,
                "false_positive_penalty": 0.0, "combined_score": -0.3,
            }

        combined = float(reward_dict["combined_score"])
        rewards.append(combined)

        _reward_log_buffer.append({
            "episode_id":            ep_id,
            "difficulty":            diff,
            "combined_reward":       combined,
            "detection_score":       reward_dict["detection_score"],
            "severity_score":        reward_dict["severity_score"],
            "correction_score":      reward_dict["correction_score"],
            "cultural_score":        reward_dict["cultural_score"],
            "false_positive_penalty":reward_dict["false_positive_penalty"],
            "planted_error_type":    planted,
            "predicted_severity":    action.get("severity", "PASS"),
            "parse_failed":          parse_failed,
        })

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "global_step", "episode_id", "difficulty",
    "combined_reward", "detection_score", "severity_score",
    "correction_score", "cultural_score", "false_positive_penalty",
    "planted_error_type", "predicted_severity", "parse_failed",
]

class DubGuardLoggingCallback(TrainerCallback):
    def __init__(self, csv_path: Path, log_interval: int, checkpoint_interval: int, output_dir: Path):
        self.csv_path           = csv_path
        self.log_interval       = log_interval
        self.checkpoint_interval= checkpoint_interval
        self.output_dir         = output_dir
        self._recent: collections.deque = collections.deque(maxlen=log_interval)

        # Write CSV header once
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        global _reward_log_buffer
        step = state.global_step

        if not _reward_log_buffer:
            return

        # Average across all completions generated this step (NUM_GENERATIONS per prompt)
        avg = {k: 0.0 for k in CSV_FIELDS if k not in ("global_step", "episode_id", "difficulty",
                                                          "planted_error_type", "predicted_severity",
                                                          "parse_failed")}
        n = len(_reward_log_buffer)
        for entry in _reward_log_buffer:
            for key in avg:
                avg[key] += entry.get(key, 0.0) / n

        # Use first entry for categorical fields
        first = _reward_log_buffer[0]
        row = {
            "global_step":            step,
            "episode_id":             first["episode_id"],
            "difficulty":             first["difficulty"],
            "planted_error_type":     first["planted_error_type"],
            "predicted_severity":     first["predicted_severity"],
            "parse_failed":           any(e["parse_failed"] for e in _reward_log_buffer),
            **avg,
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

        self._recent.append(avg["combined_reward"])
        _reward_log_buffer.clear()

        # Console summary every LOG_INTERVAL steps
        if step % self.log_interval == 0 and len(self._recent) > 0:
            roll_avg = sum(self._recent) / len(self._recent)
            print(
                f"  [step {step:>5}]  "
                f"avg_reward(last {self.log_interval}) = {roll_avg:+.4f}  "
                f"| det={avg['detection_score']:.3f}  "
                f"sev={avg['severity_score']:.3f}  "
                f"cor={avg['correction_score']:.3f}  "
                f"cul={avg['cultural_score']:.3f}  "
                f"fp={avg['false_positive_penalty']:+.3f}"
            )

        # Checkpoint every CHECKPOINT_INTERVAL steps
        if step > 0 and step % self.checkpoint_interval == 0:
            ckpt_path = self.output_dir / f"checkpoint-step-{step}"
            kwargs.get("model", None) and kwargs["model"].save_pretrained(str(ckpt_path))
            print(f"  [step {step}] checkpoint saved → {ckpt_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_dataset(episodes: list[dict]) -> Dataset:
    """Convert a list of episodes into a HuggingFace Dataset for GRPOTrainer."""
    rows = []
    for ep in episodes:
        obs = ep["observation"]
        gt  = ep["ground_truth"]
        rows.append({
            # "prompt" is the required column for GRPOTrainer (list of chat messages)
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": format_observation(obs)},
            ],
            # Extra columns passed as kwargs to the reward function
            "ground_truth_json":   json.dumps(gt),
            "max_allowed_duration": float(obs["max_allowed_dubbed_duration_seconds"]),
            "language_code":        obs["dubbed"]["language_code"],
            "episode_id":           obs["episode_id"],
            "difficulty":           obs["difficulty_level"],
            "planted_error_type":   str(gt.get("error_type") or "none"),
        })
    return Dataset.from_list(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    all_episodes = load_all(shuffle=True, seed=42)

    # Partition by difficulty
    easy_eps   = [ep for ep in all_episodes if ep["observation"]["difficulty_level"] == "easy"]
    medium_eps = [ep for ep in all_episodes if ep["observation"]["difficulty_level"] == "medium"]
    hard_eps   = [ep for ep in all_episodes if ep["observation"]["difficulty_level"] == "hard"]

    # Curriculum phase datasets
    # Phase 1: easy only
    # Phase 2: easy + medium
    # Phase 3: all
    phase_configs = [
        {
            "name":       "Phase 1 — easy",
            "episodes":   easy_eps,
            "max_steps":  EASY_THRESHOLD,
            "difficulty": "easy",
        },
        {
            "name":       "Phase 2 — easy + medium",
            "episodes":   easy_eps + medium_eps,
            "max_steps":  MEDIUM_THRESHOLD - EASY_THRESHOLD,
            "difficulty": "easy+medium",
        },
        {
            "name":       "Phase 3 — all",
            "episodes":   all_episodes,
            "max_steps":  NUM_EPISODES - MEDIUM_THRESHOLD,
            "difficulty": "all",
        },
    ]

    # ── Print startup banner ───────────────────────────────────────────────────
    print("=" * 65)
    print("  DubGuard GRPO Training")
    print("=" * 65)
    print(f"  Model          : {MODEL_NAME}")
    print(f"  Total episodes : {NUM_EPISODES}")
    print(f"  Curriculum     : easy 0-{EASY_THRESHOLD-1} | "
          f"medium {EASY_THRESHOLD}-{MEDIUM_THRESHOLD-1} | "
          f"hard {MEDIUM_THRESHOLD}-{NUM_EPISODES-1}")
    print(f"  Dataset size   : {len(all_episodes)} unique episodes")
    print(f"    easy={len(easy_eps)}  medium={len(medium_eps)}  hard={len(hard_eps)}")
    print(f"  GRPO gen/prompt: {NUM_GENERATIONS}")
    print(f"  LoRA rank      : {LORA_RANK}")
    print(f"  Learning rate  : {LEARNING_RATE}")
    print(f"  Output dir     : {OUTPUT_DIR}")
    print(f"  Log CSV        : {CSV_PATH}")
    print("=" * 65)
    print()

    # NOTE: The environment's reset() method cycles and reshuffles the episode
    # pool each time it completes a full pass, so the agent cannot memorise
    # episode order across curriculum cycles.

    # ── Load model once; LoRA weights accumulate across all phases ────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer.padding_side = "left"

    callback = DubGuardLoggingCallback(
        csv_path=CSV_PATH,
        log_interval=LOG_INTERVAL,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        output_dir=OUTPUT_DIR,
    )

    # ── Curriculum loop ───────────────────────────────────────────────────────
    completed_steps = 0

    for phase in phase_configs:
        print(f"\n{'─'*65}")
        print(f"  {phase['name']}  ({phase['max_steps']} steps)")
        print(f"{'─'*65}")

        dataset = _build_dataset(phase["episodes"])

        grpo_config = GRPOConfig(
            output_dir=str(OUTPUT_DIR / phase["name"].replace(" ", "_")),
            num_train_epochs=None,
            max_steps=phase["max_steps"],
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=NUM_GENERATIONS,
            max_completion_length=256,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=1,
            save_steps=CHECKPOINT_INTERVAL,
            save_total_limit=3,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to="none",           # set to "wandb" to enable W&B logging
            seed=42 + completed_steps,
        )

        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=grpo_reward_fn,
            args=grpo_config,
            train_dataset=dataset,
            callbacks=[callback],
        )

        trainer.train()
        completed_steps += phase["max_steps"]
        print(f"  Phase complete. Total steps so far: {completed_steps}")

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = OUTPUT_DIR / "final"
    print(f"\nSaving final model → {final_path}")
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # ── Push to HuggingFace Hub ───────────────────────────────────────────────
    try:
        print(f"Pushing to HuggingFace Hub → {HF_REPO_NAME}")
        model.push_to_hub(HF_REPO_NAME)
        tokenizer.push_to_hub(HF_REPO_NAME)
        print("Push complete.")
    except Exception as exc:
        print(f"Hub push failed (set HF_REPO_NAME and login with `huggingface-cli login`): {exc}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
