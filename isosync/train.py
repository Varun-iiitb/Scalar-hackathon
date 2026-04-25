"""
IsoSync GRPO training script.

Trains Qwen2.5-1.5B-Instruct to generate isochronous dubbing translations
under cumulative timing budget pressure using GRPO via Unsloth + TRL.

Curriculum:
  Episodes   0 –  200  →  level 1  (5 segments, 0.5s slack)
  Episodes 200 –  500  →  level 2  (8 segments, 0.2s slack)
  Episodes 500 – 1000  →  level 3  (10 segments, 0.1s slack + locale)

Run:
  python train.py
"""

import sys
import os
import csv
import json
import random
from pathlib import Path
from datetime import datetime

# ── Ensure isosync/ is on the path ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAME          = "unsloth/Qwen2.5-1.5B-Instruct"
TOTAL_EPISODES      = 1000
EASY_THRESHOLD      = 200    # episodes 0-199   → level 1
MEDIUM_THRESHOLD    = 500    # episodes 200-499 → level 2
                             # episodes 500-999 → level 3
LOG_INTERVAL        = 10     # log to CSV every N episodes
NUM_GENERATIONS     = 4      # GRPO: completions per prompt
MAX_SEQ_LEN         = 512
LORA_RANK           = 8
LEARNING_RATE       = 5e-6
LANGUAGES           = ["Hindi", "Portuguese"]
OUTPUT_DIR          = PROJECT_ROOT / "checkpoints"
LOGS_DIR            = PROJECT_ROOT / "logs"
HF_REPO_NAME        = "varun1235/isosync-qwen2.5-1.5b"
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "plots").mkdir(parents=True, exist_ok=True)

CSV_PATH = LOGS_DIR / "rewards_log.csv"

# ── Imports after path setup ───────────────────────────────────────────────────
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from data_gen import generate_episode
from rewards import compute_reward, timing_reward, semantic_reward, budget_reward, locale_reward

# ── CSV setup ─────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "episode", "level", "language",
    "total_reward", "timing_reward", "semantic_reward",
    "budget_reward", "locale_reward", "budget_deficit",
    "done_reason", "n_segments_completed",
]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


# ── Reward function for GRPOTrainer ───────────────────────────────────────────

def grpo_reward_fn(
    completions: list[str],
    reference_translation: list[str],
    original_duration: list[float],
    target_language: list[str],
    locale: list[str],
    original_text: list[str],
    budget_deficit_at_step: list[float],
    **kwargs,
) -> list[float]:
    """
    Reward function called by TRL GRPOTrainer.

    Each call receives `num_generations` completions for one prompt,
    plus the dataset columns broadcast to match.

    We reconstruct the segment dict and call compute_reward for each completion.
    """
    rewards = []
    for (completion, ref, dur, lang, loc, orig, deficit) in zip(
        completions, reference_translation, original_duration,
        target_language, locale, original_text, budget_deficit_at_step,
    ):
        # Build a minimal segment dict that rewards.py expects
        segment = {
            "original_text":         orig,
            "original_duration":     float(dur),
            "target_language":       lang,
            "locale":                loc,
            "reference_translation": ref,
        }
        r = compute_reward(completion.strip(), segment, float(deficit))
        rewards.append(float(r))
    return rewards


# ── Dataset builder ───────────────────────────────────────────────────────────

def _build_dataset(level: int, n_episodes: int = 200) -> Dataset:
    """
    Generate n_episodes episodes at the given level and flatten to
    individual segment observations for GRPOTrainer.

    Each row = one segment prompt with metadata columns for the reward fn.
    Budget deficit is pre-set to 0.0 for all rows (the environment's
    sequential deficit tracking happens during actual inference; the GRPO
    dataset teaches per-segment quality independently).
    """
    rows = []
    for ep_idx in range(n_episodes):
        lang = random.choice(LANGUAGES)
        segments = generate_episode(level=level, language=lang, seed=ep_idx)
        for seg in segments:
            rows.append({
                "prompt": [
                    {
                        "role": "user",
                        "content": _format_prompt(seg, budget_deficit=0.0),
                    }
                ],
                "reference_translation":  seg["reference_translation"],
                "original_duration":      seg["original_duration"],
                "target_language":        seg["target_language"],
                "locale":                 seg["locale"],
                "original_text":          seg["original_text"],
                "budget_deficit_at_step": 0.0,
            })
    return Dataset.from_list(rows)


def _format_prompt(seg: dict, budget_deficit: float = 0.0) -> str:
    lang = seg["target_language"]
    dur  = seg["original_duration"]
    loc  = seg["locale"]

    status = "ON TRACK" if budget_deficit < 0.2 else f"OVER BUDGET by {budget_deficit:.1f}s"

    return (
        f"You are a professional video dubbing translator.\n\n"
        f"Translate the following English segment into {lang}.\n\n"
        f"Original: {seg['original_text']}\n"
        f"Time window: {dur} seconds\n"
        f"Max syllables: {seg['max_syllables']}\n"
        f"Budget status: {status}\n"
        f"Locale: {loc}\n\n"
        f"Rules:\n"
        f"- Your translation MUST fit within {dur} seconds when spoken\n"
        f"- Preserve the core meaning\n"
        f"- Sound natural in {loc}\n"
        f"- Do not copy the English text\n\n"
        f"Respond with ONLY the translation. No explanation."
    )


# ── Episode evaluator (for CSV logging) ───────────────────────────────────────

def _run_eval_episode(model, tokenizer, level: int, language: str) -> dict:
    """
    Run one full episode with the model using greedy decoding.
    Returns per-component reward averages and budget deficit.
    Used for logging only — not part of training.
    """
    from environment import IsoSyncEnvironment
    env = IsoSyncEnvironment()
    obs = env.reset(level=level, language=language)

    totals = {k: 0.0 for k in ["timing", "semantic", "budget", "locale", "combined"]}
    n_steps = 0
    done_reason = "unknown"

    while True:
        # Tokenise the observation
        inputs = tokenizer(obs, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128,
                do_sample=False, temperature=1.0,
            )
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()

        obs, reward, done, info = env.step({"translation": raw})
        n_steps += 1

        for k in ["timing", "semantic", "budget", "locale"]:
            totals[k] += info.get(f"{k}_reward", 0.0)
        totals["combined"] += info.get("combined_reward", reward)

        if done:
            done_reason = info.get("done_reason", "completed")
            final_deficit = info.get("budget_deficit", 0.0)
            break

    n = max(1, n_steps)
    return {
        "timing_reward":   round(totals["timing"] / n, 4),
        "semantic_reward": round(totals["semantic"] / n, 4),
        "budget_reward":   round(totals["budget"] / n, 4),
        "locale_reward":   round(totals["locale"] / n, 4),
        "total_reward":    round(totals["combined"] / n, 4),
        "budget_deficit":  round(final_deficit, 4),
        "done_reason":     done_reason,
        "n_segments":      n_steps,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  IsoSync GRPO Training")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"  Model          : {MODEL_NAME}")
    print(f"  Total episodes : {TOTAL_EPISODES}")
    print(f"  Curriculum     : L1 0-{EASY_THRESHOLD-1} | "
          f"L2 {EASY_THRESHOLD}-{MEDIUM_THRESHOLD-1} | "
          f"L3 {MEDIUM_THRESHOLD}-{TOTAL_EPISODES-1}")
    print(f"  Languages      : {LANGUAGES}")
    print(f"  GRPO gens/step : {NUM_GENERATIONS}")
    print(f"  LoRA rank      : {LORA_RANK}")
    print(f"  Output dir     : {OUTPUT_DIR}")
    print("=" * 65)

    # ── Load model ─────────────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    tokenizer.padding_side = "left"

    # ── Curriculum phases ──────────────────────────────────────────────────────
    phases = [
        {"name": "Level 1 — easy",        "level": 1, "max_steps": EASY_THRESHOLD,                         "n_gen_eps": 200},
        {"name": "Level 2 — medium",       "level": 2, "max_steps": MEDIUM_THRESHOLD - EASY_THRESHOLD,      "n_gen_eps": 300},
        {"name": "Level 3 — hard+locale",  "level": 3, "max_steps": TOTAL_EPISODES - MEDIUM_THRESHOLD,      "n_gen_eps": 500},
    ]

    episode_counter = 0

    for phase in phases:
        print(f"\n{'─'*65}")
        print(f"  {phase['name']}  ({phase['max_steps']} training steps)")
        print(f"{'─'*65}")

        dataset = _build_dataset(level=phase["level"], n_episodes=phase["n_gen_eps"])
        print(f"  Dataset rows: {len(dataset)}")

        grpo_config = GRPOConfig(
            output_dir=str(OUTPUT_DIR / phase["name"].replace(" ", "_")),
            max_steps=phase["max_steps"],
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=NUM_GENERATIONS,
            max_completion_length=128,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=1,
            save_steps=100,
            save_total_limit=2,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to="none",
            seed=42,
        )

        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=grpo_reward_fn,
            args=grpo_config,
            train_dataset=dataset,
        )

        trainer.train()

        # ── Log eval metrics every LOG_INTERVAL episodes (approximate) ─────────
        for log_ep in range(0, phase["max_steps"], LOG_INTERVAL):
            lang = random.choice(LANGUAGES)
            metrics = _run_eval_episode(model, tokenizer, phase["level"], lang)
            row = {
                "episode":              episode_counter + log_ep,
                "level":                phase["level"],
                "language":             lang,
                "total_reward":         metrics["total_reward"],
                "timing_reward":        metrics["timing_reward"],
                "semantic_reward":      metrics["semantic_reward"],
                "budget_reward":        metrics["budget_reward"],
                "locale_reward":        metrics["locale_reward"],
                "budget_deficit":       metrics["budget_deficit"],
                "done_reason":          metrics["done_reason"],
                "n_segments_completed": metrics["n_segments"],
            }
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

            if log_ep % (LOG_INTERVAL * 5) == 0:
                print(
                    f"  [ep {episode_counter + log_ep:>5}]  "
                    f"total={metrics['total_reward']:+.3f}  "
                    f"timing={metrics['timing_reward']:.3f}  "
                    f"sem={metrics['semantic_reward']:.3f}  "
                    f"budget={metrics['budget_reward']:.3f}  "
                    f"locale={metrics['locale_reward']:.3f}  "
                    f"deficit={metrics['budget_deficit']:.2f}s"
                )

        episode_counter += phase["max_steps"]

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = OUTPUT_DIR / "final"
    print(f"\nSaving final model → {final_path}")
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    try:
        print(f"Pushing to HuggingFace Hub → {HF_REPO_NAME}")
        model.push_to_hub(HF_REPO_NAME)
        tokenizer.push_to_hub(HF_REPO_NAME)
        print("Push complete.")
    except Exception as exc:
        print(f"Hub push skipped: {exc}")

    # ── Generate plots ────────────────────────────────────────────────────────
    try:
        import plot_results
        plot_results.main()
        print("Plots saved to plots/")
    except Exception as exc:
        print(f"Plotting skipped: {exc}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
