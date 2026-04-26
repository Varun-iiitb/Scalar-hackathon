"""
IsoSync training script — REINFORCE with custom episode rollouts.

Why not GRPOTrainer with a static dataset?
GRPOTrainer expects a fixed prompt dataset. IsoSync's core novelty is the
SEQUENTIAL budget mechanic: each translation affects the budget available
for future segments in the same episode. A static dataset destroys this
dependency. We use a custom rollout loop instead:

  1. Run a full episode (reset → step × N → done)
  2. Collect all (prompt, translation, reward) tuples
  3. Compute REINFORCE loss: L = -log_prob(translation) * advantage
  4. Backprop and update with AdamW

Curriculum:
  Episodes   0 –  150  →  Level 1 (Portuguese, 5 segments,  0.5s slack)
  Episodes 150 –  300  →  Level 2 (Portuguese, 8 segments,  0.2s slack)
  Episodes 300 –  500  →  Level 3 (Portuguese, 10 segments, 0.05s slack)

Run:
  python train.py
"""

import sys
import csv
import random
import warnings
from pathlib import Path
from datetime import datetime
import os

# Silence noisy transformers/unsloth warnings before any imports
os.environ["TRANSFORMERS_VERBOSITY"]    = "error"
os.environ["TOKENIZERS_PARALLELISM"]    = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAME          = "unsloth/Qwen2.5-1.5B-Instruct"
TOTAL_EPISODES      = 700
EASY_THRESHOLD      = 100    # episodes 0-99    → level 1 (Portuguese, 5 seg, 0.5s)
MEDIUM_THRESHOLD    = 250    # episodes 100-249 → level 2 (Portuguese, 8 seg, 0.2s)
                             # episodes 250-699 → level 3 (Portuguese, 10 seg, 0.05s)
LOG_INTERVAL        = 10
MAX_PROMPT_LEN      = 400    # tokens — keeps each prompt well inside context
MAX_NEW_TOKENS      = 80     # enough for one translated sentence
LORA_RANK           = 16
LEARNING_RATE       = 1e-5
GRAD_CLIP           = 1.0
TEMPERATURE         = 0.7
CHECKPOINT_EVERY    = 50
OUTPUT_DIR          = PROJECT_ROOT / "checkpoints"
LOGS_DIR            = PROJECT_ROOT / "logs"
HF_REPO_NAME        = "varun1235/isosync-qwen2.5-1.5b"
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "plots").mkdir(parents=True, exist_ok=True)

CSV_PATH = LOGS_DIR / "rewards_log.csv"

import torch
from unsloth import FastLanguageModel
from environment import IsoSyncEnvironment

CSV_FIELDS = [
    "episode", "level", "language",
    "total_reward", "timing_reward", "semantic_reward",
    "budget_reward", "locale_reward", "budget_deficit",
    "done_reason", "n_segments_completed",
]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


# ── Curriculum helpers ─────────────────────────────────────────────────────────

def get_level(episode: int) -> int:
    if episode < EASY_THRESHOLD:
        return 1
    if episode < MEDIUM_THRESHOLD:
        return 2
    return 3


# ── Translation generator ─────────────────────────────────────────────────────

def generate_translation(model, tokenizer, prompt: str) -> tuple[str, torch.Tensor]:
    """
    Generate a translation from the model.

    Returns:
        translation: decoded text (new tokens only, prompt stripped)
        output_ids:  full token sequence (prompt + response) as tensor
                     — used for gradient computation in policy_gradient_loss
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LEN,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len  = inputs["input_ids"].shape[1]
    new_ids     = output_ids[0, prompt_len:]
    translation = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return translation, output_ids, prompt_len


# ── REINFORCE loss ────────────────────────────────────────────────────────────

def policy_gradient_loss(
    model,
    output_ids: torch.Tensor,
    prompt_len: int,
    reward: float,
    baseline: float,
) -> torch.Tensor:
    """
    REINFORCE loss for one (prompt + response, reward) pair.

    Steps:
      1. Forward pass through model with full sequence (prompt + response)
      2. Mask prompt tokens in labels (set to -100 so loss ignores them)
      3. CE loss = mean NLL over response tokens
      4. pg_loss = CE * advantage   (minimising this = maximising log_prob * advantage)

    Args:
        output_ids: shape [1, prompt_len + response_len]
        prompt_len: number of prompt tokens to mask from loss
        reward:     scalar reward for this step
        baseline:   running mean reward (variance reduction)
    """
    labels = output_ids.clone()
    labels[:, :prompt_len] = -100   # ignore prompt in loss

    outputs   = model(input_ids=output_ids, labels=labels)
    ce_loss   = outputs.loss                     # mean NLL over response tokens
    advantage = reward - baseline
    pg_loss   = ce_loss * advantage              # REINFORCE: minimise -log_p * advantage

    return pg_loss


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  IsoSync Training — REINFORCE with Episode Rollouts")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"  Model          : {MODEL_NAME}")
    print(f"  Total episodes : {TOTAL_EPISODES}")
    print(f"  Curriculum     :")
    print(f"    L1 (0-{EASY_THRESHOLD-1}):   Portuguese, 5 segments,  0.5s slack")
    print(f"    L2 ({EASY_THRESHOLD}-{MEDIUM_THRESHOLD-1}): Portuguese, 8 segments,  0.2s slack")
    print(f"    L3 ({MEDIUM_THRESHOLD}-{TOTAL_EPISODES-1}): Portuguese, 10 segments, 0.05s slack")
    print(f"  Learning rate  : {LEARNING_RATE}")
    print(f"  LoRA rank      : {LORA_RANK}")
    print("=" * 65)

    # ── Load model ─────────────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    tokenizer.padding_side = "left"
    model.train()

    # Clear max_length so generate() with max_new_tokens stops printing
    # "Both max_new_tokens and max_length seem to have been set" every call
    model.generation_config.max_length = None

    # Suppress transformers/unsloth runtime logging clutter
    import transformers
    transformers.logging.set_verbosity_error()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    env       = IsoSyncEnvironment()

    # Exponential moving average of reward — used as REINFORCE baseline
    running_baseline = 0.0

    # Early stopping
    best_reward      = -999.0
    patience_counter = 0
    PATIENCE         = 200    # increased — reward is volatile, give it room before stopping
    BEST_MODEL_DIR   = OUTPUT_DIR / "best_model"

    # ── Training loop ──────────────────────────────────────────────────────────
    for episode in range(TOTAL_EPISODES):
        level = get_level(episode)
        obs   = env.reset(level=level)

        # Collect full episode rollout
        step_rewards   = []
        last_info      = {}
        done           = False

        optimizer.zero_grad()
        episode_loss = 0.0

        while not done:
            translation, output_ids, prompt_len = generate_translation(
                model, tokenizer, obs
            )

            obs, reward, done, info = env.step({"translation": translation})
            last_info = info

            # Compute and accumulate gradient for this step
            pg_loss = policy_gradient_loss(
                model, output_ids, prompt_len, reward, running_baseline
            )
            pg_loss.backward()   # gradients accumulate across steps
            episode_loss += pg_loss.item()
            step_rewards.append(reward)

        # Update after full episode (gradient accumulation over all steps)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Update running baseline (EMA of mean episode reward)
        mean_reward      = sum(step_rewards) / max(1, len(step_rewards))
        running_baseline = 0.9 * running_baseline + 0.1 * mean_reward
        scheduler.step()

        # Early stopping — save best model checkpoint
        if mean_reward > best_reward:
            best_reward      = mean_reward
            patience_counter = 0
            model.save_pretrained(str(BEST_MODEL_DIR))
            tokenizer.save_pretrained(str(BEST_MODEL_DIR))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [early stopping at episode {episode} — best reward {best_reward:.4f}]")
                break

        # ── Log every LOG_INTERVAL episodes ────────────────────────────────────
        if episode % LOG_INTERVAL == 0:
            row = {
                "episode":              episode,
                "level":                level,
                "language":             env._language,
                "total_reward":         round(mean_reward, 4),
                "timing_reward":        round(last_info.get("timing_reward", 0.0), 4),
                "semantic_reward":      round(last_info.get("semantic_reward", 0.0), 4),
                "budget_reward":        round(last_info.get("budget_reward", 0.0), 4),
                "locale_reward":        round(last_info.get("locale_reward", 0.0), 4),
                "budget_deficit":       round(last_info.get("final_deficit", last_info.get("budget_deficit", 0.0)), 4),
                "done_reason":          last_info.get("done_reason", "completed"),
                "n_segments_completed": last_info.get("n_segments", len(step_rewards)),
            }
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

            print(
                f"  [ep {episode:>5}]  L{level}  "
                f"reward={mean_reward:+.3f}  "
                f"timing={row['timing_reward']:.3f}  "
                f"sem={row['semantic_reward']:.3f}  "
                f"budget={row['budget_reward']:.3f}  "
                f"deficit={row['budget_deficit']:.2f}s  "
                f"baseline={running_baseline:+.3f}"
            )

        # ── Checkpoint ─────────────────────────────────────────────────────────
        if episode > 0 and episode % CHECKPOINT_EVERY == 0:
            ckpt = OUTPUT_DIR / f"checkpoint-ep{episode}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            print(f"  [checkpoint saved → {ckpt}]")

    # ── Save final model ────────────────────────────────────────────────────────
    final_path = OUTPUT_DIR / "final"
    print(f"\nSaving final model → {final_path}")
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    try:
        hf_token = os.environ.get("HF_TOKEN")
        print(f"Pushing to HuggingFace Hub → {HF_REPO_NAME}")
        model.push_to_hub(HF_REPO_NAME, token=hf_token)
        tokenizer.push_to_hub(HF_REPO_NAME, token=hf_token)
        print("Push complete.")
    except Exception as exc:
        print(f"Hub push skipped: {exc}")

    try:
        import plot_results
        plot_results.main()
        print("Plots saved to plots/")
    except Exception as exc:
        print(f"Plotting skipped: {exc}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
