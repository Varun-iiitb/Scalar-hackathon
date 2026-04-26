---
title: IsoSync — RL-Trained Isochronous Dubbing
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# IsoSync: RL-Trained Time-Aware Dubbing

> **An LLM that learns to translate and fit the clock — not just the meaning.**

Most dubbing pipelines translate first, then fix timing as an afterthought.  
IsoSync trains an LLM to treat **timing as a first-class objective** from the start, using reinforcement learning inside a custom OpenEnv-compatible environment.

---

## The Problem

When you dub a video, every sentence has a fixed time window — the exact number of seconds the speaker's mouth is moving. Standard translation ignores this. The result: dubbed audio that overruns, gets clipped, or requires awkward speed adjustments that destroy natural rhythm.

**IsoSync solves this at generation time.** The model learns to produce translations that fit the window — not translations that need to be corrected afterward.

---

## Live Environment

The IsoSync OpenEnv environment is hosted and running on Hugging Face Spaces:

**[varun1235/dubguard-training](https://huggingface.co/spaces/varun1235/dubguard-training)**

```bash
# Reset — start a new episode
curl -X POST https://varun1235-dubguard-training.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"level": 1}'

# Step — submit a translation
curl -X POST https://varun1235-dubguard-training.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"translation": "Adicione sal à água fervente."}'
```

Endpoints: `POST /reset` · `POST /step` · `GET /info` · `GET /state` · `GET /health`

---

## Environment Design

The agent interacts with IsoSync one segment at a time.  
Each segment is one sentence from an English video, with a known target duration.

```
reset(level=1)
    → Observation (English text, duration, syllable cap, budget status)

step({"translation": "..."})
    → next Observation, Reward, done, info
```

### Observation (15 fields)
```json
{
  "prompt":                   "DUBBING TRANSLATION — STRICT LENGTH LIMIT...",
  "episode_id":               "ep_001",
  "segment_id":               2,
  "segments_total":           5,
  "original_text":            "Stir the mixture gently for two minutes.",
  "target_language":          "Portuguese",
  "locale":                   "Brazil",
  "time_window_seconds":      2.2,
  "max_syllables":            11,
  "budget_remaining_seconds": 8.4,
  "budget_bank_seconds":      0.3,
  "budget_deficit_seconds":   0.0,
  "previous_translation":     "Adicione sal à água fervente.",
  "curriculum_level":         1
}
```

### Budget Banking Mechanic
The agent manages a **shared timing budget** across the whole episode:

```
Under-run → bank += saved time        (reward for early compression)
Over-run  → draw from bank first      (bank absorbs before penalising)
           → deficit += remaining     (real overrun counts against reward)

Early termination if deficit > 3.0s
```

This creates genuine strategic depth — compress short segments early to build slack for semantically dense ones later.

---

## Reward Design

Five independent, fully deterministic reward components. No LLM judges. No subjectivity.

| Component | Weight | How it's measured |
|-----------|--------|-------------------|
| **Timing** | 30% | `1 − |estimated − target| / target` via syllable count / speaking rate |
| **Semantic** | 30% | chrF score against 2 human reference translations (sacrebleu) |
| **Budget** | 20% | `exp(−deficit / 2.0)` — smooth exponential, gradient at every deficit level |
| **Locale** | 10% | Approved/penalized Brazilian Portuguese vocabulary lists |
| **Coherence** | 10% | Numbers and entities preserved consistently across all segments |

**Combined reward ∈ [−1.0, 1.0]**

Anti-hacking penalties applied for: copying English text, single-word outputs, extreme overruns.

---

## Training

### Model
- **Base:** Qwen2.5-1.5B-Instruct (via Unsloth, 4-bit quantized)
- **LoRA:** rank 16, query + value projections, alpha 32, dropout 0.05
- **Trained model:** [varun1235/isosync-qwen2.5-1.5b](https://huggingface.co/varun1235/isosync-qwen2.5-1.5b)

### Algorithm: REINFORCE with Episode Rollouts

```python
# One full episode per update
obs = env.reset(level=current_level)
while not done:
    translation, output_ids, prompt_len = generate(model, obs)
    obs, reward, done, info = env.step({"translation": translation})

    loss = cross_entropy(output_ids) × (reward − baseline)
    loss.backward()   # gradients accumulate across steps

optimizer.step()
baseline = 0.9 × baseline + 0.1 × mean_episode_reward
```

Why not GRPOTrainer with a static dataset? Because IsoSync's budget mechanic means **each translation affects future rewards in the same episode** — a static dataset destroys this dependency.

### 4-Level Curriculum

| Level | Episodes | Segments | Slack | Notes |
|-------|----------|----------|-------|-------|
| L1 | 0–99 | 5 | 0.50s | Learn Portuguese, coarse timing |
| L2 | 100–199 | 8 | 0.30s | More segments, tighter window |
| L3 | 200–399 | 10 | 0.15s | Fine compression required |
| L4 | 400–699 | 10 | 0.05s | Expert level + locale constraints |

The 4-level ramp was critical — the original 2-level curriculum (0.5s → 0.05s) was a 10× difficulty cliff the agent couldn't cross.

---

## Results

| Metric | Value |
|--------|-------|
| Best combined reward | **0.6349** |
| Level 4 mean reward | **+0.49** |
| Training episodes | ~650 (early stopping) |
| Timing reward at L4 | 0.3 – 0.9 per segment |
| Budget deficit at L4 | < 2s across 10 segments |

### Reward Curves

![Reward Curves](isosync/plots/reward_curves_run3.png)

Key observations:
- **Timing reward** (green) rises sharply after the length-explicit prompt was added at L3
- **Combined reward** crosses 0.5 at L3 and holds at L4
- **Budget deficit** (bottom panel) trends from ~15s overrun at L1 to < 2s at L4 — the agent genuinely learns compression

---

## Before vs After

**Without IsoSync** (standard translation, no timing awareness):

```
EN:  "Stir the mixture gently for two minutes."   [target: 2.2s]
PT:  "Mexa a mistura delicadamente por dois minutos inteiros."
     estimated: 3.1s  ← 0.9s overrun, lip sync breaks
```

**With IsoSync** (RL-trained, syllable-aware):

```
EN:  "Stir the mixture gently for two minutes."   [target: 2.2s]
PT:  "Mexa suavemente por dois minutos."
     estimated: 2.1s  ← 0.1s under, fits perfectly ✓
```

---

## End-to-End Demo Pipeline

```
English Video
     │
     ▼
Whisper segmentation (timestamps + duration per sentence)
     │
     ▼
IsoSync Translator  ←  Qwen2.5-1.5B + LoRA adapter
  - syllable-capped prompt
  - budget tracking across segments
     │  Portuguese text (fits time window)
     ▼
XTTS v2  (Coqui TTS, speaker voice cloning)
     │  Portuguese audio (.wav, timed)
     ▼
ffmpeg merge  →  final dubbed audio track
     │
     ▼
Wav2Lip  (lip-sync to original video)
     │
     ▼
output_dubbed.mp4
```

**Real test result:** 15-second English clip → IsoSync + XTTS → **14.98s Portuguese audio**

---

## Demo Video

[Watch the full pipeline demo](https://www.youtube.com/playlist?list=PLkPzTK48kv6FZMGbWPP2w_E7va-XuQNsk)

---

## Repository Structure

```
├── environment.py        # IsoSync OpenEnv environment (reset, step, state)
├── rewards.py            # 5 reward components, fully deterministic
├── data_gen.py           # Synthetic episode generator, 50-sentence bank
├── train.py              # REINFORCE training loop (Unsloth + AdamW)
├── app.py                # FastAPI server (OpenEnv HTTP interface)
├── schemas.py            # Typed Pydantic schemas (Observation, Action, Reward)
├── isosync_translator.py # Drop-in inference module
├── tts_with_isosync.py   # Full pipeline: IsoSync → XTTS
├── openenv.yaml          # OpenEnv spec manifest
├── blog.md               # Full writeup
└── isosync/plots/        # Training reward curves
```

---

## Links

| Resource | Link |
|----------|------|
| HF Space (live env) | [varun1235/dubguard-training](https://huggingface.co/spaces/varun1235/dubguard-training) |
| Trained model | [varun1235/isosync-qwen2.5-1.5b](https://huggingface.co/varun1235/isosync-qwen2.5-1.5b) |
| Demo video | [YouTube Playlist](https://www.youtube.com/playlist?list=PLkPzTK48kv6FZMGbWPP2w_E7va-XuQNsk) |
| Blog | [blog.md](blog.md) |
