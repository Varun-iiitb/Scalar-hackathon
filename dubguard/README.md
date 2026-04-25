---
title: DubGuard
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# DubGuard

**An RL environment for LLM-based quality control of AI-dubbed video segments.**

DubGuard trains a language model to act as an automated QC reviewer: given a dubbed script segment with timing metadata, the agent must decide whether to **BLOCK**, **WARN**, or **PASS** the segment — and if flagging it, propose a timing-safe fix.

---

## Problem Motivation

AI dubbing pipelines (text-to-speech, machine translation, voice cloning) produce errors that are subtle and domain-specific:

- **Timing collisions** — the dubbed audio runs longer than the available gap, crashing into the next segment
- **Mistranslations** — semantic drift that changes meaning (idioms, negations, numbers)
- **Tone mismatches** — register shifts between formal/informal, calm/urgent
- **Cultural mismatches** — locale-specific norms violated (e.g., Hindi honorific dropped, Brazilian Portuguese register wrong)

Human QC reviewers catch these but are slow and expensive. DubGuard turns this into an RL problem: the agent learns to catch errors reliably while minimising false positives on clean segments.

---

## Environment Design

The environment implements the **OpenEnv** interface (`Environment` base class) and is served via **FastAPI**.

### Observation (what the agent sees)

```json
{
  "episode_id": "ep_mistranslation_023",
  "segment_id": "seg_0017",
  "difficulty_level": "medium",
  "original": {
    "text": "She never said she loved him.",
    "start_time": 12.4,
    "end_time": 14.8,
    "duration_seconds": 2.4
  },
  "dubbed": {
    "text": "उसने कहा कि वो उससे प्यार करती है।",
    "language_code": "hi",
    "locale_code": "hi-IN",
    "estimated_duration_seconds": 3.1
  },
  "next_segment_start_seconds": 14.9,
  "max_allowed_dubbed_duration_seconds": 2.4
}
```

### Action (what the agent returns)

```json
{
  "segment_id": 17,
  "error_type": "mistranslation",
  "severity": "BLOCK",
  "reason": "Negation 'never' lost — translation says she DID declare love",
  "suggested_fix": "उसने कभी नहीं कहा कि वो उससे प्यार करती है।",
  "estimated_fix_duration": 2.3
}
```

### Reward Components

| Component | Weight | Description |
|---|---|---|
| Detection | 40% | Correct error type identification |
| Severity | 20% | Correct BLOCK/WARN/PASS level |
| Correction | 25% | Semantic similarity of suggested fix + timing penalty |
| Cultural | 15% | Correct locale-specific rule application |
| False positive penalty | additive −0.3 | Penalises flagging clean segments |

**Reward range: −0.3 to 1.0**

### Dataset

195 unique episodes across 5 error types:

| Bank | Count | Error Type |
|---|---|---|
| Timing | 60 | `timing_collision` |
| Mistranslation | 84 | `mistranslation` |
| Tone | 16 | `tone_mismatch` |
| Cultural | 20 | `cultural_mismatch` |
| Clean | 15 | `none` (PASS expected) |

Languages: Hindi (`hi-IN`), Brazilian Portuguese (`pt-BR`), Argentine Spanish (`es-AR`)

### Curriculum

Training uses three difficulty phases:

1. **Easy** (steps 0–199) — timing collisions and obvious mistranslations
2. **Medium** (steps 200–499) — tone/cultural errors added
3. **Hard** (steps 500–999) — all error types, subtle cases

---

## API Reference

The environment is exposed as a REST API (FastAPI, port 7860).

### `GET /info`
Returns environment metadata: observation/action space, reward range, episode count.

### `POST /reset`
Start a new episode. Optional body: `{"difficulty": "easy"|"medium"|"hard"}`

Returns `{"observation": {...}}`

### `POST /step`
Submit a QC action. Body: the action JSON (see above).

Returns `{"reward": {"detection_score": ..., "combined_score": ...}, "done": true}`

### `GET /state`
Returns the current observation without advancing the episode.

### `GET /health`
Liveness probe — returns `{"status": "ok", "episodes_loaded": 195}`

---

## Quick Start

### Using the client (no server imports)

```python
from client import DubGuardClient

client = DubGuardClient("https://your-space.hf.space")

obs = client.reset(difficulty="easy")
print(obs["dubbed"]["text"])

result = client.step({
    "segment_id": 42,
    "error_type": "timing_collision",
    "severity": "BLOCK",
    "reason": "Dubbed audio runs 1.4s over the available gap",
    "suggested_fix": "Shortened version that fits in 2.1s",
    "estimated_fix_duration": 2.0,
})
print(result["reward"]["combined_score"])
```

### Running locally

```bash
git clone https://huggingface.co/spaces/your-username/dubguard
cd dubguard
pip install fastapi uvicorn sentence-transformers
python app.py
# Server running at http://localhost:7860
```

### Training the agent

```bash
pip install -r training/requirements.txt
python training/train.py
# Runs GRPO curriculum: easy → medium → hard
# Saves checkpoints every 100 steps
# Pushes final model to HF Hub
```

---

## Stack

| Component | Library |
|---|---|
| LLM backbone | Qwen2.5-3B-Instruct (4-bit via Unsloth) |
| RL algorithm | GRPO (TRL `GRPOTrainer`) |
| PEFT | LoRA rank-16, alpha-32 |
| Semantic similarity | `paraphrase-multilingual-MiniLM-L12-v2` |
| Server | FastAPI + Uvicorn |
| Deployment | HuggingFace Spaces (Docker) |

---

## Project Structure

```
dubguard/
├── app.py                  # FastAPI server (OpenEnv HTTP interface)
├── client.py               # HTTP client — no server imports
├── openenv.yaml            # Environment manifest
├── environment/
│   ├── base.py             # OpenEnv Environment + MCPEnvironment base classes
│   └── env.py              # DubGuardEnvironment (inherits Environment)
├── rewards/
│   ├── combined.py         # Orchestrates 5 reward components
│   ├── detection.py        # Error type identification (40%)
│   ├── severity.py         # BLOCK/WARN/PASS level (20%)
│   ├── correction.py       # Fix quality + timing (25%)
│   ├── cultural.py         # Locale-rule compliance (15%)
│   └── false_positive.py   # Penalty for flagging clean segments (−0.3)
├── data/
│   ├── loader.py           # Loads and deduplicates all 5 banks
│   └── *.json              # timing, cultural, clean, mistranslation, tone banks
└── training/
    ├── train.py            # GRPO curriculum training script
    ├── evaluate.py         # Before/after comparison with plots
    ├── agent.py            # DubGuardAgent (Unsloth + robust JSON parsing)
    └── prompts.py          # System prompt + observation formatter
```
