# IsoSync: RL-Trained Time-Aware Dubbing with Temporal Alignment

IsoSync is an OpenEnv-compatible reinforcement learning environment for training LLMs to produce Brazilian Portuguese translations that preserve meaning, satisfy strict timing constraints, and maintain natural speech rhythm for dubbing.

Most dubbing pipelines translate first and correct timing afterward. IsoSync instead treats timing and rhythm as part of the generation process itself.

---

## What the Agent Learns

Given an English segment and a target duration, the agent generates a Portuguese translation that must simultaneously:

- Fit within a strict time window
- Preserve semantic meaning
- Maintain fluency in Brazilian Portuguese
- Remain consistent across segments

---

## Reward Design

The environment uses five independent reward components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Timing    | 30%    | Syllable-based duration vs target |
| Semantic  | 30%    | chrF against two references |
| Budget    | 20%    | Smooth penalty for accumulated timing deficit |
| Locale    | 10%    | Brazilian Portuguese vocabulary constraints |
| Coherence | 10%    | Entity and number consistency across segments |

Additional penalties discourage copying English text or producing degenerate outputs.

**Combined reward range: [-1.0, 1.0]**

---

## Training Setup

- **Base model:** Qwen2.5-1.5B-Instruct
- **Fine-tuning:** 4-bit LoRA (rank 16 on query/value projections)
- **Objective:** REINFORCE-style
```
loss = CE × (reward − EMA baseline)
```
- **Optimizer:** AdamW with StepLR

### Curriculum

| Level | Episodes | Segments | Slack |
|-------|----------|----------|-------|
| L1    | 0–99     | 5        | 0.50s |
| L2    | 100–199  | 8        | 0.30s |
| L3    | 200–399  | 10       | 0.15s |
| L4    | 400–699  | 10       | 0.05s with locale constraints |

---

## Results

- **Best combined reward:** 0.6349
- **Level 4 mean reward:** +0.49
- **Early stopping** at approximately episode 650
- **Model:** [varun1235/isosync-qwen2.5-1.5b](https://huggingface.co/varun1235/isosync-qwen2.5-1.5b)

![Reward Curves](isosync/plots/reward_curves_run3.png)

---

## Demo Pipeline

1. English text with target duration
2. IsoSync translation with syllable-capped prompting
3. Qwen2.5 base model with LoRA adapter
4. XTTS v2 speech synthesis
5. Fallback compression when necessary
6. Chunk merging with final timing correction

In a 15-second test case, the system automatically adjusted outputs and produced final audio of **14.98 seconds**, aligned with the target duration.

---

## Temporal Structure Preservation

IsoSync models not only total duration but also speech rhythm.

- Input is segmented into semantically meaningful units
- Each segment is translated and synthesized independently
- Natural pauses between segments are preserved during merging
- Final output satisfies:
  - per-segment timing constraints
  - inter-segment pacing
  - overall duration limits

This produces audio that is naturally paced rather than uniformly compressed, which is critical for realistic dubbing.

---

## Example Output

**Portuguese translation:**

```
Três dicas para vídeos falando à câmera.
Primeiro, luzes atrás da pessoa, voltadas para a câmera.
Segundo, luz no lado da sombra, 45 graus atrás.
Terceiro, luz atrás da pessoa, apontada para o fundo.
```

**Final duration:** 14.98 seconds (target: 15.0 seconds)

---

## Significance

IsoSync addresses a practical limitation in multilingual media generation: producing translations that are not only accurate and fluent, but also **temporally aligned for speech**.

This has direct applications in:

- Short-form video dubbing
- Creator tooling
- Multilingual content pipelines
- Lip-sync systems using XTTS and Wav2Lip

---

## Hackathon Alignment

- OpenEnv-compatible environment
- Multi-component reward design resistant to exploitation
- End-to-end reinforcement learning pipeline
- Clear evidence of reward improvement
- Demonstrated timing alignment in real outputs

IsoSync shows that large language models can be trained to jointly optimize language, timing, and rhythm — rather than relying on post-processing corrections.

---

## Demo Video

Watch the full end-to-end pipeline in action:

[IsoSync Demo Playlist](https://www.youtube.com/playlist?list=PLkPzTK48kv6FZMGbWPP2w_E7va-XuQNsk)
