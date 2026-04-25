"""
DubGuardAgent — wraps Qwen2.5-3B-Instruct (loaded via Unsloth) and exposes
an act() method that takes an observation dict and returns an action dict.

JSON parsing is intentionally robust: three strategies are tried in order
before falling back to a safe default that never crashes the training loop.
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.prompts import SYSTEM_PROMPT, format_observation

# ── model loading ─────────────────────────────────────────────────────────────

try:
    from unsloth import FastLanguageModel
    _UNSLOTH = True
except ImportError:
    _UNSLOTH = False


class DubGuardAgent:
    """
    Wraps Qwen2.5-3B-Instruct loaded through Unsloth with 4-bit quantization.

    get_model() / get_tokenizer() expose the underlying objects so that
    GRPOTrainer can attach LoRA and run its own forward passes.
    """

    def __init__(
        self,
        model_name: str = "unsloth/Qwen2.5-3B-Instruct",
        max_seq_length: int = 1024,
        load_in_4bit: bool = True,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length

        if _UNSLOTH:
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                dtype=None,             # auto-detect bf16 / fp16
                fast_inference=False,   # keep False; training sets this via for_training()
            )
        else:
            # Fallback for environments without Unsloth (e.g. CPU dev machines)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )

        self._tokenizer.padding_side = "left"

    # ── public API ────────────────────────────────────────────────────────────

    def act(self, observation: dict, temperature: float = 0.7, max_new_tokens: int = 256) -> dict:
        """
        Format the observation, run inference, parse the JSON response.
        Always returns a valid action dict — never raises.
        """
        import torch

        formatted = format_observation(observation)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": formatted},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        if _UNSLOTH:
            FastLanguageModel.for_inference(self._model)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return self._parse_action(raw, observation)

    def get_model(self):
        return self._model

    def get_tokenizer(self):
        return self._tokenizer

    # ── JSON parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_action(raw_output: str, observation: dict) -> dict:
        """
        Try three increasingly lenient strategies to extract a valid JSON action.
        Falls back to a safe PASS default so the training loop never crashes.
        """
        seg_id = observation.get("segment_id", "seg_0000")
        seg_num = (
            int("".join(filter(str.isdigit, str(seg_id))) or "0")
            if isinstance(seg_id, str)
            else int(seg_id)
        )

        # Strategy 1 — direct parse
        try:
            parsed = json.loads(raw_output)
            parsed.setdefault("_parse_failed", False)
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2 — first { to last }
        try:
            start = raw_output.index("{")
            end   = raw_output.rindex("}") + 1
            parsed = json.loads(raw_output[start:end])
            parsed.setdefault("_parse_failed", False)
            return parsed
        except (ValueError, json.JSONDecodeError):
            pass

        # Strategy 3 — regex for any {...} block (handles nested braces poorly, but worth trying)
        try:
            match = re.search(r"\{[\s\S]*\}", raw_output)
            if match:
                parsed = json.loads(match.group())
                parsed.setdefault("_parse_failed", False)
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Safe default — PASS with a parse-error marker
        return {
            "segment_id":             seg_num,
            "error_type":             None,
            "severity":               "PASS",
            "reason":                 "parse_error: model output could not be decoded as JSON",
            "suggested_fix":          "",
            "estimated_fix_duration": 0.0,
            "_parse_failed":          True,
        }
