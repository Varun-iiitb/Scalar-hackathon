"""
Loads and normalises all five data banks into a unified episode schema.

Each returned episode is a dict with two keys:
  "observation"  — what the agent sees (no answer)
  "ground_truth" — hidden, used only by reward functions

Improvements applied during loading:
  - Deduplication across all banks (same lang + text seen before = skip)
  - Strips numeric suffixes from cultural_bank phrases (". 0", ". 1" …)
  - Strips bracketed indices from clean_bank / tone_bank originals ("[0]" …)
  - Normalises lang-keyed fields (dubbed_hi / correct_dubbed_pt / correct_tone_es …)
    into a single "text" field so downstream code never has to know the lang key
  - Computes estimated dubbed duration from word count + per-language WPM
  - Maps language codes to locale codes consistently
"""

import json
import re
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent

WPM_MAP = {"hi": 110, "pt": 130, "es": 130, "en": 150}
LOCALE_MAP = {"hi": "hi-IN", "pt": "pt-BR", "es": "es-AR"}


# ── helpers ────────────────────────────────────────────────────────────────────

def _wpm_duration(text: str, lang: str) -> float:
    words = len(re.findall(r"\w+", str(text)))
    wpm = WPM_MAP.get(lang, 150)
    return round((words / wpm) * 60.0, 2)


def _strip_index(text: str) -> str:
    """Remove trailing bracketed or dotted numeric indices: '[42]' or '. 42'."""
    text = re.sub(r"\s*\[\d+\]$", "", text.strip())
    text = re.sub(r"\.\s*\d+$", ".", text.strip())
    return text


def _get_lang_field(entry: dict, *prefixes: str) -> str:
    """Return the first non-empty value whose key starts with one of the prefixes."""
    for prefix in prefixes:
        for k, v in entry.items():
            if k.startswith(prefix) and v:
                return str(v)
    return ""


def _build_obs(
    ep_id: str,
    seg_idx: int,
    difficulty: str,
    orig_text: str,
    dubbed_text: str,
    lang: str,
    orig_dur: float,
    gap: float = 1.0,
    max_ratio: float = 1.3,
) -> dict:
    dubbed_dur = _wpm_duration(dubbed_text, lang)
    return {
        "episode_id": ep_id,
        "segment_id": f"seg_{seg_idx:04d}",
        "difficulty_level": difficulty,
        "original": {
            "text": orig_text,
            "start_time": 0.0,
            "end_time": orig_dur,
            "duration_seconds": orig_dur,
        },
        "dubbed": {
            "text": dubbed_text,
            "language_code": lang,
            "locale_code": LOCALE_MAP.get(lang, lang),
            "estimated_duration_seconds": dubbed_dur,
        },
        "next_segment_start_seconds": round(orig_dur + gap, 2),
        "max_allowed_dubbed_duration_seconds": round(orig_dur * max_ratio, 2),
    }


# ── per-bank loaders ──────────────────────────────────────────────────────────

def load_timing_bank() -> list[dict]:
    raw = json.loads((DATA_DIR / "timing_bank.json").read_text(encoding="utf-8"))
    seen, episodes = set(), []
    for i, entry in enumerate(raw):
        lang = entry["lang"]
        orig = entry["original_en"]
        dubbed = _get_lang_field(entry, f"dubbed_{lang}")
        fix = _get_lang_field(entry, f"short_fix_{lang}")
        key = (lang, orig, dubbed)
        if key in seen or not dubbed:
            continue
        seen.add(key)
        orig_dur = float(entry["original_duration"])
        severity = entry.get("severity", "BLOCK")
        episodes.append({
            "observation": _build_obs(
                ep_id=f"timing_{len(episodes):04d}",
                seg_idx=i,
                difficulty="hard" if severity == "BLOCK" else "medium",
                orig_text=orig,
                dubbed_text=dubbed,
                lang=lang,
                orig_dur=orig_dur,
                gap=0.2,      # tight gap to make timing issue visible
                max_ratio=1.0,
            ),
            "ground_truth": {
                "segment_id": i,
                "error_type": "timing_collision",
                "severity": severity,
                "suggested_fix": fix,
                "fix_duration": _wpm_duration(fix, lang),
                "locale_rule": None,
            },
        })
    return episodes


def load_cultural_bank() -> list[dict]:
    raw = json.loads((DATA_DIR / "cultural_bank.json").read_text(encoding="utf-8"))
    seen, episodes = set(), []
    for i, entry in enumerate(raw):
        lang = entry["lang"]
        wrong = _strip_index(entry.get("wrong_phrase", ""))
        correct = entry.get("correct_phrase", "")
        rule = entry.get("rule", "")
        locale = entry.get("locale", LOCALE_MAP.get(lang, lang))
        key = (lang, wrong)
        if key in seen or not wrong:
            continue
        seen.add(key)
        orig_dur = _wpm_duration(correct, "en")
        dubbed_dur = _wpm_duration(wrong, lang)
        episodes.append({
            "observation": {
                "episode_id": f"cultural_{len(episodes):04d}",
                "segment_id": f"seg_{i:04d}",
                "difficulty_level": "hard",
                "original": {
                    "text": correct,
                    "start_time": 0.0,
                    "end_time": orig_dur,
                    "duration_seconds": orig_dur,
                },
                "dubbed": {
                    "text": wrong,
                    "language_code": lang,
                    "locale_code": locale,
                    "estimated_duration_seconds": dubbed_dur,
                },
                "next_segment_start_seconds": round(orig_dur + 1.0, 2),
                "max_allowed_dubbed_duration_seconds": round(orig_dur * 1.5, 2),
            },
            "ground_truth": {
                "segment_id": i,
                "error_type": "cultural_mismatch",
                "severity": "BLOCK",
                "suggested_fix": correct,
                "fix_duration": _wpm_duration(correct, lang),
                "locale_rule": rule,
            },
        })
    return episodes


def load_clean_bank() -> list[dict]:
    raw = json.loads((DATA_DIR / "clean_bank.json").read_text(encoding="utf-8"))
    seen, episodes = set(), []
    for i, entry in enumerate(raw):
        lang = entry["lang"]
        orig = _strip_index(entry["original_en"])
        dubbed = _get_lang_field(entry, f"dubbed_{lang}")
        key = (lang, orig)
        if key in seen or not dubbed:
            continue
        seen.add(key)
        orig_dur = float(entry["original_duration"])
        episodes.append({
            "observation": _build_obs(
                ep_id=f"clean_{len(episodes):04d}",
                seg_idx=i,
                difficulty="easy",
                orig_text=orig,
                dubbed_text=dubbed,
                lang=lang,
                orig_dur=orig_dur,
                gap=1.5,
                max_ratio=1.3,
            ),
            "ground_truth": {
                "segment_id": i,
                "error_type": None,
                "severity": "PASS",
                "suggested_fix": None,
                "fix_duration": orig_dur,
                "locale_rule": None,
            },
        })
    return episodes


def load_mistranslation_bank() -> list[dict]:
    raw = json.loads((DATA_DIR / "mistranslation_bank.json").read_text(encoding="utf-8"))
    seen, episodes = set(), []
    for i, entry in enumerate(raw):
        lang = entry["lang"]
        orig = entry["original_en"]
        correct = _get_lang_field(entry, f"correct_dubbed_{lang}")
        wrong = _get_lang_field(entry, f"wrong_dubbed_{lang}")
        err_desc = entry.get("error_description", "translation_error")
        key = (lang, orig, wrong)
        if key in seen or not wrong:
            continue
        seen.add(key)
        orig_dur = _wpm_duration(orig, "en")
        episodes.append({
            "observation": _build_obs(
                ep_id=f"mistrans_{len(episodes):04d}",
                seg_idx=i,
                difficulty="medium",
                orig_text=orig,
                dubbed_text=wrong,
                lang=lang,
                orig_dur=orig_dur,
                gap=1.0,
                max_ratio=1.5,
            ),
            "ground_truth": {
                "segment_id": i,
                "error_type": "translation_error",
                "severity": "BLOCK",
                "suggested_fix": correct,
                "fix_duration": _wpm_duration(correct, lang),
                "locale_rule": None,
            },
        })
    return episodes


def load_tone_bank() -> list[dict]:
    raw = json.loads((DATA_DIR / "tone_bank.json").read_text(encoding="utf-8"))
    seen, episodes = set(), []
    for i, entry in enumerate(raw):
        lang = entry["lang"]
        orig = _strip_index(entry["original_en"])
        correct = _get_lang_field(entry, f"correct_tone_{lang}")
        wrong = _get_lang_field(entry, f"wrong_tone_{lang}")
        register = entry.get("original_register", "neutral")
        key = (lang, orig, wrong)
        if key in seen or not wrong:
            continue
        seen.add(key)
        orig_dur = _wpm_duration(orig, "en")
        episodes.append({
            "observation": _build_obs(
                ep_id=f"tone_{len(episodes):04d}",
                seg_idx=i,
                difficulty="medium",
                orig_text=orig,
                dubbed_text=wrong,
                lang=lang,
                orig_dur=orig_dur,
                gap=1.0,
                max_ratio=1.5,
            ),
            "ground_truth": {
                "segment_id": i,
                "error_type": "tone_mismatch",
                "severity": "WARN",
                "suggested_fix": correct,
                "fix_duration": _wpm_duration(correct, lang),
                "locale_rule": f"register:{register}",
            },
        })
    return episodes


# ── public API ────────────────────────────────────────────────────────────────

def load_all(shuffle: bool = True, seed: int = 42) -> list[dict]:
    """Load, deduplicate, and optionally shuffle all five banks."""
    episodes = (
        load_timing_bank()
        + load_cultural_bank()
        + load_clean_bank()
        + load_mistranslation_bank()
        + load_tone_bank()
    )
    if shuffle:
        random.Random(seed).shuffle(episodes)
    return episodes


def load_by_error_type(error_type: str | None, shuffle: bool = True, seed: int = 42) -> list[dict]:
    """Filter episodes by ground-truth error type (None = clean/PASS segments)."""
    return [
        ep for ep in load_all(shuffle=shuffle, seed=seed)
        if ep["ground_truth"]["error_type"] == error_type
    ]
