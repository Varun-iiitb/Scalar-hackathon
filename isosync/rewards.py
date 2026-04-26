"""
IsoSync reward functions — five independent, fully deterministic scorers.
No LLM calls. No subjectivity. Pure Python.

Public API:
    timing_reward(translation, target_duration, language)      -> float [0, 1]
    semantic_reward(translation, references)                   -> float [0, 1]
    budget_reward(budget_deficit)                              -> float [0, 1]
    locale_reward(translation, locale)                         -> float [0, 1]
    coherence_reward(episode_translations)                     -> float [0, 1]
    compute_reward(translation, segment, budget_deficit,
                   coherence_score)                            -> float [-1, 1]

Reward weights:
    timing    30%
    semantic  30%
    budget    20%
    locale    10%
    coherence 10%
    + anti-hacking penalties (additive)
"""

import re
import math
from sacrebleu.metrics import CHRF
import pyphen

# ── Speaking rates ─────────────────────────────────────────────────────────────
SPEAKING_RATES = {
    "Hindi":      4.5,
    "Portuguese": 5.2,
}

# ── Devanagari Unicode character sets ─────────────────────────────────────────
_HINDI_VOWEL_MATRAS = {
    'ा', 'ि', 'ी', 'ु', 'ू',
    'ृ', 'ॄ', 'े', 'ै', 'ो',
    'ौ', 'ॅ', 'ॆ', 'ॉ', 'ॊ',
}
_HINDI_INDEPENDENT_VOWELS = {
    'अ', 'आ', 'इ', 'ई', 'उ',
    'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ',
}

# ── Number word → Arabic numeral map (for coherence checking) ─────────────────
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "fifteen": "15", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "hundred": "100", "thousand": "1000",
}

# ── Locale wordlists ───────────────────────────────────────────────────────────
LOCALE_WORDLISTS = {
    "Brazil": {
        "approved": [
            "você", "tudo", "muito", "legal", "gente", "cara",
            "ótimo", "incrível", "rápido", "fácil", "agora",
            "então", "super", "bacana", "show", "né", "tá",
            "bom", "bem", "fazer", "vamos", "aqui", "isso",
            "também", "ainda", "só", "já", "assim", "aí",
            "sempre", "tipo", "isso", "aqui", "para", "com",
            "uma", "que", "por", "não", "mais", "como", "seu",
            "sua", "este", "esta", "esse", "essa", "seu", "sua",
        ],
        "penalized": ["vosotros", "usted", "hola", "gracias"],
    },
    "Mumbai": {
        "approved": [
            "यह", "है", "और", "के", "में", "को", "से",
            "पर", "एक", "हम", "आप", "वह", "जो", "कि",
            "बहुत", "अच्छा", "जल्दी", "आसान", "करें",
            "देखें", "यहां", "अब", "तो", "भी", "यार",
            "अरे", "भाई", "सही", "ठीक", "चलो", "बस",
        ],
        "penalized": [],
    },
}

# Keep backward-compatible alias
LOCALE_WORDS = {k: set(v["approved"]) for k, v in LOCALE_WORDLISTS.items()}

# ── Singletons ────────────────────────────────────────────────────────────────
_CHRF     = CHRF()
_pyphen_pt = pyphen.Pyphen(lang="pt_BR")


# ── Syllable counting ─────────────────────────────────────────────────────────

def _count_syllables_hindi(text: str) -> int:
    count = 0
    chars = list(text)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in _HINDI_VOWEL_MATRAS or ch in _HINDI_INDEPENDENT_VOWELS:
            count += 1
        elif 'क' <= ch <= 'ह':
            next_ch = chars[i + 1] if i + 1 < len(chars) else ''
            if next_ch not in _HINDI_VOWEL_MATRAS and next_ch != '्':
                count += 1
        i += 1
    return max(1, count)


def _count_syllables_portuguese(text: str) -> int:
    count = 0
    for word in text.split():
        clean = "".join(ch for ch in word if ch.isalpha())
        if not clean:
            continue
        hyphenated = _pyphen_pt.inserted(clean.lower(), hyphen="|")
        count += hyphenated.count("|") + 1
    return max(1, count)


def count_syllables(text: str, language: str) -> int:
    if language == "Hindi":
        return _count_syllables_hindi(text)
    return _count_syllables_portuguese(text)


# ── REWARD 1: Timing ──────────────────────────────────────────────────────────

def timing_reward(translation: str, target_duration: float, language: str) -> float:
    """
    Score how well the translation fits the target time window.
    Estimates spoken duration = syllable_count / speaking_rate.
    Returns float in [0.0, 1.0].
    """
    rate      = SPEAKING_RATES.get(language, 5.0)
    syllables = count_syllables(translation, language)
    estimated = syllables / rate
    score     = max(0.0, 1.0 - abs(estimated - target_duration) / target_duration)
    return round(score, 4)


# ── REWARD 2: Semantic ────────────────────────────────────────────────────────

def semantic_reward(translation: str, references: list[str]) -> float:
    """
    Score semantic faithfulness using chrF against multiple references.
    Multi-reference chrF picks the best match automatically.
    Returns float in [0.0, 1.0].
    """
    if not references:
        return 0.0
    score = _CHRF.sentence_score(translation, references).score / 100.0
    return round(max(0.0, min(1.0, score)), 4)


# ── REWARD 3: Budget ──────────────────────────────────────────────────────────

def budget_reward(budget_deficit: float) -> float:
    """
    Score how well the agent manages the cumulative timing deficit.
    Smooth exponential decay: every second of improvement raises the reward,
    so the agent has gradient at every deficit level instead of plateaus.
        deficit  0.0s →  reward 1.000
        deficit  1.0s →  reward 0.607
        deficit  2.0s →  reward 0.368
        deficit  3.0s →  reward 0.223
        deficit  5.0s →  reward 0.082
        deficit 10.0s →  reward 0.007
    Returns float in [0.0, 1.0].
    """
    if budget_deficit < 0:
        budget_deficit = 0.0
    return round(math.exp(-budget_deficit / 2.0), 4)


# ── REWARD 4: Locale ─────────────────────────────────────────────────────────

def locale_reward(translation: str, locale: str) -> float:
    """
    Score locale-appropriateness using approved/penalized word lists.
    Gives partial credit (0.05 floor) so agent always has a gradient.
    Returns float in [0.0, 1.0].
    """
    if locale not in LOCALE_WORDLISTS:
        return 0.1

    wordlist = LOCALE_WORDLISTS[locale]
    words = [w.strip("।.,!?\"'()").lower() for w in translation.split()]
    if not words:
        return 0.0

    approved_count   = sum(1 for w in words if w in wordlist["approved"])
    penalized_count  = sum(1 for w in words if w in wordlist["penalized"])

    base_score = approved_count / max(len(words), 1)
    penalty    = penalized_count * 0.2
    score      = max(0.05, base_score - penalty)
    return round(min(1.0, score), 4)


# ── REWARD 5: Coherence ───────────────────────────────────────────────────────

def coherence_reward(episode_translations: list[dict]) -> float:
    """
    Score consistency of numbers across all translations completed so far.

    For each segment, we extract numeric quantities from the original English
    (both digit form like "30" and word form like "thirty") and check whether
    the corresponding Arabic numeral appears in the translation. Agents that
    drop numbers entirely get penalised — agents that consistently preserve
    quantities get full credit.

    This rewards durable world-modeling across the episode, not just
    per-segment quality.

    Args:
        episode_translations: list of {"original": str, "translation": str}
            accumulated so far this episode.

    Returns float in [0.0, 1.0].
    """
    if not episode_translations:
        return 1.0

    preserved = 0
    total     = 0

    for entry in episode_translations:
        original    = entry.get("original", "").lower()
        translation = entry.get("translation", "")

        # Collect all numeric quantities mentioned in the original
        quantities = set()

        # Direct digit sequences  e.g. "30"
        for m in re.findall(r'\b\d+\b', original):
            quantities.add(m)

        # Number words  e.g. "thirty" → "30"
        for word, digit in _NUMBER_WORDS.items():
            if re.search(r'\b' + word + r'\b', original):
                quantities.add(digit)

        for qty in quantities:
            total += 1
            # Accept either the digit or its Devanagari/standard form
            if qty in translation:
                preserved += 1

    if total == 0:
        # No numbers in any segment yet — neutral score
        return 0.8

    return round(preserved / total, 4)


# ── Anti-hacking penalties ────────────────────────────────────────────────────

def _anti_hacking_penalty(
    translation: str,
    original_english: str,
    language: str,
    estimated_duration: float,
    target_duration: float,
) -> float:
    penalty = 0.0
    if translation.strip() == original_english.strip():
        penalty -= 1.0
    if len(translation.split()) < 2:
        penalty -= 1.0
    if estimated_duration > target_duration * 1.8:
        penalty -= 0.5
    if language == "Hindi":
        if not any('ऀ' <= ch <= 'ॿ' for ch in translation):
            penalty -= 0.3
    return penalty


# ── Combined reward ───────────────────────────────────────────────────────────

def compute_reward(
    translation: str,
    segment: dict,
    budget_deficit: float,
    coherence_score: float = 0.8,
) -> float:
    """
    Weighted combined reward for one translation step.

    Weights:
        timing    30%   (fits the time window)
        semantic  30%   (preserves meaning vs reference)
        budget    20%   (cumulative deficit after banking)
        locale    10%   (sounds natural in locale)
        coherence 10%   (numbers/entities consistent across episode)
        + anti-hacking penalties (additive, can make total negative)

    Args:
        translation:     model output string
        segment:         segment dict with original_text, original_duration,
                         target_language, locale, reference_translations
        budget_deficit:  effective deficit AFTER banking (seconds over budget)
        coherence_score: pre-computed coherence score from coherence_reward()

    Returns float clamped to [-1.0, 1.0].
    """
    language        = segment["target_language"]
    target_duration = segment["original_duration"]
    references      = segment["reference_translations"]
    locale          = segment["locale"]
    original_en     = segment["original_text"]

    t_reward = timing_reward(translation, target_duration, language)
    s_reward = semantic_reward(translation, references)
    b_reward = budget_reward(budget_deficit)
    l_reward = locale_reward(translation, locale)

    rate      = SPEAKING_RATES.get(language, 5.0)
    estimated = count_syllables(translation, language) / rate

    penalty = _anti_hacking_penalty(
        translation, original_en, language, estimated, target_duration
    )

    combined = (
        0.30 * t_reward
        + 0.30 * s_reward
        + 0.20 * b_reward
        + 0.10 * l_reward
        + 0.10 * coherence_score
        + penalty
    )

    return round(max(-1.0, min(1.0, combined)), 4)


if __name__ == "__main__":
    seg = {
        "original_text":          "This recipe is ready in just ten minutes.",
        "original_duration":      2.3,
        "target_language":        "Portuguese",
        "locale":                 "Brazil",
        "reference_translations": [
            "Esta receita fica pronta em apenas dez minutos.",
            "Essa receita está pronta em só dez minutos.",
        ],
    }
    t = "Esta receita fica pronta em apenas 10 minutos."
    history = [{"original": seg["original_text"], "translation": t}]

    print("timing   :", timing_reward(t, seg["original_duration"], seg["target_language"]))
    print("semantic :", semantic_reward(t, seg["reference_translations"]))
    print("budget   :", budget_reward(0.0))
    print("locale   :", locale_reward(t, seg["locale"]))
    print("coherence:", coherence_reward(history))
    print("combined :", compute_reward(t, seg, 0.0, coherence_reward(history)))
