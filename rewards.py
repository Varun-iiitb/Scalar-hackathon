"""
IsoSync reward functions — four independent, fully deterministic scorers.
No LLM calls. No subjectivity. Pure Python.

Public API:
    timing_reward(translation, target_duration, language) -> float [0, 1]
    semantic_reward(translation, reference)               -> float [0, 1]
    budget_reward(budget_deficit)                         -> float [0, 1]
    locale_reward(translation, locale)                    -> float [0, 1]
    compute_reward(translation, segment, budget_deficit)  -> float [-1, 1]
"""

import unicodedata

import pyphen
from sacrebleu.metrics import CHRF

# ── Speaking rates (syllables per second) ─────────────────────────────────────
SPEAKING_RATES = {
    "Hindi":      4.5,
    "Portuguese": 5.2,
}

# ── Locale wordlists ───────────────────────────────────────────────────────────
# Words common in natural speech for each locale — higher fraction = more locale-natural
LOCALE_WORDS = {
    "Mumbai": {
        # Common Mumbai Hindi words/particles (Devanagari)
        "यार", "अरे", "बस", "भाई", "हाँ", "नहीं", "तो", "क्या", "अच्छा",
        "बिलकुल", "थोड़ा", "जरा", "एकदम", "चलो", "देखो", "लगता", "मतलब",
        "सही", "पक्का", "कमाल", "ठीक", "वैसे", "दरअसल", "खैर", "फिर",
        "जब", "तब", "यहाँ", "वहाँ", "कब", "कैसे", "क्यों", "कहाँ",
    },
    "Brazil": {
        # Common Brazilian Portuguese informal words
        "cara", "legal", "bacana", "então", "né", "tá", "gente", "muito",
        "bem", "aqui", "agora", "também", "ainda", "só", "já", "ótimo",
        "show", "tipo", "vamos", "sempre", "pode", "isso", "assim", "aí",
        "bom", "boa", "mais", "mas", "com", "para", "por", "uma", "um",
    },
}

# ── Singleton CHRF metric (thread-safe, expensive to init) ───────────────────
_CHRF = CHRF()

# ── Pyphen dictionaries ────────────────────────────────────────────────────────
# Hindi doesn't have a pyphen dict; we use a Unicode syllable approximation
_pyphen_pt = pyphen.Pyphen(lang="pt_BR")


def _count_syllables_hindi(text: str) -> int:
    """
    Approximate syllable count for Hindi (Devanagari) text.

    Each vowel letter / vowel sign / standalone consonant cluster
    in Devanagari maps roughly to one syllable. This is a heuristic.
    """
    count = 0
    for ch in text:
        cat = unicodedata.category(ch)
        name = unicodedata.name(ch, "")
        # Devanagari vowels (Lo), vowel signs (Mc/Mn), independent vowels
        if cat in ("Lo", "Mc", "Mn") and "DEVANAGARI" in name:
            if "VOWEL SIGN" in name or "VOWEL" in name or cat == "Lo":
                count += 1
    return max(1, count)


def _count_syllables_portuguese(text: str) -> int:
    """Count syllables in Portuguese text using pyphen hyphenation."""
    count = 0
    for word in text.split():
        # Strip punctuation
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

    Estimates spoken duration via syllable count / speaking rate, then
    penalises proportionally to the absolute deviation from target.

    Returns float in [0.0, 1.0].
    """
    rate = SPEAKING_RATES.get(language, 5.0)
    syllables = count_syllables(translation, language)
    estimated = syllables / rate
    score = max(0.0, 1.0 - abs(estimated - target_duration) / target_duration)
    return round(score, 4)


# ── REWARD 2: Semantic ────────────────────────────────────────────────────────

def semantic_reward(translation: str, reference: str) -> float:
    """
    Score semantic faithfulness using chrF (character n-gram F-score).

    chrF is language-agnostic and works well for morphologically rich
    languages like Hindi. No model call — pure string matching.

    Returns float in [0.0, 1.0].
    """
    score = _CHRF.sentence_score(translation, [reference]).score / 100.0
    return round(max(0.0, min(1.0, score)), 4)


# ── REWARD 3: Budget ──────────────────────────────────────────────────────────

def budget_reward(budget_deficit: float) -> float:
    """
    Score how well the agent is managing the cumulative timing budget.

    budget_deficit = total seconds the episode has overrun so far.
      < 0.2s  → perfect (1.0)
      > 2.0s  → fail (0.0)
      linear interpolation between

    Returns float in [0.0, 1.0].
    """
    if budget_deficit < 0.2:
        return 1.0
    if budget_deficit > 2.0:
        return 0.0
    return round(max(0.0, 1.0 - budget_deficit / 2.0), 4)


# ── REWARD 4: Locale ─────────────────────────────────────────────────────────

def locale_reward(translation: str, locale: str) -> float:
    """
    Score how locale-appropriate the translation sounds.

    Computed as the fraction of words in the translation that appear
    in the locale's approved wordlist. Even one matching word gives
    partial credit; the idea is to encourage natural locale-specific phrasing.

    Returns float in [0.0, 1.0].
    """
    wordlist = LOCALE_WORDS.get(locale, set())
    if not wordlist:
        return 0.5  # unknown locale — neutral score

    words = [w.strip("।.,!?\"'()") for w in translation.split()]
    if not words:
        return 0.0

    matches = sum(1 for w in words if w in wordlist)
    return round(min(1.0, matches / max(1, len(words))), 4)


# ── Anti-hacking penalties ────────────────────────────────────────────────────

def _anti_hacking_penalty(
    translation: str,
    original_english: str,
    language: str,
    estimated_duration: float,
    target_duration: float,
) -> float:
    """
    Return a negative penalty if the model is gaming the reward.

    Checks (all additive):
      -1.0  if translation == original English (copy-paste)
      -1.0  if translation is < 2 words (trivially short)
      -0.5  if estimated duration > 1.8× target (absurdly long)
      -0.3  if language is Hindi but no Devanagari script found
    """
    penalty = 0.0

    if translation.strip() == original_english.strip():
        penalty -= 1.0

    if len(translation.split()) < 2:
        penalty -= 1.0

    if estimated_duration > target_duration * 1.8:
        penalty -= 0.5

    # For Hindi, check that at least some Devanagari characters exist
    if language == "Hindi":
        has_devanagari = any("ऀ" <= ch <= "ॿ" for ch in translation)
        if not has_devanagari:
            penalty -= 0.3

    return penalty


# ── Combined reward ───────────────────────────────────────────────────────────

def compute_reward(
    translation: str,
    segment: dict,
    budget_deficit: float,
) -> float:
    """
    Compute the weighted combined reward for one translation step.

    Weights:
      timing   35%
      semantic 35%
      budget   20%
      locale   10%
      + anti-hacking penalties (additive, can make total negative)

    Args:
        translation:   the model's output string
        segment:       segment dict from data_gen.generate_episode()
        budget_deficit: cumulative seconds over budget so far this episode

    Returns:
        float clamped to [-1.0, 1.0]
    """
    language        = segment["target_language"]
    target_duration = segment["original_duration"]
    reference       = segment["reference_translation"]
    locale          = segment["locale"]
    original_en     = segment["original_text"]

    t_reward = timing_reward(translation, target_duration, language)
    s_reward = semantic_reward(translation, reference)
    b_reward = budget_reward(budget_deficit)
    l_reward = locale_reward(translation, locale)

    # Estimated duration (needed for anti-hacking check)
    rate      = SPEAKING_RATES.get(language, 5.0)
    syllables = count_syllables(translation, language)
    estimated = syllables / rate

    penalty = _anti_hacking_penalty(
        translation, original_en, language, estimated, target_duration
    )

    combined = (
        0.35 * t_reward
        + 0.35 * s_reward
        + 0.20 * b_reward
        + 0.10 * l_reward
        + penalty
    )

    return round(max(-1.0, min(1.0, combined)), 4)


if __name__ == "__main__":
    # Quick sanity check
    seg = {
        "original_text":         "Add a pinch of salt to the boiling water.",
        "original_duration":     2.5,
        "target_language":       "Hindi",
        "locale":                "Mumbai",
        "reference_translation": "उबलते पानी में एक चुटकी नमक डालें।",
    }
    translation = "उबलते पानी में एक चुटकी नमक डालें।"

    print("timing :", timing_reward(translation, seg["original_duration"], seg["target_language"]))
    print("semantic:", semantic_reward(translation, seg["reference_translation"]))
    print("budget :", budget_reward(0.0))
    print("locale :", locale_reward(translation, seg["locale"]))
    print("combined:", compute_reward(translation, seg, 0.0))
