"""
IsoSync reward functions — four independent, fully deterministic scorers.
No LLM calls. No subjectivity. Pure Python.

Public API:
    timing_reward(translation, target_duration, language) -> float [0, 1]
    semantic_reward(translation, references)              -> float [0, 1]
    budget_reward(budget_deficit)                         -> float [0, 1]
    locale_reward(translation, locale)                    -> float [0, 1]
    compute_reward(translation, segment, budget_deficit)  -> float [-1, 1]
"""

from sacrebleu.metrics import CHRF
import pyphen

# ── Speaking rates ─────────────────────────────────────────────────────────────
SPEAKING_RATES = {
    "Hindi":      4.5,
    "Portuguese": 5.2,
}

# ── Devanagari Unicode ranges ─────────────────────────────────────────────────
# Explicit vowel matras (dependent vowel signs) — each counts as one syllable
_HINDI_VOWEL_MATRAS = {
    'ा', 'ि', 'ी', 'ु', 'ू',
    'ृ', 'ॄ', 'े', 'ै', 'ो',
    'ौ', 'ॅ', 'ॆ', 'ॉ', 'ॊ',
}
# Independent vowels (standalone vowel letters)
_HINDI_INDEPENDENT_VOWELS = {
    'अ', 'आ', 'इ', 'ई', 'उ',
    'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ',
}

# ── Locale wordlists ───────────────────────────────────────────────────────────
LOCALE_WORDS = {
    "Mumbai": {
        "यार", "अरे", "बस", "भाई", "हाँ", "नहीं", "तो", "क्या", "अच्छा",
        "बिलकुल", "थोड़ा", "जरा", "एकदम", "चलो", "देखो", "मतलब",
        "सही", "पक्का", "कमाल", "ठीक", "वैसे", "दरअसल", "फिर",
    },
    "Brazil": {
        "cara", "legal", "bacana", "então", "né", "tá", "gente", "muito",
        "bem", "aqui", "agora", "também", "ainda", "só", "já", "ótimo",
        "show", "tipo", "vamos", "sempre", "isso", "assim", "aí", "bom",
    },
}

# ── Singleton CHRF (expensive to init) ────────────────────────────────────────
_CHRF = CHRF()

# ── Pyphen for Portuguese ──────────────────────────────────────────────────────
_pyphen_pt = pyphen.Pyphen(lang="pt_BR")


# ── Syllable counting ─────────────────────────────────────────────────────────

def _count_syllables_hindi(text: str) -> int:
    """
    Count syllables in Devanagari (Hindi) text.

    Each vowel matra or independent vowel = 1 syllable.
    Each Devanagari consonant without a following matra carries
    an inherent 'a' vowel and counts as half a syllable (we add
    consonant pairs as 1 to stay conservative and avoid over-counting).
    """
    count = 0
    chars = list(text)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in _HINDI_VOWEL_MATRAS or ch in _HINDI_INDEPENDENT_VOWELS:
            count += 1
        elif 'क' <= ch <= 'ह':  # consonant range ka–ha
            # Check if next char is a matra (then the matra handles it)
            next_ch = chars[i + 1] if i + 1 < len(chars) else ''
            if next_ch not in _HINDI_VOWEL_MATRAS and next_ch != '्':  # not virama
                count += 1  # inherent 'a' vowel
        i += 1
    return max(1, count)


def _count_syllables_portuguese(text: str) -> int:
    """Count syllables in Brazilian Portuguese using pyphen hyphenation."""
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

    Estimates spoken duration as syllable_count / speaking_rate, then
    penalises proportionally to the absolute deviation from target.

    Returns float in [0.0, 1.0].
    """
    rate = SPEAKING_RATES.get(language, 5.0)
    syllables = count_syllables(translation, language)
    estimated = syllables / rate
    score = max(0.0, 1.0 - abs(estimated - target_duration) / target_duration)
    return round(score, 4)


# ── REWARD 2: Semantic ────────────────────────────────────────────────────────

def semantic_reward(translation: str, references: list[str]) -> float:
    """
    Score semantic faithfulness using chrF against multiple references.

    sacrebleu's CHRF.sentence_score accepts a list of references and
    automatically picks the best match — critical for avoiding false
    penalties when the model produces a correct but differently-worded
    translation.

    Args:
        translation: model output
        references:  list of 2+ acceptable reference translations

    Returns float in [0.0, 1.0].
    """
    if not references:
        return 0.0
    score = _CHRF.sentence_score(translation, references).score / 100.0
    return round(max(0.0, min(1.0, score)), 4)


# ── REWARD 3: Budget ──────────────────────────────────────────────────────────

def budget_reward(budget_deficit: float) -> float:
    """
    Score how well the agent is managing the cumulative timing budget.

      < 0.2s  → perfect  (1.0)
      > 2.0s  → fail     (0.0)
      linear interpolation in between

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
    Score locale-appropriateness as fraction of words in locale wordlist.

    Returns float in [0.0, 1.0].
    """
    wordlist = LOCALE_WORDS.get(locale, set())
    if not wordlist:
        return 0.5
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
    penalty = 0.0

    if translation.strip() == original_english.strip():
        penalty -= 1.0

    if len(translation.split()) < 2:
        penalty -= 1.0

    if estimated_duration > target_duration * 1.8:
        penalty -= 0.5

    if language == "Hindi":
        has_devanagari = any('ऀ' <= ch <= 'ॿ' for ch in translation)
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

    Weights: timing 35% | semantic 35% | budget 20% | locale 10%
    Plus additive anti-hacking penalties.

    segment must contain:
        original_text, original_duration, target_language,
        locale, reference_translations (list of str)

    Returns float clamped to [-1.0, 1.0].
    """
    language        = segment["target_language"]
    target_duration = segment["original_duration"]
    references      = segment["reference_translations"]   # list of 2+
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
        0.35 * t_reward
        + 0.35 * s_reward
        + 0.20 * b_reward
        + 0.10 * l_reward
        + penalty
    )

    return round(max(-1.0, min(1.0, combined)), 4)


if __name__ == "__main__":
    seg = {
        "original_text":          "Add a pinch of salt to the boiling water.",
        "original_duration":      2.5,
        "target_language":        "Portuguese",
        "locale":                 "Brazil",
        "reference_translations": [
            "Adicione uma pitada de sal à água fervente.",
            "Coloque uma pitada de sal na água fervendo.",
        ],
    }
    translation = "Adicione uma pitada de sal à água fervente."
    print("timing :", timing_reward(translation, seg["original_duration"], seg["target_language"]))
    print("semantic:", semantic_reward(translation, seg["reference_translations"]))
    print("budget :", budget_reward(0.0))
    print("locale :", locale_reward(translation, seg["locale"]))
    print("combined:", compute_reward(translation, seg, 0.0))
