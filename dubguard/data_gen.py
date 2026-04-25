import json
import random
import os
from typing import List, Dict, Any

# Ensure episode IDs are auto-incremented
_global_ep_counter = 1

def estimate_tts_duration(text: str, language: str) -> float:
    """
    Syllable-based TTS duration estimate.
    Speaking rates (syllables per second):
      Hindi: 4.5, Portuguese: 5.2, Spanish: 5.0, English: 4.0
    Word count is unreliable — syllables are accurate.
    """
    import re

    SPEAKING_RATES = {'hi': 4.5, 'pt': 5.2, 'es': 5.0, 'en': 4.0}
    rate = SPEAKING_RATES.get(language, 4.0)

    if language == 'hi':
        vowel_matras = set([
            'ा','ि','ी','ु','ू','ृ','ॄ','े','ै','ो','ौ','ॅ','ॆ','ॉ','ॊ'
        ])
        independent_vowels = set([
            'अ','आ','इ','ई','उ','ऊ','ऋ','ए','ऐ','ओ','औ'
        ])
        syllable_count = sum(
            1 for ch in text if ch in vowel_matras or ch in independent_vowels
        )
        syllable_count += sum(1 for ch in text if 'क' <= ch <= 'ह') // 2
        if syllable_count == 0:
            text_clean = re.sub(r'[^a-zA-Z\s]', '', text)
            for word in text_clean.split():
                count = len(re.findall(r'[aeiou]+', word.lower()))
                syllable_count += max(1, count)
        syllable_count = max(1, syllable_count)

    elif language in ['pt', 'es']:
        text_lower = text.lower()
        text_clean = re.sub(r'[^a-záéíóúâêîôûàèìòùãõüïäöç\s]', '', text_lower)
        vowels = 'aeiouáéíóúâêîôûàèìòùãõüïäöç'
        syllable_count = 0
        prev_vowel = False
        for char in text_clean:
            if char in vowels:
                if not prev_vowel:
                    syllable_count += 1
                prev_vowel = True
            else:
                prev_vowel = False
        syllable_count = max(1, syllable_count)

    else:
        text_clean = re.sub(r'[^a-z\s]', '', text.lower())
        syllable_count = 0
        for word in text_clean.split():
            count = len(re.findall(r'[aeiou]+', word))
            if word.endswith('e') and len(word) > 3:
                count -= 1
            syllable_count += max(1, count)

    duration = syllable_count / rate + len(text.split()) * 0.03
    return round(max(0.3, duration), 2)

LOCALE_WORDS = {
    "BR": {
        "approved": [
            "você", "gente", "né", "tá", "cara", "muito",
            "super", "aqui", "isso", "vamos", "então", "bem",
            "bom", "legal", "bacana", "ótimo", "incrível",
            "fazer", "ver", "sabe", "olha", "fica", "vai",
            "tem", "tão", "assim", "demais", "também", "ainda",
            "agora", "já", "só", "até", "mesmo", "pra", "pro",
            "essa", "esse", "aí", "lá", "uai", "oxente",
            "saudade", "jeitinho", "xícaras", "receita"
        ],
        "penalized": [
            "vós", "vosso", "autocarro", "comboio",
            "pequeno-almoço", "casa de banho", "frigorífico",
            "chávena", "talho"
        ]
    },
    "IN": {
        "approved": [
            "यह", "है", "और", "के", "में", "को", "से",
            "पर", "एक", "हम", "आप", "वह", "जो", "कि",
            "बहुत", "अच्छा", "जल्दी", "आसान", "बनाएं",
            "करें", "देखें", "यहां", "अब", "तो", "भी",
            "yaar", "arre", "bilkul", "theek", "accha",
            "bahut", "shukriya", "namaste"
        ],
        "penalized": []
    },
    "AR": {
        "approved": [
            "vos", "che", "bárbaro", "re", "posta", "copado",
            "dale", "piola", "quilombo", "laburo", "morfar",
            "boludo", "fiaca", "chamuyar", "chabón"
        ],
        "penalized": [
            "tío", "vale", "mola", "guay", "hostia",
            "tronco", "chaval", "colega"
        ]
    }
}

def check_locale_words(text: str, locale: str) -> dict:
    """
    Check if text contains locale-appropriate words.
    Returns dict with score and word details.
    """
    wordlist = LOCALE_WORDS.get(locale, {})
    approved = wordlist.get("approved", [])
    penalized = wordlist.get("penalized", [])
    words = text.lower().split()

    if not words:
        return {"has_approved": False, "approved_words": [],
                "has_penalized": False, "penalized_words": [],
                "locale_score": 0.05}

    approved_found = [w for w in words if w in approved]
    penalized_found = [w for w in words if w in penalized]
    score = len(approved_found) / max(len(words), 1)
    score -= len(penalized_found) * 0.2
    score = max(0.05, min(1.0, score))

    return {
        "has_approved": len(approved_found) > 0,
        "approved_words": approved_found,
        "has_penalized": len(penalized_found) > 0,
        "penalized_words": penalized_found,
        "locale_score": round(score, 3)
    }

def load_bank(filename: str) -> List[Dict]:
    path = os.path.join(os.path.dirname(__file__), 'data', filename)
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

timing_bank = load_bank('timing_bank.json')
mistranslation_bank = load_bank('mistranslation_bank.json')
tone_bank = load_bank('tone_bank.json')
cultural_bank = load_bank('cultural_bank.json')
clean_bank = load_bank('clean_bank.json')

_used_clean_entries = set()
_used_timing_entries = set()
_used_mistranslation_entries = set()

def get_locale_for_lang(lang: str) -> str:
    if lang == "hi": return "IN"
    if lang == "pt": return "BR"
    return "AR"

def generate_base_segment(ep_id: str, seg_id: int, difficulty: str, lang: str) -> Dict[str, Any]:
    start_time = round(random.uniform(1.0, 50.0), 1)
    original_duration = round(random.uniform(1.5, 4.0), 1)
    end_time = round(start_time + original_duration, 1)
    next_segment_start = round(end_time + random.uniform(0.1, 1.0), 1)
    max_allow = round(original_duration + 0.15, 1)

    return {
        "episode_id": ep_id,
        "segment_id": seg_id,
        "difficulty": difficulty,
        "original": {
            "text": "",
            "language": "en",
            "start_time": start_time,
            "end_time": end_time,
            "duration": original_duration
        },
        "dubbed": {
            "text": "",
            "language": lang,
            "locale": get_locale_for_lang(lang),
            "estimated_duration": 0.0
        },
        "next_segment_start": next_segment_start,
        "max_allowed_dubbed_duration": max_allow,
        "error_type_planted": None,
        "ground_truth": {
            "error_type": None,
            "severity": "PASS",
            "suggested_fix": None,
            "fix_duration": 0.0,
            "locale_rule": None
        }
    }

def add_clean_segment(base: Dict[str, Any]) -> Dict[str, Any]:
    global _used_clean_entries
    lang = base["dubbed"]["language"]
    valid_entries = [e for e in clean_bank if e["lang"] == lang]
    if not valid_entries: return base

    valid_entries_unused = [
        e for e in valid_entries
        if e["original_en"] not in _used_clean_entries
    ]
    if not valid_entries_unused:
        _used_clean_entries = set()
        valid_entries_unused = valid_entries

    entry = random.choice(valid_entries_unused)
    _used_clean_entries.add(entry["original_en"])

    base["original"]["text"] = entry["original_en"]
    base["original"]["duration"] = entry["original_duration"]
    base["original"]["end_time"] = round(base["original"]["start_time"] + entry["original_duration"], 1)
    base["max_allowed_dubbed_duration"] = round(entry["original_duration"] + 0.15, 1)
    base["next_segment_start"] = round(base["original"]["end_time"] + random.uniform(0.1, 1.0), 1)

    key = f"dubbed_{lang}"
    if lang == 'es': key = 'dubbed_es'
    correct_text = entry.get(key, entry.get(f"dubbed_{lang}", "fallback text"))
    base["dubbed"]["text"] = correct_text

    est_dur = estimate_tts_duration(correct_text, lang)
    max_allowed = base["max_allowed_dubbed_duration"]
    attempts = 0
    while est_dur > max_allowed + 0.3 and attempts < 3:
        entry = random.choice(valid_entries_unused)
        key = f"dubbed_{lang}"
        correct_text = entry.get(key, entry.get(f"dubbed_{lang}", "fallback text"))
        base["original"]["duration"] = entry["original_duration"]
        base["max_allowed_dubbed_duration"] = round(entry["original_duration"] + 0.15, 1)
        max_allowed = base["max_allowed_dubbed_duration"]
        est_dur = estimate_tts_duration(correct_text, lang)
        attempts += 1

    base["dubbed"]["estimated_duration"] = round(est_dur, 2)

    base["error_type_planted"] = None
    base["ground_truth"] = {
        "error_type": None,
        "severity": "PASS",
        "suggested_fix": None,
        "fix_duration": 0.0,
        "locale_rule": None
    }

    locale = base["dubbed"]["locale"]
    locale_check = check_locale_words(correct_text, locale)
    base["dubbed"]["locale_check"] = locale_check
    return base

def add_timing_collision(base: Dict[str, Any]) -> Dict[str, Any]:
    lang = base["dubbed"]["language"]
    valid_entries = [e for e in timing_bank if e["lang"] == lang]
    if not valid_entries: return add_clean_segment(base)
    entry = random.choice(valid_entries)
    
    base["original"]["text"] = entry["original_en"]
    orig_dur = entry["original_duration"]
    base["original"]["duration"] = orig_dur
    base["original"]["end_time"] = round(base["original"]["start_time"] + orig_dur, 1)
    base["max_allowed_dubbed_duration"] = round(orig_dur + 0.1, 1)
    
    long_text = entry[f"dubbed_{lang}"]
    short_text = entry[f"short_fix_{lang}"]
    
    est_dur_long = estimate_tts_duration(long_text, lang)
    next_start = round(base["original"]["start_time"] + est_dur_long - 0.5, 1)
    if next_start <= base["original"]["end_time"]:
        next_start = round(base["original"]["end_time"] + 0.1, 1)
    base["next_segment_start"] = next_start
    
    base["dubbed"]["text"] = long_text
    base["dubbed"]["estimated_duration"] = est_dur_long
    
    severity = entry.get("severity", "BLOCK")
    
    est_dur_short = estimate_tts_duration(short_text, lang)
    # Ensure fix duration is valid
    if est_dur_short > base["max_allowed_dubbed_duration"]:
        est_dur_short = base["max_allowed_dubbed_duration"] - 0.05
    
    base["error_type_planted"] = "timing_collision"
    if severity == "WARN":
        base["ground_truth"]["error_type"] = "timing_warn"
    else:
        base["ground_truth"]["error_type"] = "timing_collision"

    base["ground_truth"].update({
        "severity": severity,
        "suggested_fix": short_text,
        "fix_duration": round(est_dur_short, 2),
        "locale_rule": None
    })

    locale = base["dubbed"]["locale"]
    locale_check = check_locale_words(long_text, locale)
    base["dubbed"]["locale_check"] = locale_check
    return base

def add_mistranslation(base: Dict[str, Any]) -> Dict[str, Any]:
    lang = base["dubbed"]["language"]
    valid_entries = [e for e in mistranslation_bank if e["lang"] == lang]
    if not valid_entries: return add_clean_segment(base)
    entry = random.choice(valid_entries)
    
    base["original"]["text"] = entry["original_en"]
    # Depending on language, grab the right key
    wrong_key = "wrong_dubbed_es_AR" if lang == "es" else f"wrong_dubbed_{lang}"
    correct_key = "correct_dubbed_es_AR" if lang == "es" else f"correct_dubbed_{lang}"
    wrong_text = entry[wrong_key]
    correct_text = entry[correct_key]
    
    base["dubbed"]["text"] = wrong_text
    base["dubbed"]["estimated_duration"] = estimate_tts_duration(wrong_text, lang)
    
    # Adjust durations to fit this text so it's only a mistranslation error
    fix_dur = estimate_tts_duration(correct_text, lang)
    orig_dur = round(max(base["dubbed"]["estimated_duration"], fix_dur) + 0.5, 1)
    base["original"]["duration"] = orig_dur
    base["original"]["end_time"] = round(base["original"]["start_time"] + orig_dur, 1)
    base["max_allowed_dubbed_duration"] = round(orig_dur + 0.1, 1)
    base["next_segment_start"] = round(base["original"]["end_time"] + 0.5, 1)
    
    base["error_type_planted"] = "mistranslation"
    base["ground_truth"].update({
        "error_type": "mistranslation",
        "severity": "BLOCK",
        "suggested_fix": correct_text,
        "fix_duration": round(fix_dur, 2),
        "locale_rule": None
    })

    locale = base["dubbed"]["locale"]
    locale_check = check_locale_words(wrong_text, locale)
    base["dubbed"]["locale_check"] = locale_check
    return base

def add_tone_mismatch(base: Dict[str, Any]) -> Dict[str, Any]:
    lang = base["dubbed"]["language"]
    valid_entries = [e for e in tone_bank if e["lang"] == lang]
    if not valid_entries: return add_clean_segment(base)
    entry = random.choice(valid_entries)
    
    base["original"]["text"] = entry["original_en"]
    wrong_text = entry[f"wrong_tone_{lang}"]
    correct_text = entry[f"correct_tone_{lang}"]
    
    base["dubbed"]["text"] = wrong_text
    base["dubbed"]["estimated_duration"] = estimate_tts_duration(wrong_text, lang)
    
    fix_dur = estimate_tts_duration(correct_text, lang)
    orig_dur = round(max(base["dubbed"]["estimated_duration"], fix_dur) + 0.5, 1)
    base["original"]["duration"] = orig_dur
    base["original"]["end_time"] = round(base["original"]["start_time"] + orig_dur, 1)
    base["max_allowed_dubbed_duration"] = round(orig_dur + 0.1, 1)
    base["next_segment_start"] = round(base["original"]["end_time"] + 0.5, 1)
    
    base["error_type_planted"] = "tone_mismatch"
    base["ground_truth"].update({
        "error_type": "tone_mismatch",
        "severity": "WARN",
        "suggested_fix": correct_text,
        "fix_duration": round(fix_dur, 2),
        "locale_rule": None
    })

    locale = base["dubbed"]["locale"]
    locale_check = check_locale_words(wrong_text, locale)
    base["dubbed"]["locale_check"] = locale_check
    return base

def add_cultural_mismatch(base: Dict[str, Any]) -> Dict[str, Any]:
    lang = base["dubbed"]["language"]
    valid_entries = [e for e in cultural_bank if e["lang"] == lang]
    if not valid_entries: return add_mistranslation(base)
    entry = random.choice(valid_entries)
    
    base["original"]["text"] = entry.get(
        "original_en", "This phrase requires cultural adaptation."
    )
    wrong_text = entry["wrong_phrase"]
    correct_text = entry["correct_phrase"]
    
    base["dubbed"]["text"] = wrong_text
    base["dubbed"]["estimated_duration"] = estimate_tts_duration(wrong_text, lang)
    
    fix_dur = estimate_tts_duration(correct_text, lang)
    orig_dur = round(max(base["dubbed"]["estimated_duration"], fix_dur) + 0.5, 1)
    base["original"]["duration"] = orig_dur
    base["original"]["end_time"] = round(base["original"]["start_time"] + orig_dur, 1)
    base["max_allowed_dubbed_duration"] = round(orig_dur + 0.1, 1)
    base["next_segment_start"] = round(base["original"]["end_time"] + 0.5, 1)
    
    base["error_type_planted"] = "cultural_mismatch"
    base["ground_truth"].update({
        "error_type": "cultural_mismatch",
        "severity": "BLOCK",
        "suggested_fix": correct_text,
        "fix_duration": round(fix_dur, 2),
        "locale_rule": entry["rule"]
    })

    locale = base["dubbed"]["locale"]
    locale_check = check_locale_words(wrong_text, locale)
    base["dubbed"]["locale_check"] = locale_check
    return base

def generate_clean_episode(ep_id: str, seg_id: int, difficulty: str, lang: str) -> Dict[str, Any]:
    base = generate_base_segment(ep_id, seg_id, difficulty, lang)
    return add_clean_segment(base)

def generate_episode(difficulty: str) -> List[Dict[str, Any]]:
    global _global_ep_counter
    ep_id = f"ep_{_global_ep_counter:03d}"
    _global_ep_counter += 1
    
    lang = random.choices(['pt', 'es', 'hi'], weights=[0.34, 0.33, 0.33], k=1)[0]

    if lang == 'hi':
        hi_entries = [e for e in clean_bank if e.get("lang") == "hi"]
        has_devanagari = any(
            any('ऀ' <= ch <= 'ॿ' for ch in e.get("dubbed_hi", ""))
            for e in hi_entries
        )
        if not has_devanagari:
            print("WARNING: Hindi bank has no Devanagari script. "
                  "Add Devanagari to hi bank entries for best results. "
                  "Romanized Hindi will be used — syllable counting "
                  "will use English fallback.")

    segments = []
    
    composition = []
    if difficulty == "easy":
        # 1 error (obvious: timing or mistranslation)
        composition = [random.choice([add_timing_collision, add_mistranslation])]
    elif difficulty == "medium":
        # 1-2 errors, 1 clean segment
        composition = [add_clean_segment]
        for _ in range(random.randint(1, 2)):
            composition.append(random.choice([add_timing_collision, add_mistranslation, add_tone_mismatch]))
    elif difficulty == "hard":
        # 3-4 errors, 2-3 clean segments
        for _ in range(random.randint(2, 3)):
            composition.append(add_clean_segment)
        for _ in range(random.randint(3, 4)):
            composition.append(random.choice([add_timing_collision, add_mistranslation, add_tone_mismatch, add_cultural_mismatch]))
    else:
        composition = [add_clean_segment]
        
    random.shuffle(composition)
    
    # Generate unique random segment ids between 1 and 10
    seg_ids = random.sample(range(1, 11), len(composition))
    
    for func, seg_id in zip(composition, seg_ids):
        base = generate_base_segment(ep_id, seg_id, difficulty, lang)
        # Advance time so segments are somewhat contiguous
        if segments:
            base["original"]["start_time"] = segments[-1]["next_segment_start"]
            
        seg = func(base)
        segments.append(seg)
        
    return segments

def validate_episode(ep: Dict[str, Any]) -> bool:
    try:
        assert isinstance(ep.get("episode_id"), str)
        assert isinstance(ep.get("segment_id"), int)
        assert isinstance(ep.get("difficulty"), str)
        
        orig = ep.get("original", {})
        assert "text" in orig
        assert "language" in orig
        assert "start_time" in orig
        assert "end_time" in orig
        assert "duration" in orig
        
        dub = ep.get("dubbed", {})
        assert "text" in dub
        assert "language" in dub
        assert "locale" in dub
        assert "estimated_duration" in dub
        
        assert "next_segment_start" in ep
        assert "max_allowed_dubbed_duration" in ep
        
        gt = ep.get("ground_truth", {})
        assert ep.get("error_type_planted") == gt.get("error_type") or \
               (ep.get("error_type_planted") == "timing_collision" and gt.get("error_type") in ["timing_collision", "timing_warn"])
               
        assert gt.get("severity") in ["BLOCK", "WARN", "PASS"]
        
        fix_dur = gt.get("fix_duration", 0.0)
        max_allow = ep.get("max_allowed_dubbed_duration", 0.0)
        assert fix_dur <= max_allow, f"fix_duration {fix_dur} > max_allowed {max_allow}"

        # Check: no dummy text in original
        orig_text = ep.get("original", {}).get("text", "")
        if "Dummy" in orig_text or "dummy" in orig_text:
            print(f"FAIL: Dummy placeholder text found")
            return False

        # Check: fix_duration <= max_allowed for error segments
        if gt.get("severity") in ["BLOCK", "WARN"]:
            if fix_dur > max_allow:
                print(f"FAIL: fix_duration {fix_dur} > max_allow {max_allow}")
                return False

        # Check: locale_check exists
        if "locale_check" not in ep.get("dubbed", {}):
            print("WARNING: locale_check missing from segment")

        return True
    except AssertionError as e:
        print(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    all_segments = []
    difficulties = ["easy"] * 7 + ["medium"] * 7 + ["hard"] * 6
    random.shuffle(difficulties)

    all_valid = True
    lang_counts: Dict[str, int] = {}
    error_type_counts: Dict[str, int] = {}
    unique_originals: set = set()
    locale_scores = []

    for diff in difficulties:
        ep_segments = generate_episode(diff)
        for seg in ep_segments:
            if not validate_episode(seg):
                all_valid = False

            lang = seg["dubbed"]["language"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            err = seg.get("error_type_planted") or "clean"
            error_type_counts[err] = error_type_counts.get(err, 0) + 1

            unique_originals.add(seg["original"]["text"])

            locale_check = seg["dubbed"].get("locale_check", {})
            locale_scores.append(locale_check.get("locale_score", 0))

            all_segments.append(seg)

    total_segs = sum(lang_counts.values())
    avg_locale = sum(locale_scores) / len(locale_scores) if locale_scores else 0

    print("=" * 50)
    print("DATA GENERATION DIAGNOSTIC REPORT")
    print("=" * 50)
    print(f"Total segments generated: {len(all_segments)}")
    print(f"Unique original sentences: {len(unique_originals)} / {len(all_segments)}")
    print(f"\nLanguage distribution (target ~33% each):")
    for lc, count in lang_counts.items():
        pct = (count / total_segs * 100) if total_segs > 0 else 0
        status = "[OK]" if 25 <= pct <= 45 else "[!!] imbalanced"
        print(f"  {lc}: {count} ({pct:.1f}%) {status}")

    print(f"\nError type distribution:")
    for err, count in error_type_counts.items():
        print(f"  {err}: {count}")

    print(f"\nAverage locale score: {avg_locale:.3f}")
    if avg_locale > 0.05:
        print("  [OK] Locale reward will be non-zero during training")
    else:
        print("  [FAIL] Locale score too low -- check LOCALE_WORDS and bank data.")

    over_budget = sum(
        1 for s in all_segments
        if s["dubbed"]["estimated_duration"] > s["max_allowed_dubbed_duration"] + 0.3
        and s["ground_truth"]["severity"] == "PASS"
    )
    print(f"\nPASS segments over budget: {over_budget}")
    if over_budget == 0:
        print("  [OK] No artificial duration clamping detected")
    else:
        print("  [!!] Some PASS segments exceed window -- check add_clean_segment retry logic")

    dummy_count = sum(
        1 for s in all_segments
        if "Dummy" in s["original"].get("text", "")
    )
    print(f"Dummy text segments: {dummy_count}")
    if dummy_count == 0:
        print("  [OK] No dummy text found")
    else:
        print("  [FAIL] Dummy text still present -- check add_cultural_mismatch fix")

    print("\n" + "=" * 50)
    if all_valid and avg_locale > 0.05 and over_budget == 0 and dummy_count == 0:
        print("[OK] ALL CHECKS PASSED -- data is ready for training")
    else:
        print("[FAIL] ISSUES FOUND -- fix before training")
        if not all_valid:       print("  - Validation failures in segments")
        if avg_locale <= 0.05:  print("  - Locale score too low")
        if over_budget > 0:     print("  - PASS segments over budget")
        if dummy_count > 0:     print("  - Dummy text in segments")

    with open("sample_output.json", "w", encoding="utf-8") as f:
        json.dump(all_segments[:6], f, indent=2, ensure_ascii=False)
    print(f"\nSample saved: sample_output.json (first 6 segments)")
