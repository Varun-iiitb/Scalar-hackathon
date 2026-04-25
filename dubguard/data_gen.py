import json
import random
import os
from typing import List, Dict, Any

# Ensure episode IDs are auto-incremented
_global_ep_counter = 1

def estimate_tts_duration(text: str, language: str) -> float:
    """
    Rough TTS duration estimate based on word count and language WPM.
    Hindi: ~110 WPM, Portuguese: ~130 WPM, Spanish: ~130 WPM, English: ~150 WPM
    """
    word_count = len(text.split())
    if language == 'hi':
        wpm = 110.0
    elif language in ['es', 'pt']:
        wpm = 130.0
    else: # Default to English
        wpm = 150.0
        
    minutes = word_count / wpm
    seconds = minutes * 60.0
    return round(seconds, 2)

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

def get_locale_for_lang(lang: str) -> str:
    if lang == "hi": return "IN"
    if lang == "pt": return "BR"
    return "AR"

def generate_base_segment(ep_id: str, seg_id: int, difficulty: str, lang: str) -> Dict[str, Any]:
    start_time = round(random.uniform(1.0, 50.0), 1)
    original_duration = round(random.uniform(1.0, 5.0), 1) 
    end_time = round(start_time + original_duration, 1)
    next_segment_start = round(end_time + random.uniform(0.1, 1.0), 1)
    max_allow = round(original_duration + 0.1, 1)

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
    lang = base["dubbed"]["language"]
    valid_entries = [e for e in clean_bank if e["lang"] == lang]
    if not valid_entries: return base
    entry = random.choice(valid_entries)
    
    base["original"]["text"] = entry["original_en"]
    base["original"]["duration"] = entry["original_duration"]
    base["original"]["end_time"] = round(base["original"]["start_time"] + entry["original_duration"], 1)
    base["max_allowed_dubbed_duration"] = round(entry["original_duration"] + 0.1, 1)
    base["next_segment_start"] = round(base["original"]["end_time"] + random.uniform(0.1, 1.0), 1)
    
    key = f"dubbed_{lang}"
    if lang == 'es': key = 'dubbed_es' 
    correct_text = entry.get(key, entry.get(f"dubbed_{lang}", "fallback text"))
    base["dubbed"]["text"] = correct_text
    
    # Force the estimated duration to fit to represent a clean valid segment
    est_dur = estimate_tts_duration(correct_text, lang)
    if est_dur > base["max_allowed_dubbed_duration"]:
        est_dur = base["max_allowed_dubbed_duration"] - 0.1
    base["dubbed"]["estimated_duration"] = round(est_dur, 2)
    
    base["error_type_planted"] = None
    base["ground_truth"] = {
        "error_type": None,
        "severity": "PASS",
        "suggested_fix": None,
        "fix_duration": round(est_dur, 2),
        "locale_rule": None
    }
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
    return base

def add_cultural_mismatch(base: Dict[str, Any]) -> Dict[str, Any]:
    lang = base["dubbed"]["language"]
    valid_entries = [e for e in cultural_bank if e["lang"] == lang]
    if not valid_entries: return add_mistranslation(base)
    entry = random.choice(valid_entries)
    
    base["original"]["text"] = "Dummy translation text (cultural context mismatch)." 
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
    return base

def generate_clean_episode(ep_id: str, seg_id: int, difficulty: str, lang: str) -> Dict[str, Any]:
    base = generate_base_segment(ep_id, seg_id, difficulty, lang)
    return add_clean_segment(base)

def generate_episode(difficulty: str) -> List[Dict[str, Any]]:
    global _global_ep_counter
    ep_id = f"ep_{_global_ep_counter:03d}"
    _global_ep_counter += 1
    
    lang = random.choice(['hi', 'es', 'pt'])
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
        
        return True
    except AssertionError as e:
        print(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    easy_ep = generate_episode("easy")
    medium_ep = generate_episode("medium")
    hard_ep = generate_episode("hard")
    
    # Run validator
    all_valid = True
    for ep_list in [easy_ep, medium_ep, hard_ep]:
        for seg in ep_list:
            if not validate_episode(seg):
                all_valid = False
                
    if not all_valid:
        print("Warning: Some generated segments failed validation.")
        
    print(json.dumps(easy_ep, indent=2))
    print(json.dumps(medium_ep, indent=2))
    print(json.dumps(hard_ep, indent=2))
