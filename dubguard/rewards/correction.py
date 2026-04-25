import re
import warnings

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None

_model = None

# Syllable-based speaking rates (syllables per second) — more accurate than WPM
SPEAKING_RATES = {
    "hi": 4.5,
    "pt": 5.2,
    "es": 5.0,
    "en": 4.0,
}

def get_correction_reward(agent_action, ground_truth, language="en"):
    """
    Rewards quality of the suggested fix using two checks:
    (a) Semantic similarity to ground truth via paraphrase-multilingual-MiniLM-L12-v2.
    (b) Timing fit using WPM estimates, heavily penalising if max_allowed_duration is exceeded.
    """
    global _model

    gt_fix = ground_truth.get("suggested_fix")
    agent_fix = agent_action.get("suggested_fix")

    if not gt_fix:
        return 1.0 if not agent_fix else 0.0

    if not agent_fix:
        return 0.0

    if _model is None:
        if SentenceTransformer is None:
            warnings.warn(
                "sentence_transformers or scikit-learn not installed. "
                "Semantic similarity defaults to 0.0."
            )
        else:
            _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    sim_score = 0.0
    if _model is not None and cosine_similarity is not None:
        gt_emb = _model.encode([gt_fix])
        agent_emb = _model.encode([agent_fix])
        sim_score = cosine_similarity(gt_emb, agent_emb)[0][0]

    word_count = len(re.findall(r"\w+", str(agent_fix)))
    rate = SPEAKING_RATES.get(str(language)[:2].lower(), 4.0)
    # Approximate syllables as 1.5× word count for non-Devanagari text
    estimated_duration = (word_count * 1.5) / rate + word_count * 0.03
    max_duration = ground_truth.get("fix_duration", 0.0)

    timing_penalty = 0.0
    if max_duration > 0 and estimated_duration > max_duration:
        timing_penalty = -0.5

    return max(0.0, float(sim_score) + timing_penalty)
