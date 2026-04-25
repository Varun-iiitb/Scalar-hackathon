"""
IsoSync data generator.

Produces synthetic dubbing episodes from a hardcoded bank of 50 English
sentences (cooking, travel, lifestyle — typical Reels content) with
reference translations in Hindi and Portuguese.

Public API:
    generate_episode(level: int, language: str) -> list[dict]
"""

import random
from typing import Literal

# ── Speaking rates (syllables per second) ─────────────────────────────────────
SPEAKING_RATES = {
    "Hindi":      4.5,   # Mumbai Hindi
    "Portuguese": 5.2,   # Brazilian Portuguese
}

LOCALES = {
    "Hindi":      "Mumbai",
    "Portuguese": "Brazil",
}

# ── Curriculum config ─────────────────────────────────────────────────────────
CURRICULUM = {
    1: {"n_segments": 5,  "duration_slack": 0.5, "locale_constraints": False},
    2: {"n_segments": 8,  "duration_slack": 0.2, "locale_constraints": False},
    3: {"n_segments": 10, "duration_slack": 0.1, "locale_constraints": True},
}

# ── Sentence bank ─────────────────────────────────────────────────────────────
# 50 entries: (english, duration_seconds, hindi_ref, portuguese_ref)
SENTENCE_BANK = [
    # ── COOKING (15) ──────────────────────────────────────────────────────────
    (
        "Add a pinch of salt to the boiling water.",
        2.5,
        "उबलते पानी में एक चुटकी नमक डालें।",
        "Adicione uma pitada de sal à água fervente.",
    ),
    (
        "Stir the mixture gently for two minutes.",
        2.2,
        "मिश्रण को दो मिनट तक धीरे से हिलाएं।",
        "Mexa a mistura delicadamente por dois minutos.",
    ),
    (
        "Chop the onions into small pieces.",
        2.0,
        "प्याज को छोटे-छोटे टुकड़ों में काटें।",
        "Pique as cebolas em pedaços pequenos.",
    ),
    (
        "Heat the oil in a pan over medium flame.",
        2.3,
        "मध्यम आंच पर पैन में तेल गरम करें।",
        "Aqueça o óleo em uma panela em fogo médio.",
    ),
    (
        "Add the spices and cook for one minute.",
        2.1,
        "मसाले डालें और एक मिनट तक पकाएं।",
        "Adicione as especiarias e cozinhe por um minuto.",
    ),
    (
        "Pour the batter slowly into the mold.",
        2.0,
        "घोल को धीरे-धीरे सांचे में डालें।",
        "Despeje a massa lentamente na forma.",
    ),
    (
        "Let the dough rest for thirty minutes.",
        2.2,
        "आटे को तीस मिनट के लिए रखा छोड़ दें।",
        "Deixe a massa descansar por trinta minutos.",
    ),
    (
        "Garnish with fresh coriander leaves.",
        1.8,
        "ताजे धनिए की पत्तियों से सजाएं।",
        "Decore com folhas frescas de coentro.",
    ),
    (
        "Squeeze half a lemon over the dish.",
        2.0,
        "व्यंजन पर आधा नींबू निचोड़ें।",
        "Esprema meio limão sobre o prato.",
    ),
    (
        "The aroma of spices fills the kitchen.",
        2.1,
        "मसालों की खुशबू रसोई में फैल जाती है।",
        "O aroma das especiarias enche a cozinha.",
    ),
    (
        "Simmer the curry on low heat for twenty minutes.",
        2.8,
        "करी को धीमी आंच पर बीस मिनट तक पकाएं।",
        "Cozinhe o curry em fogo baixo por vinte minutos.",
    ),
    (
        "Roll the dough flat with a rolling pin.",
        2.0,
        "बेलन से आटे को चपटा बेलें।",
        "Abra a massa com um rolo de cozinha.",
    ),
    (
        "Fry until the edges turn golden brown.",
        2.1,
        "किनारे सुनहरे भूरे होने तक तलें।",
        "Frite até as bordas ficarem douradas.",
    ),
    (
        "Mix all ingredients in a large bowl.",
        2.0,
        "एक बड़े कटोरे में सभी सामग्री मिलाएं।",
        "Misture todos os ingredientes em uma tigela grande.",
    ),
    (
        "This recipe is ready in just ten minutes.",
        2.3,
        "यह रेसिपी केवल दस मिनट में तैयार हो जाती है।",
        "Esta receita fica pronta em apenas dez minutos.",
    ),
    # ── TRAVEL (15) ───────────────────────────────────────────────────────────
    (
        "The sunset view from this hill is breathtaking.",
        2.5,
        "इस पहाड़ी से सूर्यास्त का नज़ारा अद्भुत है।",
        "A vista do pôr do sol desta colina é de tirar o fôlego.",
    ),
    (
        "Pack light when traveling in summer.",
        2.0,
        "गर्मियों में यात्रा करते समय हल्का सामान लें।",
        "Leve menos bagagem ao viajar no verão.",
    ),
    (
        "The local market opens early in the morning.",
        2.5,
        "स्थानीय बाज़ार सुबह जल्दी खुल जाता है।",
        "O mercado local abre cedo pela manhã.",
    ),
    (
        "Try the street food for an authentic experience.",
        2.4,
        "प्रामाणिक अनुभव के लिए स्ट्रीट फूड ज़रूर चखें।",
        "Experimente a comida de rua para uma experiência autêntica.",
    ),
    (
        "Book your tickets at least a week in advance.",
        2.6,
        "कम से कम एक हफ्ते पहले अपने टिकट बुक करें।",
        "Reserve seus ingressos com pelo menos uma semana de antecedência.",
    ),
    (
        "The beaches here are clean and uncrowded.",
        2.2,
        "यहाँ के समुद्र तट साफ और कम भीड़ वाले हैं।",
        "As praias aqui são limpas e pouco movimentadas.",
    ),
    (
        "Visit the old city to see the architecture.",
        2.4,
        "वास्तुकला देखने के लिए पुराने शहर की यात्रा करें।",
        "Visite a cidade antiga para ver a arquitetura.",
    ),
    (
        "Carry a reusable water bottle while hiking.",
        2.3,
        "ट्रेकिंग के दौरान पुन: उपयोगी पानी की बोतल रखें।",
        "Leve uma garrafa reutilizável ao fazer trilha.",
    ),
    (
        "The train journey through the mountains is beautiful.",
        2.6,
        "पहाड़ों से गुजरती ट्रेन यात्रा बहुत सुंदर होती है।",
        "A viagem de trem pelas montanhas é linda.",
    ),
    (
        "Always carry your passport when traveling abroad.",
        2.5,
        "विदेश यात्रा के समय हमेशा पासपोर्ट साथ रखें।",
        "Sempre leve seu passaporte ao viajar para o exterior.",
    ),
    (
        "The best time to visit is during the winter.",
        2.5,
        "घूमने का सबसे अच्छा समय सर्दियों में है।",
        "A melhor época para visitar é durante o inverno.",
    ),
    (
        "Explore the city by walking through its lanes.",
        2.5,
        "गलियों में घूम कर शहर को एक्सप्लोर करें।",
        "Explore a cidade caminhando por suas ruelas.",
    ),
    (
        "The waterfall is just two kilometers from here.",
        2.4,
        "झरना यहाँ से केवल दो किलोमीटर दूर है।",
        "A cachoeira fica a apenas dois quilômetros daqui.",
    ),
    (
        "Check the weather before planning outdoor activities.",
        2.6,
        "बाहरी गतिविधियाँ करने से पहले मौसम जांचें।",
        "Verifique o clima antes de planejar atividades ao ar livre.",
    ),
    (
        "The hotel offers a stunning view of the sea.",
        2.4,
        "होटल से समुद्र का शानदार नज़ारा दिखता है।",
        "O hotel oferece uma vista deslumbrante do mar.",
    ),
    # ── LIFESTYLE (20) ────────────────────────────────────────────────────────
    (
        "Start your morning with a glass of warm water.",
        2.5,
        "अपनी सुबह एक गिलास गुनगुने पानी से शुरू करें।",
        "Comece sua manhã com um copo de água morna.",
    ),
    (
        "Exercise for at least thirty minutes every day.",
        2.3,
        "हर दिन कम से कम तीस मिनट व्यायाम करें।",
        "Faça exercício por pelo menos trinta minutos todos os dias.",
    ),
    (
        "A good night's sleep improves your mood.",
        2.1,
        "अच्छी रात की नींद आपके मूड को बेहतर बनाती है।",
        "Uma boa noite de sono melhora seu humor.",
    ),
    (
        "Cut down on sugar for better energy levels.",
        2.2,
        "बेहतर ऊर्जा के लिए चीनी कम करें।",
        "Reduza o açúcar para ter mais energia.",
    ),
    (
        "Read for twenty minutes before going to bed.",
        2.3,
        "सोने से पहले बीस मिनट पढ़ें।",
        "Leia por vinte minutos antes de dormir.",
    ),
    (
        "Stay hydrated by drinking eight glasses of water.",
        2.5,
        "आठ गिलास पानी पीकर हाइड्रेटेड रहें।",
        "Mantenha-se hidratado bebendo oito copos de água.",
    ),
    (
        "Meditation helps reduce stress and anxiety.",
        2.0,
        "ध्यान तनाव और चिंता को कम करने में मदद करता है।",
        "A meditação ajuda a reduzir o estresse e a ansiedade.",
    ),
    (
        "Take short breaks when working for long hours.",
        2.3,
        "लंबे समय तक काम करते समय छोटे ब्रेक लें।",
        "Faça pausas curtas ao trabalhar por muitas horas.",
    ),
    (
        "Grow your own herbs on a small balcony.",
        2.2,
        "एक छोटी बालकनी पर अपनी जड़ी-बूटियाँ उगाएं।",
        "Cultive suas próprias ervas em uma pequena varanda.",
    ),
    (
        "Declutter your space for a clearer mind.",
        2.0,
        "स्पष्ट मन के लिए अपने स्थान को साफ करें।",
        "Organize seu espaço para ter uma mente mais clara.",
    ),
    (
        "Practice gratitude by writing three things daily.",
        2.4,
        "रोज़ तीन चीजें लिखकर कृतज्ञता का अभ्यास करें।",
        "Pratique a gratidão escrevendo três coisas diariamente.",
    ),
    (
        "Limit screen time before sleeping for better rest.",
        2.6,
        "बेहतर आराम के लिए सोने से पहले स्क्रीन टाइम कम करें।",
        "Limite o uso de telas antes de dormir para descansar melhor.",
    ),
    (
        "Walk ten thousand steps every day for fitness.",
        2.5,
        "फिटनेस के लिए हर दिन दस हज़ार कदम चलें।",
        "Caminhe dez mil passos todos os dias para manter a forma.",
    ),
    (
        "Cook at home more often to eat healthier.",
        2.3,
        "स्वस्थ खाने के लिए अधिक बार घर पर खाना बनाएं।",
        "Cozinhe em casa com mais frequência para comer de forma saudável.",
    ),
    (
        "Spend time in nature to recharge your energy.",
        2.4,
        "अपनी ऊर्जा को रिचार्ज करने के लिए प्रकृति में समय बिताएं।",
        "Passe tempo na natureza para recarregar suas energias.",
    ),
    (
        "Learn a new skill during your free time.",
        2.2,
        "अपने खाली समय में कोई नई कौशल सीखें।",
        "Aprenda uma nova habilidade no seu tempo livre.",
    ),
    (
        "Surround yourself with people who inspire you.",
        2.4,
        "ऐसे लोगों से घिरे रहें जो आपको प्रेरित करें।",
        "Cerque-se de pessoas que te inspiram.",
    ),
    (
        "Your mental health is just as important as physical.",
        2.8,
        "आपका मानसिक स्वास्थ्य उतना ही महत्वपूर्ण है जितना शारीरिक।",
        "Sua saúde mental é tão importante quanto a física.",
    ),
    (
        "Plan your week on Sunday evening.",
        1.8,
        "रविवार शाम को अपने हफ्ते की योजना बनाएं।",
        "Planeje sua semana na tarde de domingo.",
    ),
    (
        "Small daily habits lead to big life changes.",
        2.3,
        "छोटी दैनिक आदतें बड़े जीवन परिवर्तन लाती हैं।",
        "Pequenos hábitos diários levam a grandes mudanças na vida.",
    ),
]


def generate_episode(
    level: int = 1,
    language: Literal["Hindi", "Portuguese"] = "Hindi",
    seed: int | None = None,
) -> list[dict]:
    """
    Generate one dubbing episode at the given curriculum level and language.

    Returns a list of segment dicts, each containing:
        segment_id, original_text, original_duration, target_language,
        locale, reference_translation, max_syllables
    """
    if level not in CURRICULUM:
        raise ValueError(f"level must be 1, 2, or 3 — got {level}")
    if language not in SPEAKING_RATES:
        raise ValueError(f"language must be 'Hindi' or 'Portuguese' — got {language}")

    cfg = CURRICULUM[level]
    rng = random.Random(seed)
    pool = rng.sample(SENTENCE_BANK, min(cfg["n_segments"], len(SENTENCE_BANK)))

    ref_idx = 2 if language == "Hindi" else 3  # index into SENTENCE_BANK tuple

    segments = []
    for i, entry in enumerate(pool):
        english, base_duration, hindi_ref, pt_ref = entry
        reference = hindi_ref if language == "Hindi" else pt_ref

        # Add a small random jitter to duration (±0.15 s) for variety
        duration = round(base_duration + rng.uniform(-0.15, 0.15), 2)
        duration = max(1.5, min(4.0, duration))

        rate = SPEAKING_RATES[language]
        max_syllables = int(duration * rate)

        segments.append({
            "segment_id":           i,
            "original_text":        english,
            "original_duration":    duration,
            "target_language":      language,
            "locale":               LOCALES[language],
            "reference_translation": reference,
            "max_syllables":        max_syllables,
            "duration_slack":       cfg["duration_slack"],
            "locale_constraints":   cfg["locale_constraints"],
        })

    return segments


if __name__ == "__main__":
    # Quick smoke test
    for lang in ("Hindi", "Portuguese"):
        for lvl in (1, 2, 3):
            ep = generate_episode(level=lvl, language=lang, seed=0)
            print(f"Level {lvl} | {lang} | {len(ep)} segments")
            seg = ep[0]
            print(f"  [{seg['segment_id']}] {seg['original_text']}")
            print(f"       → {seg['reference_translation']}")
            print(f"       duration={seg['original_duration']}s  max_syl={seg['max_syllables']}")
