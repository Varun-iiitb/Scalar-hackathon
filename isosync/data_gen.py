"""
IsoSync data generator.

Produces synthetic dubbing episodes from a hardcoded bank of 50 English
sentences (cooking, travel, lifestyle) with TWO reference translations per
language — used for multi-reference chrF scoring in rewards.py.

Curriculum language assignment:
  Level 1 → Portuguese (5 segments, 0.5s slack)
  Level 2 → Portuguese (8 segments, 0.2s slack)
  Level 3 → Hindi      (10 segments, 0.1s slack, locale constraints active)

Small models like Qwen2.5-1.5B handle Latin-script Portuguese well out of the
box, giving non-zero reward signal from episode 1. Hindi is introduced at
level 3 only, after the model has already learned to compress translations.

Public API:
    generate_episode(level: int) -> list[dict]
"""

import random

# ── Speaking rates (syllables per second) ─────────────────────────────────────
SPEAKING_RATES = {
    "Hindi":      4.5,   # Mumbai Hindi
    "Portuguese": 5.2,   # Brazilian Portuguese
}

LOCALES = {
    "Hindi":      "Mumbai",
    "Portuguese": "Brazil",
}

# ── Curriculum: level → config (language is fixed per level) ──────────────────
# Level 3 is now harder Portuguese (not Hindi) to prevent catastrophic forgetting.
# Tighter slack (0.05s) forces the agent to learn maximum compression while
# retaining Portuguese semantic quality — reward curves stay monotone.
CURRICULUM = {
    1: {"n_segments": 5,  "duration_slack": 0.5,  "locale_constraints": False, "language": "Portuguese"},
    2: {"n_segments": 8,  "duration_slack": 0.2,  "locale_constraints": False, "language": "Portuguese"},
    3: {"n_segments": 10, "duration_slack": 0.05, "locale_constraints": True,  "language": "Portuguese"},
}

# ── Sentence bank ─────────────────────────────────────────────────────────────
# Each entry: (english, duration_seconds, hindi_ref1, hindi_ref2, pt_ref1, pt_ref2)
SENTENCE_BANK = [
    # ── COOKING (15) ──────────────────────────────────────────────────────────
    (
        "Add a pinch of salt to the boiling water.", 2.5,
        "उबलते पानी में एक चुटकी नमक डालें।",
        "उबलते पानी में थोड़ा नमक मिलाएं।",
        "Adicione uma pitada de sal à água fervente.",
        "Coloque uma pitada de sal na água fervendo.",
    ),
    (
        "Stir the mixture gently for two minutes.", 2.2,
        "मिश्रण को दो मिनट तक धीरे से हिलाएं।",
        "दो मिनट तक मिश्रण को हल्के हाथ से चलाएं।",
        "Mexa a mistura delicadamente por dois minutos.",
        "Misture suavemente por dois minutos.",
    ),
    (
        "Chop the onions into small pieces.", 2.0,
        "प्याज को छोटे-छोटे टुकड़ों में काटें।",
        "प्याज को बारीक काट लें।",
        "Pique as cebolas em pedaços pequenos.",
        "Corte as cebolas em cubinhos.",
    ),
    (
        "Heat the oil in a pan over medium flame.", 2.3,
        "मध्यम आंच पर पैन में तेल गरम करें।",
        "पैन में तेल डालकर मध्यम आंच पर गर्म करें।",
        "Aqueça o óleo em uma panela em fogo médio.",
        "Coloque o óleo na panela e aqueça em fogo médio.",
    ),
    (
        "Add the spices and cook for one minute.", 2.1,
        "मसाले डालें और एक मिनट तक पकाएं।",
        "अब मसाले मिलाएं और एक मिनट तक भूनें।",
        "Adicione as especiarias e cozinhe por um minuto.",
        "Coloque os temperos e refogue por um minuto.",
    ),
    (
        "Pour the batter slowly into the mold.", 2.0,
        "घोल को धीरे-धीरे सांचे में डालें।",
        "मिश्रण को आराम से सांचे में उंडेलें।",
        "Despeje a massa lentamente na forma.",
        "Coloque a massa aos poucos na fôrma.",
    ),
    (
        "Let the dough rest for thirty minutes.", 2.2,
        "आटे को तीस मिनट के लिए रखा छोड़ दें।",
        "आटे को ढककर तीस मिनट के लिए रख दें।",
        "Deixe a massa descansar por trinta minutos.",
        "Deixe a massa repousar por meia hora.",
    ),
    (
        "Garnish with fresh coriander leaves.", 1.8,
        "ताजे धनिए की पत्तियों से सजाएं।",
        "ऊपर से ताजा धनिया डालें।",
        "Decore com folhas frescas de coentro.",
        "Finalize com coentro fresco picado.",
    ),
    (
        "Squeeze half a lemon over the dish.", 2.0,
        "व्यंजन पर आधा नींबू निचोड़ें।",
        "ऊपर से आधे नींबू का रस निचोड़ें।",
        "Esprema meio limão sobre o prato.",
        "Adicione o suco de meio limão por cima.",
    ),
    (
        "The aroma of spices fills the kitchen.", 2.1,
        "मसालों की खुशबू रसोई में फैल जाती है।",
        "मसालों की सुगंध पूरी रसोई में बिखर जाती है।",
        "O aroma das especiarias enche a cozinha.",
        "O perfume das especiarias toma conta da cozinha.",
    ),
    (
        "Simmer the curry on low heat for twenty minutes.", 2.8,
        "करी को धीमी आंच पर बीस मिनट तक पकाएं।",
        "बीस मिनट तक करी को धीमी आंच पर उबलने दें।",
        "Cozinhe o curry em fogo baixo por vinte minutos.",
        "Deixe o curry apurar em fogo baixo por vinte minutos.",
    ),
    (
        "Roll the dough flat with a rolling pin.", 2.0,
        "बेलन से आटे को चपटा बेलें।",
        "बेलन की सहायता से आटे को पतला बेल लें।",
        "Abra a massa com um rolo de cozinha.",
        "Use o rolo para esticar a massa.",
    ),
    (
        "Fry until the edges turn golden brown.", 2.1,
        "किनारे सुनहरे भूरे होने तक तलें।",
        "जब तक किनारे सुनहरे न हो जाएं तब तक तलें।",
        "Frite até as bordas ficarem douradas.",
        "Frite até dourar bem nas bordas.",
    ),
    (
        "Mix all ingredients in a large bowl.", 2.0,
        "एक बड़े कटोरे में सभी सामग्री मिलाएं।",
        "सभी चीजें एक बड़ी कटोरी में डालकर अच्छे से मिलाएं।",
        "Misture todos os ingredientes em uma tigela grande.",
        "Coloque todos os ingredientes numa tigela grande e misture.",
    ),
    (
        "This recipe is ready in just ten minutes.", 2.3,
        "यह रेसिपी केवल दस मिनट में तैयार हो जाती है।",
        "यह डिश सिर्फ दस मिनट में बनकर तैयार है।",
        "Esta receita fica pronta em apenas dez minutos.",
        "Essa receita está pronta em só dez minutos.",
    ),
    # ── TRAVEL (15) ───────────────────────────────────────────────────────────
    (
        "The sunset view from this hill is breathtaking.", 2.5,
        "इस पहाड़ी से सूर्यास्त का नज़ारा अद्भुत है।",
        "यहाँ की पहाड़ी से सूरज डूबते देखना अविश्वसनीय है।",
        "A vista do pôr do sol desta colina é de tirar o fôlego.",
        "O pôr do sol visto daqui é simplesmente incrível.",
    ),
    (
        "Pack light when traveling in summer.", 2.0,
        "गर्मियों में यात्रा करते समय हल्का सामान लें।",
        "गर्मी में कम और हल्का सामान लेकर चलें।",
        "Leve menos bagagem ao viajar no verão.",
        "Viaje com pouca mala no verão.",
    ),
    (
        "The local market opens early in the morning.", 2.5,
        "स्थानीय बाज़ार सुबह जल्दी खुल जाता है।",
        "यहाँ का बाज़ार सुबह-सुबह ही खुल जाता है।",
        "O mercado local abre cedo pela manhã.",
        "O mercado daqui abre bem cedinho.",
    ),
    (
        "Try the street food for an authentic experience.", 2.4,
        "प्रामाणिक अनुभव के लिए स्ट्रीट फूड ज़रूर चखें।",
        "असली स्वाद के लिए यहाँ का स्ट्रीट फूड ज़रूर खाएं।",
        "Experimente a comida de rua para uma experiência autêntica.",
        "Prove a comida de rua para vivenciar o lugar de verdade.",
    ),
    (
        "Book your tickets at least a week in advance.", 2.6,
        "कम से कम एक हफ्ते पहले अपने टिकट बुक करें।",
        "टिकट कम से कम एक हफ्ता पहले बुक कर लें।",
        "Reserve seus ingressos com pelo menos uma semana de antecedência.",
        "Compre suas passagens com uma semana de antecedência.",
    ),
    (
        "The beaches here are clean and uncrowded.", 2.2,
        "यहाँ के समुद्र तट साफ और कम भीड़ वाले हैं।",
        "यहाँ के बीच बहुत साफ हैं और ज़्यादा भीड़ नहीं होती।",
        "As praias aqui são limpas e pouco movimentadas.",
        "As praias dessa região são limpas e tranquilas.",
    ),
    (
        "Visit the old city to see the architecture.", 2.4,
        "वास्तुकला देखने के लिए पुराने शहर की यात्रा करें।",
        "पुरानी वास्तुकला देखनी हो तो पुराने शहर में जरूर जाएं।",
        "Visite a cidade antiga para ver a arquitetura.",
        "Explore a parte antiga da cidade para admirar a arquitetura.",
    ),
    (
        "Carry a reusable water bottle while hiking.", 2.3,
        "ट्रेकिंग के दौरान पुन: उपयोगी पानी की बोतल रखें।",
        "ट्रेक पर जाते समय रिफिल होने वाली बोतल साथ लें।",
        "Leve uma garrafa reutilizável ao fazer trilha.",
        "Use uma garrafa retornável nas trilhas.",
    ),
    (
        "The train journey through the mountains is beautiful.", 2.6,
        "पहाड़ों से गुजरती ट्रेन यात्रा बहुत सुंदर होती है।",
        "पहाड़ों के बीच से ट्रेन में जाना एक अद्भुत अनुभव है।",
        "A viagem de trem pelas montanhas é linda.",
        "Viajar de trem pelas montanhas é uma experiência linda.",
    ),
    (
        "Always carry your passport when traveling abroad.", 2.5,
        "विदेश यात्रा के समय हमेशा पासपोर्ट साथ रखें।",
        "विदेश जाते समय पासपोर्ट हमेशा अपने पास रखें।",
        "Sempre leve seu passaporte ao viajar para o exterior.",
        "Nunca viaje ao exterior sem o seu passaporte.",
    ),
    (
        "The best time to visit is during the winter.", 2.5,
        "घूमने का सबसे अच्छा समय सर्दियों में है।",
        "यहाँ आने का सबसे अच्छा मौसम सर्दियों का होता है।",
        "A melhor época para visitar é durante o inverno.",
        "O inverno é a melhor estação para visitar.",
    ),
    (
        "Explore the city by walking through its lanes.", 2.5,
        "गलियों में घूम कर शहर को एक्सप्लोर करें।",
        "शहर की गलियों में पैदल घूमकर उसे खोजें।",
        "Explore a cidade caminhando por suas ruelas.",
        "Descubra a cidade a pé pelas suas vielas.",
    ),
    (
        "The waterfall is just two kilometers from here.", 2.4,
        "झरना यहाँ से केवल दो किलोमीटर दूर है।",
        "यहाँ से मात्र दो किलोमीटर दूर एक झरना है।",
        "A cachoeira fica a apenas dois quilômetros daqui.",
        "A cachoeira está a só dois quilômetros daqui.",
    ),
    (
        "Check the weather before planning outdoor activities.", 2.6,
        "बाहरी गतिविधियाँ करने से पहले मौसम जांचें।",
        "बाहर जाने से पहले मौसम का हाल जरूर देख लें।",
        "Verifique o clima antes de planejar atividades ao ar livre.",
        "Sempre cheque o tempo antes de planejar atividades externas.",
    ),
    (
        "The hotel offers a stunning view of the sea.", 2.4,
        "होटल से समुद्र का शानदार नज़ारा दिखता है।",
        "इस होटल से समुद्र का खूबसूरत नज़ारा दिखाई देता है।",
        "O hotel oferece uma vista deslumbrante do mar.",
        "Desse hotel a vista do mar é de encher os olhos.",
    ),
    # ── LIFESTYLE (20) ────────────────────────────────────────────────────────
    (
        "Start your morning with a glass of warm water.", 2.5,
        "अपनी सुबह एक गिलास गुनगुने पानी से शुरू करें।",
        "सुबह उठते ही एक गिलास गर्म पानी पिएं।",
        "Comece sua manhã com um copo de água morna.",
        "Inicie o dia tomando um copo de água morna.",
    ),
    (
        "Exercise for at least thirty minutes every day.", 2.3,
        "हर दिन कम से कम तीस मिनट व्यायाम करें।",
        "रोज़ कम से कम आधे घंटे की एक्सरसाइज़ करें।",
        "Faça exercício por pelo menos trinta minutos todos os dias.",
        "Exercite-se por pelo menos meia hora diariamente.",
    ),
    (
        "A good night's sleep improves your mood.", 2.1,
        "अच्छी रात की नींद आपके मूड को बेहतर बनाती है।",
        "रात को अच्छी नींद लेने से मूड अच्छा रहता है।",
        "Uma boa noite de sono melhora seu humor.",
        "Dormir bem à noite deixa o seu humor melhor.",
    ),
    (
        "Cut down on sugar for better energy levels.", 2.2,
        "बेहतर ऊर्जा के लिए चीनी कम करें।",
        "ऊर्जा बनाए रखने के लिए मीठा कम खाएं।",
        "Reduza o açúcar para ter mais energia.",
        "Consuma menos açúcar para manter o nível de energia.",
    ),
    (
        "Read for twenty minutes before going to bed.", 2.3,
        "सोने से पहले बीस मिनट पढ़ें।",
        "रात को सोने से पहले बीस मिनट किताब पढ़ें।",
        "Leia por vinte minutos antes de dormir.",
        "Leia vinte minutos antes de ir para a cama.",
    ),
    (
        "Stay hydrated by drinking eight glasses of water.", 2.5,
        "आठ गिलास पानी पीकर हाइड्रेटेड रहें।",
        "दिन में आठ गिलास पानी पीकर खुद को हाइड्रेट रखें।",
        "Mantenha-se hidratado bebendo oito copos de água.",
        "Beba oito copos de água por dia para se manter hidratado.",
    ),
    (
        "Meditation helps reduce stress and anxiety.", 2.0,
        "ध्यान तनाव और चिंता को कम करने में मदद करता है।",
        "ध्यान करने से तनाव और घबराहट कम होती है।",
        "A meditação ajuda a reduzir o estresse e a ansiedade.",
        "Meditar é ótimo para aliviar o estresse e a ansiedade.",
    ),
    (
        "Take short breaks when working for long hours.", 2.3,
        "लंबे समय तक काम करते समय छोटे ब्रेक लें।",
        "घंटों काम करें तो बीच-बीच में थोड़ा ब्रेक लें।",
        "Faça pausas curtas ao trabalhar por muitas horas.",
        "Descanse um pouco quando trabalhar por muitas horas seguidas.",
    ),
    (
        "Grow your own herbs on a small balcony.", 2.2,
        "एक छोटी बालकनी पर अपनी जड़ी-बूटियाँ उगाएं।",
        "छोटी सी बालकनी में भी अपनी जड़ी-बूटियाँ उगाई जा सकती हैं।",
        "Cultive suas próprias ervas em uma pequena varanda.",
        "Você pode cultivar ervas na sua varanda, mesmo sendo pequena.",
    ),
    (
        "Declutter your space for a clearer mind.", 2.0,
        "स्पष्ट मन के लिए अपने स्थान को साफ करें।",
        "जगह साफ रखने से मन भी शांत रहता है।",
        "Organize seu espaço para ter uma mente mais clara.",
        "Um espaço organizado ajuda a clarear a mente.",
    ),
    (
        "Practice gratitude by writing three things daily.", 2.4,
        "रोज़ तीन चीजें लिखकर कृतज्ञता का अभ्यास करें।",
        "हर रोज़ तीन अच्छी बातें लिखकर कृतज्ञ रहना सीखें।",
        "Pratique a gratidão escrevendo três coisas diariamente.",
        "Escreva três coisas boas por dia para praticar a gratidão.",
    ),
    (
        "Limit screen time before sleeping for better rest.", 2.6,
        "बेहतर आराम के लिए सोने से पहले स्क्रीन टाइम कम करें।",
        "अच्छी नींद के लिए रात को सोने से पहले मोबाइल कम इस्तेमाल करें।",
        "Limite o uso de telas antes de dormir para descansar melhor.",
        "Reduza o tempo de tela antes de dormir para descansar melhor.",
    ),
    (
        "Walk ten thousand steps every day for fitness.", 2.5,
        "फिटनेस के लिए हर दिन दस हज़ार कदम चलें।",
        "फिट रहने के लिए रोज़ दस हज़ार कदम चलना जरूरी है।",
        "Caminhe dez mil passos todos os dias para manter a forma.",
        "Dar dez mil passos por dia é ótimo para a saúde.",
    ),
    (
        "Cook at home more often to eat healthier.", 2.3,
        "स्वस्थ खाने के लिए अधिक बार घर पर खाना बनाएं।",
        "घर का खाना खाने से सेहत बेहतर रहती है।",
        "Cozinhe em casa com mais frequência para comer de forma saudável.",
        "Comer em casa com mais frequência faz bem à saúde.",
    ),
    (
        "Spend time in nature to recharge your energy.", 2.4,
        "अपनी ऊर्जा को रिचार्ज करने के लिए प्रकृति में समय बिताएं।",
        "प्रकृति के बीच समय बिताने से ऊर्जा वापस आती है।",
        "Passe tempo na natureza para recarregar suas energias.",
        "Estar em contato com a natureza renova as energias.",
    ),
    (
        "Learn a new skill during your free time.", 2.2,
        "अपने खाली समय में कोई नई कौशल सीखें।",
        "खाली वक्त में कुछ नया सीखने की कोशिश करें।",
        "Aprenda uma nova habilidade no seu tempo livre.",
        "Use o tempo livre para aprender algo novo.",
    ),
    (
        "Surround yourself with people who inspire you.", 2.4,
        "ऐसे लोगों से घिरे रहें जो आपको प्रेरित करें।",
        "ऐसे लोगों के साथ रहें जो आपको आगे बढ़ने की प्रेरणा दें।",
        "Cerque-se de pessoas que te inspiram.",
        "Fique perto de pessoas que te motivam a crescer.",
    ),
    (
        "Your mental health is just as important as physical.", 2.8,
        "आपका मानसिक स्वास्थ्य उतना ही महत्वपूर्ण है जितना शारीरिक।",
        "मानसिक स्वास्थ्य भी शारीरिक स्वास्थ्य जितना ही जरूरी है।",
        "Sua saúde mental é tão importante quanto a física.",
        "A saúde mental merece tanta atenção quanto a saúde física.",
    ),
    (
        "Plan your week on Sunday evening.", 1.8,
        "रविवार शाम को अपने हफ्ते की योजना बनाएं।",
        "हर रविवार शाम को अगले हफ्ते की प्लानिंग करें।",
        "Planeje sua semana na tarde de domingo.",
        "Use o domingo à tarde para planejar a semana.",
    ),
    (
        "Small daily habits lead to big life changes.", 2.3,
        "छोटी दैनिक आदतें बड़े जीवन परिवर्तन लाती हैं।",
        "रोज़ की छोटी-छोटी आदतें ज़िंदगी को बदल देती हैं।",
        "Pequenos hábitos diários levam a grandes mudanças na vida.",
        "Hábitos pequenos praticados todo dia transformam a vida.",
    ),
]


def generate_episode(
    level: int = 1,
    seed: int | None = None,
) -> list[dict]:
    """
    Generate one dubbing episode at the given curriculum level.

    Language is determined by curriculum level:
      Level 1 & 2 → Portuguese (model handles Latin script well out of the box)
      Level 3      → Hindi (introduced after model has learned to compress)

    Returns a list of segment dicts, each containing:
        segment_id, original_text, original_duration, target_language,
        locale, reference_translations (list of 2), max_syllables,
        duration_slack, locale_constraints
    """
    if level not in CURRICULUM:
        raise ValueError(f"level must be 1, 2, or 3 — got {level}")

    cfg = CURRICULUM[level]
    language = cfg["language"]
    rng = random.Random(seed)
    pool = rng.sample(SENTENCE_BANK, min(cfg["n_segments"], len(SENTENCE_BANK)))

    # Indices into the SENTENCE_BANK tuple:
    # (en, dur, hi_ref1, hi_ref2, pt_ref1, pt_ref2)
    ref1_idx = 2 if language == "Hindi" else 4
    ref2_idx = 3 if language == "Hindi" else 5

    segments = []
    for i, entry in enumerate(pool):
        english, base_duration = entry[0], entry[1]
        ref1, ref2 = entry[ref1_idx], entry[ref2_idx]

        # Small jitter for variety (±0.15 s)
        duration = round(base_duration + rng.uniform(-0.15, 0.15), 2)
        duration = max(1.5, min(4.0, duration))

        rate = SPEAKING_RATES[language]
        max_syllables = int(duration * rate)

        segments.append({
            "segment_id":            i,
            "original_text":         english,
            "original_duration":     duration,
            "target_language":       language,
            "locale":                LOCALES[language],
            "reference_translations": [ref1, ref2],   # list — used by chrF
            "max_syllables":         max_syllables,
            "duration_slack":        cfg["duration_slack"],
            "locale_constraints":    cfg["locale_constraints"],
        })

    return segments


if __name__ == "__main__":
    for lvl in (1, 2, 3):
        ep = generate_episode(level=lvl, seed=0)
        lang = CURRICULUM[lvl]["language"]
        print(f"Level {lvl} | {lang} | {len(ep)} segments")
        seg = ep[0]
        print(f"  [{seg['segment_id']}] {seg['original_text']}")
        print(f"       ref1 → {seg['reference_translations'][0]}")
        print(f"       ref2 → {seg['reference_translations'][1]}")
        print(f"       duration={seg['original_duration']}s  max_syl={seg['max_syllables']}")
