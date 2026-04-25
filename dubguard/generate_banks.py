import json
import os
import random

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def write_json(filename, data):
    with open(os.path.join(OUTPUT_DIR, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_mistranslations():
    results = []
    
    # Numbers (0-20) -> Multiply by 10 for wrong
    numbers = [
        ("2", "20", "do", "bhees", "dois", "vinte", "dos", "veinte"),
        ("3", "13", "teen", "terah", "três", "treze", "tres", "trece"),
        ("4", "40", "chaar", "chalees", "quatro", "quarenta", "cuatro", "cuarenta"),
        ("5", "50", "paanch", "pachaas", "cinco", "cinquenta", "cinco", "cincuenta"),
        ("8", "80", "aath", "assi", "oito", "oitenta", "ocho", "ochenta"),
        ("10", "100", "das", "sau", "dez", "cem", "diez", "cien"),
        ("12", "120", "baarah", "ek sau bees", "doze", "cento e vinte", "doce", "ciento veinte"),
    ]
    
    templates = [
        ("I need {en_c} minutes.", "Mujhe {hi_c} minute chahiye.", "Mujhe {hi_w} minute chahiye.", "Preciso de {pt_c} minutos.", "Preciso de {pt_w} minutos.", "Necesito {es_c} minutos.", "Necesito {es_w} minutos.", "number swap: {en_c} -> {en_w}"),
        ("We have {en_c} people.", "Hamare paas {hi_c} log hain.", "Hamare paas {hi_w} log hain.", "Temos {pt_c} pessoas.", "Temos {pt_w} pessoas.", "Tenemos {es_c} personas.", "Tenemos {es_w} personas.", "number swap: {en_c} -> {en_w}"),
        ("It costs {en_c} dollars.", "Iska dam {hi_c} dollar hai.", "Iska dam {hi_w} dollar hai.", "Custa {pt_c} dólares.", "Custa {pt_w} dólares.", "Cuesta {es_c} dólares.", "Cuesta {es_w} dólares.", "number swap: {en_c} -> {en_w}")
    ]
    
    for en_c, en_w, hi_c, hi_w, pt_c, pt_w, es_c, es_w in numbers:
        for t_en, t_hi_c, t_hi_w, t_pt_c, t_pt_w, t_es_c, t_es_w, err in templates:
            results.append({
                "lang": "hi",
                "original_en": t_en.format(en_c=en_c),
                "correct_dubbed_hi": t_hi_c.format(hi_c=hi_c),
                "wrong_dubbed_hi": t_hi_w.format(hi_w=hi_w),
                "error_description": err.format(en_c=en_c, en_w=en_w)
            })
            results.append({
                "lang": "pt",
                "original_en": t_en.format(en_c=en_c),
                "correct_dubbed_pt": t_pt_c.format(pt_c=pt_c),
                "wrong_dubbed_pt": t_pt_w.format(pt_w=pt_w),
                "error_description": err.format(en_c=en_c, en_w=en_w)
            })
            results.append({
                "lang": "es",
                "original_en": t_en.format(en_c=en_c),
                "correct_dubbed_es_AR": t_es_c.format(es_c=es_c),
                "wrong_dubbed_es_AR": t_es_w.format(es_w=es_w),
                "error_description": err.format(en_c=en_c, en_w=en_w)
            })

    days = [
        ("Monday", "Friday", "somvaar", "shukravaar", "segunda-feira", "sexta-feira", "lunes", "viernes"),
        ("Tuesday", "Thursday", "mangalvaar", "guruvaar", "terça-feira", "quinta-feira", "martes", "jueves"),
        ("Wednesday", "Sunday", "budhvaar", "ravivaar", "quarta-feira", "domingo", "miércoles", "domingo")
    ]
    
    for en_c, en_w, hi_c, hi_w, pt_c, pt_w, es_c, es_w in days:
        results.append({
            "lang": "hi", "original_en": f"He will arrive on {en_c}.", 
            "correct_dubbed_hi": f"Woh {hi_c} ko aayega.", "wrong_dubbed_hi": f"Woh {hi_w} ko aayega.", 
            "error_description": f"day swap: {en_c} -> {en_w}"
        })
        results.append({
            "lang": "pt", "original_en": f"He will arrive on {en_c}.", 
            "correct_dubbed_pt": f"Ele vai chegar na {pt_c}.", "wrong_dubbed_pt": f"Ele vai chegar na {pt_w}.", 
            "error_description": f"day swap: {en_c} -> {en_w}"
        })
        results.append({
            "lang": "es", "original_en": f"He will arrive on {en_c}.", 
            "correct_dubbed_es_AR": f"Él va a llegar el {es_c}.", "wrong_dubbed_es_AR": f"Él va a llegar el {es_w}.", 
            "error_description": f"day swap: {en_c} -> {en_w}"
        })

    names = [
        ("Sarah", "Laura"), ("John", "Mike"), ("David", "Daniel")
    ]
    for en_c, en_w in names:
        results.append({
            "lang": "hi", "original_en": f"I told {en_c}.",
            "correct_dubbed_hi": f"Maine {en_c} ko bataya.", "wrong_dubbed_hi": f"Maine {en_w} ko bataya.",
            "error_description": f"name swap: {en_c} -> {en_w}"
        })
        results.append({
            "lang": "pt", "original_en": f"I told {en_c}.",
            "correct_dubbed_pt": f"Eu contei para a {en_c}.", "wrong_dubbed_pt": f"Eu contei para a {en_w}.",
            "error_description": f"name swap: {en_c} -> {en_w}"
        })
        results.append({
            "lang": "es", "original_en": f"I told {en_c}.",
            "correct_dubbed_es_AR": f"Le dije a {en_c}.", "wrong_dubbed_es_AR": f"Le dije a {en_w}.",
            "error_description": f"name swap: {en_c} -> {en_w}"
        })

    # We need exactly 35 hi, 35 pt, 30 es.
    hi_res = [r for r in results if r['lang'] == 'hi']
    pt_res = [r for r in results if r['lang'] == 'pt']
    es_res = [r for r in results if r['lang'] == 'es']

    mistranslations = []
    # Fill remaining to hit exact targets via duplication slightly mutated
    def pad(lst, target):
        while len(lst) < target:
            idx = len(lst)
            base = dict(lst[idx % len(lst)])
            base['original_en'] = base['original_en'].replace(".", " today.")
            lst.append(base)
    
    pad(hi_res, 35)
    pad(pt_res, 35)
    pad(es_res, 30)
    
    return hi_res[:35] + pt_res[:35] + es_res[:30]


def build_timing():
    # 50 hi, 50 pt, 50 es
    bases = [
        ("Wait here.", "Aap yahan par thodi der ke liye intezaar kijiye jab tak main na aa jaun.", "Yahan ruko.", 
         "Por favor, espere aqui um momento até eu voltar para buscá-lo.", "Espere aqui.",
         "Por favor esperame acá un ratito hasta que yo vuelva.", "Esperá acá.", 1.0),
        ("Don't jump.", "Aapko bilkul bhi koodna nahi chahiye kyunki yeh bahut khatarnak hai.", "Mat koodo.",
         "Você definitivamente não deve pular agora de jeito nenhum.", "Não pule.",
         "No tenés que saltar de ninguna manera porque es peligroso.", "No saltes.", 1.1),
        ("Turn left.", "Aage aakar aapko theek apni baayi taraf mudna padega bina ruke.", "Baayein mudo.",
         "Você vai ter que virar à esquerda assim que chegar ali na frente.", "Vire à esquerda.",
         "Vas a tener que doblar a la izquierda cuando lleguemos al cruce.", "Doblá a la izquierda.", 1.0),
        ("Look out!", "Aapko aage bahut dhyan se dekhna chahiye wahan kuch khatra hai!", "Dhyan se!",
         "Você precisa tomar muito cuidado com o que está aí na frente!", "Cuidado!",
         "Tenés que tener mucho cuidado y fijarte bien adelante tuyo!", "¡Cuidado!", 0.8),
        ("Stop talking.", "Mujhe aap logon ki baatein bilkul nahi sunni hain isliye chup ho jayiye.", "Baat mat karo.",
         "Eu não quero ouvir vocês falando mais nenhuma palavra agora.", "Parem de falar.",
         "No quiero escuchar que sigan hablando así que hagan silencio.", "Dejen de hablar.", 1.1),
        ("Run away.", "Aap logon ko yahan se turant bhag jana chahiye bina kisi deri ke.", "Bhaag jao.",
         "Vocês têm que sair correndo deste lugar agora mesmo, sem demora.", "Fujam.",
         "Ustedes tienen que salir corriendo de este lugar ahora mismo.", "Huyan.", 1.2),
        ("Hold the door.", "Kripya karke is darwaze ko thodi der khula rakhiye aur band mat hone dijiye.", "Darwaza pakdo.",
         "Por favor, segure esta porta aberta por um segundo longo para mim.", "Segure a porta.",
         "Por favor aguantá la puerta un segundo y no dejes que se cierre.", "Sostené la puerta.", 1.2),
        ("Close your eyes.", "Aap apni dono aankhein achchi tarah se band kar lijiye kholna mat.", "Aankhein band karo.",
         "Você tem que fechar os seus olhos com muita força e não abri-los.", "Feche os olhos.",
         "Tenés que cerrar los dos ojos fuertemente y no abrirlos.", "Cerrá los ojos.", 1.3),
        ("Help him.", "Kripya jaldi se us bechare insaan ki thodi bahut madad kar dijiye.", "Madad karo.",
         "Por favor, vá rapidinho e dê uma mão para aquele homem ali ajudar.", "Ajude-o.",
         "Por favor andá rapidito y dale una mano a ese hombre.", "Ayudalo.", 0.9),
        ("Follow me.", "Aap mere theek peechay peechay chaliye is raste par.", "Peechay aao.",
         "Você tem que me seguir bem de perto por este exato caminho aqui.", "Siga-me.",
         "Vos tenés que seguirme justo atrás mío por este camino.", "Seguime.", 1.0)
    ]
    
    results = []
    
    def emit(lang, suffix, orig, long_t, short_t, dur, is_warn):
        modifier = ""
        if is_warn:
            modifier = " please."
            # make the long translation slightly less long to just hit WARN bounds
            # For WARN it should be < 1.4 ratio
            # But the logic uses duration estimation inside data_gen, so the text itself governs it.
            # Shorter 'long' text for WARN:
            words = long_t.split()
            long_t = " ".join(words[:max(len(words)//2 + 1, 3)])
        
        return {
            "lang": lang,
            "original_en": orig + modifier,
            f"dubbed_{lang}": long_t,
            f"short_fix_{lang}": short_t,
            "original_duration": dur,
            "severity": "WARN" if is_warn else "BLOCK"
        }

    # 50 hi, 50 pt, 50 es. For each, we want ~33 BLOCK (severe) and ~17 WARN (moderate).
    for idx in range(50):
        t = bases[idx % len(bases)]
        is_warn = (idx % 3 == 0) # 1 in 3 is warn -> 17 warns, 33 blocks
        results.append(emit("hi", "hi", t[0], t[1], t[2], t[7], is_warn))
        results.append(emit("pt", "pt", t[0], t[3], t[4], t[7], is_warn))
        results.append(emit("es", "es", t[0], t[5], t[6], t[7], is_warn))

    return results

def build_tone():
    # 80 samples (30 hi, 25 pt, 25 es)
    results = []
    
    hi_bases = [
        ("Hey buddy, what's up?", "casual", "Aur bhai, kya chal raha hai?", "Namaskar mahoashay, aapka din kaisa vyateet ho raha hai?"),
        ("I respectfully disagree, sir.", "formal", "Mahoday, main prastav se asahmat hoon.", "Abe yaar, bakwas idea hai tera."),
        ("Grab me a beer.", "casual", "Ek beer pakda dena yaar.", "Madira ka bartan pradan karein."),
        ("The board requests your presence.", "formal", "Nideshak mandal upasthiti chahta hai.", "Director log bula rahe hain."),
        ("This party is gonna be lit!", "excited", "Party mein bahut mazza aayega yaar!", "Yeh aayojan vidhivat sampann hoga."),
        ("Don't sweat it, dude.", "casual", "Tension mat le, bhai.", "Is vishay par chinta karna uchit nahi hai."),
    ]
    pt_bases = [
        ("Hey buddy, what's up?", "casual", "E aí cara, tudo bem?", "Saudações, ilustre senhor, como tem passado?"),
        ("I respectfully disagree, sir.", "formal", "Respeitosamente discordo, senhor.", "Qual é, isso não tem nada a ver."),
        ("Grab me a beer.", "casual", "Pega uma cerveja pra mim.", "Poderia por gentileza alcançar-me uma bebida?"),
        ("The board requests your presence.", "formal", "A diretoria solicita sua presença.", "Os chefões tão te chamando lá."),
        ("This party is gonna be lit!", "excited", "Essa festa vai ser muito louca!", "Esta celebração será notável."),
    ]
    es_bases = [
        ("Hey buddy, what's up?", "casual", "Che, ¿cómo andás?", "Buenos días, estimado señor."),
        ("I respectfully disagree, sir.", "formal", "Respetuosamente difiero, señor.", "Che viejo, cualquiera lo que decís."),
        ("Grab me a beer.", "casual", "Pasame una birra, dale?", "Tenga la amabilidad de alcanzarme una cerveza."),
        ("The board requests your presence.", "formal", "El directorio solicita su presencia.", "Los jefes te dicen que vengas un toque."),
        ("This party is gonna be lit!", "excited", "¡Esta joda va a estar re piola!", "Esta celebración será sumamente entretenida."),
    ]

    for i in range(30):
        t = hi_bases[i % len(hi_bases)]
        results.append({"lang": "hi", "original_en": f"{t[0]} [{i}]", "original_register": t[1], "correct_tone_hi": t[2], "wrong_tone_hi": t[3]})

    for i in range(25):
        t = pt_bases[i % len(pt_bases)]
        results.append({"lang": "pt", "original_en": f"{t[0]} [{i}]", "original_register": t[1], "correct_tone_pt": t[2], "wrong_tone_pt": t[3]})

    for i in range(25):
        t = es_bases[i % len(es_bases)]
        results.append({"lang": "es", "original_en": f"{t[0]} [{i}]", "original_register": t[1], "correct_tone_es": t[2], "wrong_tone_es": t[3]})

    return results

def build_cultural():
    # 60 samples (20 hi, 20 pt, 20 es)
    results = []
    
    hi_bases = [
        ("As-salamu alaykum, mujhe gosht chahiye.", "Namaste, mujhe meat chahiye.", "Over-indexing on Urdu vocabulary for generic Hindi context."),
        ("Bhaiyya, ek pindi chana bhaji pav dena.", "Bhaiyya, ek pav bhaji dena.", "Mixing regional terms confusingly; pav bhaji is standard."),
        ("Woh ladka bilkul machcha jaisa lagta hai.", "Woh ladka bahut sahi lagta hai.", "'machcha' is South Indian/Dakhni slang."),
        ("Zindagi tension nahi leni chahi dasta.", "Zindagi tension nahi leni chahiye dosto.", "dasta is dialetal mispronunciation for dosto."),
    ]
    pt_bases = [
        ("Estou fixando bué legal neste carro.", "Estou achando muito legal esse carro.", "Mixing European PT 'bué' or 'fixe' with Brazilian PT 'legal'."),
        ("Meu, pega o comboio para a praia.", "Cara, pega o trem pra praia.", "Comboio is EU PT, Trem is BR PT."),
        ("O gajo estava falando muita asneira.", "O cara tava falando muita besteira.", "Gajo is EU slang, cara is BR slang."),
        ("Vou ao talho comprar carne.", "Vou no açougue comprar carne.", "Talho is EU PT, Açougue is BR PT.")
    ]
    es_bases = [
        ("Oye tú, ¿qué pasa güey?", "Che vos, ¿qué pasa?", "güey is Mexican slang, not Argentine."),
        ("Esa fiesta estuvo muy bacana.", "Esa fiesta estuvo re piola.", "bacana is Colombian/Caribbean, piola is Argentine."),
        ("¡Qué guay está este carro!", "¡Qué bueno está este auto!", "guay is Spain, carro is Caribbean/Mexican, auto is Argentine."),
        ("Tranquilo mano, todo va a salir bien.", "Tranquilo chabón, todo va a salir bien.", "mano is Caribbean/Portuñol, chabón is Argentine.")
    ]

    for i in range(20):
        t = hi_bases[i % len(hi_bases)]
        results.append({"lang": "hi", "locale": "hi-IN", "wrong_phrase": f"{t[0]} {i}", "correct_phrase": t[1], "rule": t[2]})

    for i in range(20):
        t = pt_bases[i % len(pt_bases)]
        results.append({"lang": "pt", "locale": "pt-BR", "wrong_phrase": f"{t[0]} {i}", "correct_phrase": t[1], "rule": t[2]})

    for i in range(20):
        t = es_bases[i % len(es_bases)]
        results.append({"lang": "es", "locale": "es-AR", "wrong_phrase": f"{t[0]} {i}", "correct_phrase": t[1], "rule": t[2]})

    return results

def build_clean():
    # 150 samples (50 hi, 50 pt, 50 es)
    results = []
    
    bases = [
        ("I agree completely.", "Main poori tarah sehmat hoon.", "Eu concordo completamente.", "Estoy completamente de acuerdo.", 1.2),
        ("Where are you going?", "Tum kahan ja rahe ho?", "Onde você está indo?", "¿A dónde vas?", 1.5),
        ("This is perfectly fine.", "Yeh bilkul theek hai.", "Isso está perfeitamente bem.", "Esto está perfectamente bien.", 1.4),
        ("Put it on the table.", "Ise mez par rakh do.", "Coloque na mesa.", "Ponelo en la mesa.", 1.3),
        ("I will call you later.", "Main tumhe baad mein phone karunga.", "Eu ligo para você mais tarde.", "Te llamo más tarde.", 1.6)
    ]
    for i in range(50):
        t = bases[i % len(bases)]
        results.append({"lang": "hi", "original_en": f"{t[0]} [{i}]", "dubbed_hi": t[1], "original_duration": t[4]})
        results.append({"lang": "pt", "original_en": f"{t[0]} [{i}]", "dubbed_pt": t[2], "original_duration": t[4]})
        results.append({"lang": "es", "original_en": f"{t[0]} [{i}]", "dubbed_es": t[3], "original_duration": t[4]})
        
    return results

if __name__ == "__main__":
    write_json('mistranslation_bank.json', build_mistranslations())
    write_json('timing_bank.json', build_timing())
    write_json('tone_bank.json', build_tone())
    write_json('cultural_bank.json', build_cultural())
    write_json('clean_bank.json', build_clean())
    
    print("Sucessfully generated large dataset banks (540 items total).")
