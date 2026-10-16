"""Dalle misure all'analisi: quali modelli entrano, a che velocità, per quali usi."""

import json
import os

GB = 1024 ** 3
EVAL_CTX = 8192          # contesto usato per verificare se un modello "entra"
GPU_RUNTIME_BYTES = 0.8 * GB   # buffer di calcolo e contesto CUDA/Metal/Vulkan
OS_RESERVED_RAM = 4 * GB
DEFAULT_GPU_OVERHEAD_S = 0.003  # costo fisso per token (lancio kernel, campionamento)
CPU_OVERHEAD_S = 0.001
MIN_USABLE_TPS = 5
MAX_PROMPT_WAIT_S = 20  # attesa massima accettabile prima che inizi la risposta

SPEED_LABELS = [(30, "veloce"), (15, "fluido"), (8, "usabile"), (3, "lento"), (0, "molto lento")]
PLACEMENT_LABELS = {"gpu": "Tutto in GPU", "split": "GPU + RAM", "cpu": "Solo CPU", "no": "Non entra"}

VERDICTS = [
    ("Non adatto", "Il sistema non riesce a far girare nemmeno i modelli piccoli a una velocità utilizzabile."),
    ("Uso di base", "Adatto a modelli piccoli (3-4 miliardi di parametri): riassunti, bozze, domande semplici, uso personale."),
    ("Adeguato per iniziare", "Gestisce modelli da 8 miliardi di parametri: un chatbot sui documenti per pochi utenti è realistico."),
    ("Buono per l'azienda", "Fa girare modelli da 12-24 miliardi di parametri: RAG, agenti e automazioni con buona qualità."),
    ("Ottimo", "Modelli da 27-32 miliardi di parametri o MoE di classe superiore: qualità vicina ai servizi cloud per molti compiti."),
    ("Server AI", "Modelli da 70 miliardi di parametri e oltre: qualità alta e più utenti contemporanei."),
]

LLM_USE_CASES = [
    {"id": "rag", "name": "Chatbot sui documenti aziendali (RAG)", "uses": "rag", "min_tier": 2, "good_tier": 3,
     "min_tps": 10, "ctx": 16384,
     "desc": "Risposte basate su manuali, contratti e procedure interne, con citazione delle fonti."},
    {"id": "helpdesk", "name": "Assistente clienti / helpdesk interno", "uses": "chat", "min_tier": 2, "good_tier": 3,
     "min_tps": 15, "ctx": 8192, "desc": "Risposte rapide a clienti o colleghi, anche più conversazioni insieme."},
    {"id": "agents", "name": "Agenti AI e automazioni", "uses": "tools", "min_tier": 3, "good_tier": 4,
     "min_tps": 15, "ctx": 32768, "desc": "Il modello usa strumenti (email, gestionale, API) e porta a termine compiti in più passaggi."},
    {"id": "sql", "name": "Domande al database in linguaggio naturale (Text-to-SQL)", "uses": "sql", "min_tier": 3,
     "good_tier": 4, "min_tps": 10, "ctx": 16384, "desc": "Report e analisi sul gestionale senza scrivere query."},
    {"id": "coding", "name": "Assistente di programmazione", "uses": "coding", "min_tier": 4, "good_tier": 5,
     "min_tps": 20, "ctx": 32768, "desc": "Completamento e revisione del codice, agenti di coding in locale."},
    {"id": "writing", "name": "Riassunti, traduzioni, bozze di email", "uses": "chat", "min_tier": 1, "good_tier": 2,
     "min_tps": 8, "ctx": 8192, "desc": "Compiti di scrittura quotidiani."},
    {"id": "vision", "name": "Analisi di immagini e documenti scansionati", "uses": "vision", "min_tier": 1,
     "good_tier": 3, "min_tps": 8, "ctx": 8192, "desc": "Lettura di fatture, moduli e foto con modelli multimodali."},
]


def load_catalog():
    path = os.path.join(os.path.dirname(__file__), "data", "models.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def speed_label(tps):
    for threshold, label in SPEED_LABELS:
        if tps >= threshold:
            return label
    return SPEED_LABELS[-1][1]


# ---------------------------------------------------------------- risorse di calcolo

def compute_profile(system, gpus, bench=None, ollama=None):
    """Riduce l'hardware a: memoria GPU utilizzabile, banda effettiva GPU e CPU, overhead per token."""
    ram = system["ram_total"]
    discrete = [g for g in gpus if not g["integrated"] and g["vendor"] in ("nvidia", "amd", "intel")
                and g.get("vram_bytes")]
    usable_driver = [g for g in discrete if not any("non attivo" in n for n in g["notes"])]
    primary = None
    group = []
    if usable_driver:
        primary = usable_driver[0]
        group = [g for g in usable_driver if g["vendor"] == primary["vendor"]]
    elif any(g["vendor"] == "apple" for g in gpus):
        primary = next(g for g in gpus if g["vendor"] == "apple")
        group = [primary]
    elif any(g["integrated"] for g in gpus):
        primary = next(g for g in gpus if g["integrated"])
        group = [primary]

    p = {"primary": primary, "gpu_count": len(group), "unified": bool(primary and primary["unified"]),
         "usable_gpu_bytes": 0, "gpu_bw": None, "gpu_bw_source": None,
         "gpu_overhead_s": DEFAULT_GPU_OVERHEAD_S, "calibrated": False, "notes": []}

    if primary:
        if p["unified"]:
            if primary["vendor"] == "apple":
                frac = 0.75 if ram >= 64 * GB else 0.67
            else:
                frac = 0.5  # le iGPU usano di norma fino a metà della RAM
            p["usable_gpu_bytes"] = max(int(ram * frac), primary.get("vram_bytes") or 0)
        else:
            p["usable_gpu_bytes"] = sum(g["vram_bytes"] for g in group) - int(0.4 * GB) * len(group)
            used = sum(g.get("vram_used_bytes") or 0 for g in group)
            if used > 1.5 * GB:
                p["notes"].append(f"Al momento del test {used / GB:.1f} GB di VRAM erano già occupati da altri programmi.")

        theo = min((g["bandwidth_gbs"] for g in group if g.get("bandwidth_gbs")), default=None)
        if not theo and p["unified"] and primary["vendor"] != "apple":
            theo = system.get("ram_bandwidth_gbs")
        measured = None
        if bench and bench.get("bandwidth_gbs") and _same_gpu(bench.get("adapter", ""), primary["name"]):
            measured = bench["bandwidth_gbs"]
        if measured and theo:
            p["gpu_bw"], p["gpu_bw_source"] = min(measured, 0.9 * theo), "misurata"
        elif measured:
            p["gpu_bw"], p["gpu_bw_source"] = measured, "misurata"
        elif theo:
            p["gpu_bw"], p["gpu_bw_source"] = 0.8 * theo, "stimata da scheda tecnica"

        # Calibrazione con il test reale di Ollama
        if ollama and ollama.get("effective_bandwidth_gbs") and p["gpu_bw"]:
            per_token = 1 / ollama["gen_tps"]
            transfer = ollama["model_size_bytes"] / (p["gpu_bw"] * 1e9)
            p["gpu_overhead_s"] = min(max(per_token - transfer, 0.0005), 0.015)
            p["calibrated"] = True

    cpu_bw = system.get("ram_bandwidth_gbs") or 30.0
    p["cpu_bw"] = cpu_bw
    p["ram_for_models"] = max(0, int((ram - OS_RESERVED_RAM) * 0.9))
    p["prompt_ref"] = _prompt_reference(primary, bench, ollama, system)
    return p


def _same_gpu(a, b):
    a, b = a.lower(), b.lower()
    tokens = [t for t in b.replace("nvidia", "").replace("geforce", "").split() if any(c.isdigit() for c in t)]
    return a in b or b in a or (tokens and all(t in a for t in tokens))


def _prompt_reference(primary, bench, ollama, system):
    """Velocità di lettura del prompt (token/s) normalizzata a un modello da 1 miliardo di parametri."""
    if ollama and ollama.get("prompt_tps") and ollama.get("gpu_fraction", 0) >= 0.99:
        params = _parse_params(ollama.get("parameter_size")) or 3.0
        return {"tps_per_b": ollama["prompt_tps"] * params, "source": "misurata con Ollama"}
    if primary and bench and bench.get("fp32_tflops"):
        boost = 6 if primary["vendor"] == "nvidia" and (primary.get("compute_cap") or 0) >= 7 else 2
        return {"tps_per_b": bench["fp32_tflops"] * boost * 0.4 * 1e12 / 2e9, "source": "stimata dal benchmark"}
    if primary and primary.get("bandwidth_gbs"):
        # Approssimazione prudente: la potenza di calcolo cresce insieme alla banda
        return {"tps_per_b": primary["bandwidth_gbs"] * 40, "source": "stimata da scheda tecnica"}
    threads = system["cpu"].get("threads") or 4
    return {"tps_per_b": threads * 30, "source": "stimata (CPU)", "cpu_only": True}


def _parse_params(s):
    try:
        s = s.upper().strip()
        if s.endswith("B"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1000
    except (AttributeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------- modelli

def evaluate_model(model, quant, prof, disk_free=None):
    bits = quant["bits"]
    weights = model["params_b"] * 1e9 * bits / 8
    active = (model.get("active_b") or model["params_b"]) * 1e9 * bits / 8
    kv_per_token = model["kv_kb"] * 1024
    kv = kv_per_token * EVAL_CTX
    per_token_bytes = active + kv_per_token * 2048  # lettura della KV cache a metà conversazione
    usable = prof["usable_gpu_bytes"]
    r = {"quant": quant["id"], "weights_gb": round(weights / GB, 1),
         "memory_gb": round((weights + kv + GPU_RUNTIME_BYTES) / GB, 1)}

    if prof["gpu_bw"] and weights + kv + GPU_RUNTIME_BYTES <= usable:
        r["placement"] = "gpu"
        t = per_token_bytes / (prof["gpu_bw"] * 1e9) + prof["gpu_overhead_s"]
        free = usable - weights - GPU_RUNTIME_BYTES
        r["max_ctx"] = int(min(model["ctx_max"], free / kv_per_token))
        r["gpu_fraction"] = 1.0
    elif prof["gpu_bw"] and not prof["unified"] and usable > kv + GPU_RUNTIME_BYTES \
            and weights - (usable - kv - GPU_RUNTIME_BYTES) <= prof["ram_for_models"]:
        f = (usable - kv - GPU_RUNTIME_BYTES) / weights
        if f < 0.1:
            return _cpu_only(r, weights, per_token_bytes, prof, model)
        r["placement"] = "split"
        r["gpu_fraction"] = round(f, 2)
        t = (f * per_token_bytes / (prof["gpu_bw"] * 1e9) + (1 - f) * per_token_bytes / (prof["cpu_bw"] * 1e9)
             + prof["gpu_overhead_s"] + CPU_OVERHEAD_S)
        r["max_ctx"] = EVAL_CTX
    elif not prof["unified"] or not prof["gpu_bw"]:
        return _cpu_only(r, weights, per_token_bytes, prof, model)
    else:
        r["placement"] = "no"
        return r

    r["tps"] = round(1 / t, 1)
    r["prompt_tps"] = _prompt_speed(model, prof, r["gpu_fraction"])
    if disk_free and weights > disk_free:
        r["disk_warning"] = True
    return r


def _cpu_only(r, weights, per_token_bytes, prof, model):
    if weights + model["kv_kb"] * 1024 * EVAL_CTX > prof["ram_for_models"]:
        r["placement"] = "no"
        return r
    r["placement"] = "cpu"
    r["gpu_fraction"] = 0.0
    r["tps"] = round(1 / (per_token_bytes / (prof["cpu_bw"] * 1e9) + CPU_OVERHEAD_S), 1)
    r["max_ctx"] = EVAL_CTX
    r["prompt_tps"] = _prompt_speed(model, prof, 0.0)
    return r


def _prompt_speed(model, prof, gpu_fraction):
    active_b = model.get("active_b") or model["params_b"]
    ref = prof["prompt_ref"]
    gpu_pp = ref["tps_per_b"] / active_b if not ref.get("cpu_only") else None
    cpu_pp = (os.cpu_count() or 4) * 30 / active_b
    if gpu_pp is None or gpu_fraction == 0:
        return round(cpu_pp)
    return round(gpu_pp * max(gpu_fraction, 0.15) if gpu_fraction < 1 else gpu_pp)


def evaluate_models(catalog, prof, disk_free=None):
    out = []
    for m in catalog["models"]:
        quants = [evaluate_model(m, q, prof, disk_free) for q in catalog["quants"]]
        out.append({"model": m, "quants": quants, "q4": quants[0]})
    return out


def effective_tier(entry):
    """Classe del modello, un livello in meno se non sta tutto in GPU.

    Con il modello diviso tra GPU e RAM (o solo su CPU) la lettura dei documenti
    lunghi è molto più lenta e le prestazioni calano con più utenti.
    """
    penalty = 0 if entry["q4"]["placement"] == "gpu" else 1
    return max(entry["model"]["tier"] - penalty, 1)


def _runs_well(entry):
    q = entry["q4"]
    if q["placement"] == "gpu":
        return q["tps"] >= 12
    return q["placement"] in ("split", "cpu") and q["tps"] >= 15


def best_model(models):
    """Il modello di classe più alta che gira bene, a parità di classe il più veloce."""
    good = [e for e in models if _runs_well(e)]
    return max(good, key=lambda e: (effective_tier(e), e["q4"]["tps"]), default=None)


def overall_tier(models):
    """Livello complessivo. Dal 3 in su ("per l'azienda") il modello deve stare tutto in GPU:
    diviso tra GPU e RAM serve bene un solo utente alla volta."""
    tiers = [effective_tier(e) if e["q4"]["placement"] == "gpu" else min(effective_tier(e), 2)
             for e in models if _runs_well(e)]
    if tiers:
        return max(tiers)
    if any(e["q4"]["placement"] != "no" and e["q4"]["tps"] >= MIN_USABLE_TPS for e in models):
        return 1
    return 0


def concurrency(models, prof):
    """Conversazioni contemporanee da 8k token che il miglior modello completamente in GPU può servire."""
    best = None
    for e in models:
        q4 = e["q4"]
        if q4["placement"] == "gpu" and q4["tps"] >= 15 and (not best or e["model"]["tier"] > best["model"]["tier"]):
            best = e
    if not best:
        return None
    m = best["model"]
    weights = m["params_b"] * 1e9 * 4.85 / 8
    free = prof["usable_gpu_bytes"] - weights - GPU_RUNTIME_BYTES
    sessions = int(free // (m["kv_kb"] * 1024 * EVAL_CTX))
    return {"model": m["name"], "sessions": max(1, min(sessions, 64)), "tps_single": best["q4"]["tps"]}


# ---------------------------------------------------------------- casi d'uso

def evaluate_use_cases(models, prof, gpus):
    results = []
    for uc in LLM_USE_CASES:
        candidates = []
        for e in models:
            m, q4 = e["model"], e["q4"]
            if uc["uses"] not in m["uses"] or q4["placement"] == "no" or q4["tps"] < uc["min_tps"]:
                continue
            # In GPU il contesto deve entrare in VRAM; con GPU + RAM la KV cache può crescere in RAM
            if q4["placement"] == "gpu" and q4["max_ctx"] < min(uc["ctx"], m["ctx_max"]):
                continue
            # Un prompt di metà contesto (documenti, cronologia) va letto entro MAX_PROMPT_WAIT_S
            if q4["prompt_tps"] < uc["ctx"] / 2 / MAX_PROMPT_WAIT_S:
                continue
            candidates.append(e)
        best = max(candidates, key=lambda e: (effective_tier(e), e["q4"]["tps"]), default=None)
        if not best:
            rating, model, tps = "no", None, None
        else:
            tier, tps = effective_tier(best), best["q4"]["tps"]
            model = best["model"]["name"]
            if tier >= uc["good_tier"] and tps >= 2 * uc["min_tps"]:
                rating = "ottimo"
            elif tier >= uc["good_tier"]:
                rating = "buono"
            elif tier >= uc["min_tier"]:
                rating = "sufficiente"
            else:
                rating = "limitato"
        results.append({"id": uc["id"], "name": uc["name"], "desc": uc["desc"], "rating": rating,
                        "model": model, "tps": tps,
                        "placement": best["q4"]["placement"] if best else None})
    results.extend(_non_llm_use_cases(prof, gpus))
    return results


def _non_llm_use_cases(prof, gpus):
    p = prof["primary"]
    vram = prof["usable_gpu_bytes"] / GB if p else 0
    vendor = p["vendor"] if p else None
    nvidia = vendor == "nvidia"
    out = []

    if not p:
        rating, note = "limitato", "Funziona su CPU con whisper.cpp, ma circa 1-3 volte il tempo reale dell'audio."
    elif vram >= 6:
        rating, note = "ottimo", "Whisper large-v3 in GPU: un'ora di audio in pochi minuti."
    elif vram >= 2:
        rating, note = "buono", "Whisper large-v3-turbo quantizzato entra in GPU."
    else:
        rating, note = "sufficiente", "Modelli Whisper piccoli o medi."
    out.append({"id": "speech", "name": "Trascrizione di audio e riunioni (Whisper)", "rating": rating, "note": note,
                "desc": "Verbali, interviste, chiamate al servizio clienti."})

    if not p or (vendor == "intel" and p["integrated"]):
        rating, note = "no", "Senza una GPU adeguata la generazione di immagini è troppo lenta per l'uso reale."
    elif vram >= 16 and vendor in ("nvidia", "apple", "amd"):
        rating, note = "ottimo", "SDXL e FLUX in buona qualità."
    elif vram >= 8:
        rating, note = "buono", "SDXL e FLUX quantizzato."
    elif vram >= 4:
        rating, note = "sufficiente", "Stable Diffusion 1.5 e SDXL con ottimizzazioni per poca VRAM."
    else:
        rating, note = "no", "VRAM insufficiente."
    if rating in ("ottimo", "buono") and vendor == "amd":
        note += " Su AMD serve ROCm (Linux) o ZLUDA."
    out.append({"id": "images", "name": "Generazione di immagini (Stable Diffusion, FLUX)", "rating": rating,
                "note": note, "desc": "Immagini per marketing, prototipi, cataloghi."})

    if nvidia and vram >= 40:
        rating, note = "ottimo", "LoRA anche su modelli da 30B+, QLoRA fino a 70B."
    elif nvidia and vram >= 20:
        rating, note = "buono", "QLoRA su modelli fino a 30B circa, LoRA su modelli da 8B."
    elif nvidia and vram >= 10:
        rating, note = "sufficiente", "QLoRA su modelli da 7-8B (es. con Unsloth)."
    elif vendor == "apple" and vram >= 16:
        rating, note = "sufficiente", "LoRA con MLX su modelli fino a ~8B (14B con 32 GB e più)."
    elif vendor == "amd" and vram >= 16:
        rating, note = "limitato", "Possibile con ROCm su Linux, ma con meno strumenti pronti."
    else:
        rating, note = "no", "Serve una GPU NVIDIA con almeno 10-12 GB di VRAM."
    out.append({"id": "finetune", "name": "Addestramento su misura (fine-tuning LoRA/QLoRA)", "rating": rating,
                "note": note, "desc": "Insegnare al modello il linguaggio e i formati della propria azienda."})
    return out


# ---------------------------------------------------------------- framework

def evaluate_frameworks(system, prof):
    osf = system["os_family"]
    sw = system["software"]
    p = prof["primary"]
    vendor = p["vendor"] if p else None
    cc = (p or {}).get("compute_cap") or 0
    integrated = bool(p and p["integrated"])
    linux, windows, mac = osf == "Linux", osf == "Windows", osf == "Darwin"
    apple_silicon = vendor == "apple"
    fw = []

    def add(name, status, note, desc, installed=None):
        fw.append({"name": name, "status": status, "note": note, "desc": desc, "installed": installed})

    # Ollama
    if vendor == "nvidia":
        st, note = ("ok", "Accelerazione CUDA.") if cc >= 5 or not cc else ("no", "GPU troppo vecchia per CUDA in Ollama.")
    elif vendor == "amd" and not integrated:
        st, note = "warn", "ROCm: supportate RX 6800 e successive, serie 7000/9000 e Instinct; altrimenti backend Vulkan."
    elif apple_silicon:
        st, note = "ok", "Accelerazione Metal sulla memoria unificata."
    elif vendor in ("intel", "amd"):
        st, note = "warn", "Backend Vulkan per GPU integrate/Intel: funziona, prestazioni variabili."
    else:
        st, note = "cpu", "Funziona solo su CPU."
    add("Ollama", st, note, "Il modo più semplice per avviare un LLM in locale, con API compatibile OpenAI.",
        sw.get("ollama"))

    # llama.cpp
    backend = {"nvidia": "CUDA", "amd": "ROCm/HIP o Vulkan", "apple": "Metal", "intel": "SYCL o Vulkan"}.get(vendor)
    add("llama.cpp", "ok" if backend else "cpu",
        f"Backend {backend}." if backend else "Ottimizzato anche per CPU (AVX2/AVX-512).",
        "Il motore su cui si basano Ollama e LM Studio: massima compatibilità, formato GGUF.")

    # LM Studio
    if mac and not apple_silicon:
        st, note = "no", "Richiede un Mac con Apple Silicon."
    elif system["arch"].lower() in ("aarch64", "arm64") and linux:
        st, note = "warn", "Su Linux ARM il supporto è limitato."
    else:
        st, note = ("ok", "Interfaccia grafica, backend " + ("MLX/Metal" if apple_silicon else "CUDA/Vulkan") + ".") \
            if p else ("cpu", "Funziona su CPU.")
    add("LM Studio", st, note, "Applicazione desktop per provare i modelli senza riga di comando.", sw.get("lm_studio"))

    # vLLM
    if vendor == "nvidia" and cc and cc < 7:
        st, note = "no", f"Serve compute capability 7.0+ (la tua GPU è {cc})."
    elif vendor == "nvidia" and linux:
        st, note = "ok", "Ideale per servire più utenti contemporaneamente (batching continuo)."
        if cc and cc < 8:
            note += " Senza Ampere (8.0+) mancano FlashAttention 2 e bf16."
    elif vendor == "nvidia" and windows:
        st, note = "warn", "Non nativo su Windows: si usa via WSL2 o Docker Desktop."
    elif vendor == "amd" and linux and not integrated:
        st, note = "warn", "Con ROCm su Instinct MI200/MI300 e Radeon RX 7900/9000."
    elif apple_silicon:
        st, note = "no", "Nessuna accelerazione GPU su Mac: usare Ollama, LM Studio o MLX."
    else:
        st, note = "no", "Richiede GPU NVIDIA o AMD su Linux."
    add("vLLM", st, note, "Server di inferenza ad alte prestazioni per produzione e multi-utente.")

    # SGLang / TensorRT-LLM
    if vendor == "nvidia" and cc >= 8 and linux:
        add("TensorRT-LLM / SGLang", "ok", "Massime prestazioni su NVIDIA Ampere e successive.",
            "Motori di inferenza ottimizzati per server NVIDIA.")
    elif vendor == "nvidia" and cc >= 7.5:
        add("TensorRT-LLM / SGLang", "warn",
            "Supporto parziale" + (" (su Windows via WSL2)." if windows else " su GPU pre-Ampere."),
            "Motori di inferenza ottimizzati per server NVIDIA.")
    else:
        add("TensorRT-LLM / SGLang", "no", "Pensati per GPU NVIDIA recenti su Linux.",
            "Motori di inferenza ottimizzati per server NVIDIA.")

    # PyTorch / Transformers
    if vendor == "nvidia":
        st, note = "ok", "CUDA" + (f" {p['cuda_version']} (driver)." if p.get("cuda_version") else ".")
    elif vendor == "amd" and not integrated:
        st, note = ("ok" if linux else "warn"), "ROCm su Linux" + ("" if linux else "; su Windows supporto ancora parziale") + "."
    elif apple_silicon:
        st, note = "warn", "Backend MPS: funziona, ma alcune operazioni ricadono sulla CPU."
    elif vendor == "intel":
        st, note = "warn", "Backend XPU per Intel Arc e GPU integrate recenti."
    else:
        st, note = "cpu", "Solo CPU: adatto a modelli piccoli ed embedding."
    add("PyTorch / Hugging Face", st, note, "La base per sviluppo, embedding, fine-tuning e modelli non-LLM.",
        sw.get("cuda_toolkit") if vendor == "nvidia" else None)

    # MLX
    add("MLX", "ok" if apple_silicon else "no",
        "Ottimizzato per Apple Silicon, spesso più veloce di llama.cpp su Mac." if apple_silicon else "Solo Mac con Apple Silicon.",
        "Framework di Apple per inferenza e fine-tuning sulla memoria unificata.")

    # Unsloth
    if vendor == "nvidia" and cc >= 7:
        st, note = "ok", "Fine-tuning LoRA/QLoRA veloce e con poca VRAM."
    elif vendor == "nvidia":
        st, note = "no", "Serve compute capability 7.0+."
    elif vendor == "amd" and linux and not integrated:
        st, note = "warn", "Supporto AMD in fase iniziale."
    else:
        st, note = "no", "Richiede GPU NVIDIA."
    add("Unsloth", st, note, "Fine-tuning efficiente dei modelli open source.")

    # OpenVINO
    add("OpenVINO", "ok" if vendor == "intel" or "intel" in system["cpu"]["name"].lower() else "warn",
        "Ottimizzato per CPU, GPU e NPU Intel." if vendor == "intel" or "intel" in system["cpu"]["name"].lower()
        else "Pensato per hardware Intel.", "Toolkit di Intel per l'inferenza.")

    # Docker con GPU
    if mac:
        st, note = "no", "I container su Mac non accedono alla GPU: i modelli vanno eseguiti fuori da Docker."
    elif vendor == "nvidia" and linux:
        if sw.get("nvidia_container_toolkit"):
            st, note = "ok", "NVIDIA Container Toolkit installato."
        else:
            st, note = "warn", "Installa NVIDIA Container Toolkit per dare la GPU ai container."
    elif vendor == "nvidia" and windows:
        st, note = "ok", "Docker Desktop con backend WSL2 supporta le GPU NVIDIA."
    elif vendor == "amd" and linux:
        st, note = "warn", "Possibile con le immagini ROCm (/dev/kfd e /dev/dri)."
    else:
        st, note = "warn", "Accelerazione GPU nei container limitata su questa configurazione."
    add("Docker con GPU", st, note, "Per installare stack completi (RAGFlow, Open WebUI, vLLM) in modo riproducibile.",
        sw.get("docker"))
    return fw


# ---------------------------------------------------------------- consigli

def recommendations(system, gpus, prof, tier, ollama):
    rec = []
    p = prof["primary"]
    ram_gb = system["ram_total"] / GB
    vram_gb = prof["usable_gpu_bytes"] / GB

    for g in gpus:
        for n in g["notes"]:
            if "non attivo" in n:
                rec.append(f"{g['name']}: installa il driver NVIDIA proprietario, senza il quale la GPU resta inutilizzata.")
    if p and not p["unified"]:
        if vram_gb < 7:
            rec.append("Con una GPU da 16 GB (es. RTX 5060 Ti 16 GB o RTX 4060 Ti 16 GB) passeresti a modelli da 12-14B, "
                       "la soglia di qualità per un uso aziendale.")
        elif vram_gb < 15:
            rec.append("Con 24 GB di VRAM (RTX 3090, RTX 4090) potresti usare modelli da 27-32B, molto più affidabili "
                       "per agenti e Text-to-SQL.")
        elif vram_gb < 40:
            rec.append("Per modelli da 70B o per molti utenti servono 48 GB e oltre (RTX 6000 Ada, RTX PRO 6000, 2 GPU da 24 GB).")
    if not p:
        rec.append("Senza GPU il sistema è adatto solo a test con modelli piccoli: per l'uso reale valuta una GPU NVIDIA da 16 GB "
                   "o un Mac con Apple Silicon e almeno 32 GB di memoria.")
    if p and p["vendor"] == "apple":
        rec.append("Su Mac la quota di memoria usabile dalla GPU si può alzare (sudo sysctl iogpu.wired_limit_mb=...) "
                   "per caricare modelli più grandi, lasciando almeno 8 GB al sistema.")
        if ram_gb < 32:
            rec.append("Per un uso aziendale su Mac conviene partire da 32 GB di memoria unificata (meglio 64 GB).")
    if p and not p["unified"] and ram_gb < 32:
        rec.append(f"La RAM ({ram_gb:.0f} GB) limita i modelli che non entrano in GPU: con 32-64 GB potresti caricare "
                   "modelli MoE grandi in modalità GPU + RAM.")
    disk = system.get("disk_free")
    if disk is not None and disk < 50 * GB:
        rec.append(f"Spazio libero su disco basso ({disk / GB:.0f} GB): ogni modello occupa da 2 a oltre 40 GB.")
    if not system["software"].get("ollama"):
        rec.append("Per una prova immediata installa Ollama (ollama.com) e rilancia il test: misurerà la velocità reale.")
    elif ollama and not ollama.get("running"):
        rec.append("Avvia Ollama e rilancia il test per misurare la velocità reale invece di stimarla.")
    if system["os_family"] == "Windows" and p and p["vendor"] == "nvidia" and tier >= 3:
        rec.append("Per un server multi-utente su Windows usa vLLM tramite WSL2, oppure valuta Linux sulla macchina dedicata.")
    if tier >= 3:
        rec.append("Per più utenti contemporanei usa un server con batching (vLLM, o llama.cpp server con --parallel) "
                   "invece dell'app desktop.")
    return rec


def analyze(system, gpus, bench=None, ollama=None):
    catalog = load_catalog()
    prof = compute_profile(system, gpus, bench, ollama)
    models = evaluate_models(catalog, prof, system.get("disk_free"))
    tier = overall_tier(models)
    title, text = VERDICTS[tier]
    return {
        "profile": prof,
        "models": models,
        "catalog_updated": catalog.get("updated"),
        "tier": tier,
        "verdict": {"title": title, "text": text},
        "concurrency": concurrency(models, prof),
        "use_cases": evaluate_use_cases(models, prof, gpus),
        "frameworks": evaluate_frameworks(system, prof),
        "recommendations": recommendations(system, gpus, prof, tier, ollama),
    }


def round_gb(b):
    return None if b is None else round(b / GB, 1)

