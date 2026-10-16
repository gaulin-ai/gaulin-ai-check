"""Test reale con Ollama, se installato: misura i token/s di un modello vero.

Il risultato serve anche a calibrare le stime per tutti gli altri modelli.
"""

import json
import os
import shutil
import urllib.error
import urllib.request

# Modelli densi piccoli, in ordine di preferenza: con questi la calibrazione è affidabile
PREFERRED = ["llama3.2:3b", "qwen2.5:3b", "qwen3:4b", "gemma3:4b", "llama3.1:8b",
             "qwen3:8b", "qwen2.5:7b", "mistral:7b", "gemma3:12b", "qwen3:14b"]
DEFAULT_PULL = "llama3.2:3b"
MOE_FAMILIES = {"qwen3moe", "gptoss", "llama4", "deepseek2", "mixtral", "olmoe", "granitemoe"}

PROMPT = (
    "Sei un consulente che aiuta una piccola azienda italiana a organizzare i propri documenti. "
    "Spiega in modo chiaro e ordinato, con un elenco puntato, come impostare un archivio digitale "
    "dei contratti dei fornitori: quali cartelle creare, come nominare i file, chi deve avere accesso, "
    "ogni quanto controllare le scadenze e come gestire le versioni dei documenti. "
    "Aggiungi infine tre errori comuni da evitare e un breve riepilogo finale."
)


def _base_url():
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if "://" not in host:
        host = "http://" + host
    host = host.replace("://0.0.0.0", "://127.0.0.1")
    if host.count(":") == 1:  # senza porta
        host += ":11434"
    return host.rstrip("/")


def _call(path, payload=None, timeout=10):
    url = _base_url() + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _is_chat_model(m):
    caps = m.get("capabilities")
    if caps is not None:
        return "completion" in caps
    fam = (m.get("details") or {}).get("family", "")
    return "bert" not in fam and "embed" not in m["name"]


def _pick_model(models):
    names = {m["name"]: m for m in models}
    for p in PREFERRED:
        if p in names:
            return names[p]
    dense = [m for m in models if _is_chat_model(m)
             and (m.get("details") or {}).get("family") not in MOE_FAMILIES]
    return min(dense, key=lambda m: m.get("size", 0)) if dense else None


def run(model=None, pull=False, log=print):
    result = {"installed": bool(shutil.which("ollama")), "running": False}
    try:
        result["version"] = _call("/api/version", timeout=3).get("version")
        result["running"] = True
    except (urllib.error.URLError, OSError, ValueError):
        if result["installed"]:
            log("  Ollama è installato ma non in esecuzione: avvialo per il test reale.")
        return result

    models = _call("/api/tags").get("models", [])
    result["models_installed"] = [m["name"] for m in models]
    chosen = next((m for m in models if m["name"] == model), None) if model else _pick_model(models)
    want = model or DEFAULT_PULL
    if chosen is None and pull:
        log(f"  Scarico {want} per il test (può richiedere qualche minuto)...")
        try:
            _call("/api/pull", {"model": want, "stream": False}, timeout=3600)
            models = _call("/api/tags").get("models", [])
            chosen = next((m for m in models if m["name"] == want), None)
        except (urllib.error.URLError, OSError, ValueError) as e:
            log(f"  Download non riuscito: {e}")
    if chosen is None:
        log("  Nessun modello adatto al test in Ollama (usa --pull per scaricarne uno piccolo).")
        return result

    name = chosen["name"]
    details = chosen.get("details") or {}
    result.update({"model": name, "model_size_bytes": chosen.get("size"),
                   "parameter_size": details.get("parameter_size"),
                   "quantization": details.get("quantization_level"),
                   "family": details.get("family")})
    log(f"  Carico {name} in memoria...")
    try:
        warm = _call("/api/generate", {"model": name, "prompt": "Ciao", "stream": False,
                                       "options": {"num_predict": 1}, "keep_alive": "2m"}, timeout=600)
        result["load_seconds"] = round(warm.get("load_duration", 0) / 1e9, 1)
        runs = []
        for i in range(2):
            log(f"  Generazione di prova {i + 1}/2...")
            r = _call("/api/generate", {"model": name, "prompt": PROMPT, "stream": False,
                                        "options": {"num_predict": 160, "temperature": 0, "seed": 42,
                                                    "num_ctx": 4096},
                                        "keep_alive": "2m"}, timeout=900)
            runs.append(r)
        gen = [r["eval_count"] / (r["eval_duration"] / 1e9) for r in runs if r.get("eval_duration")]
        result["gen_tps"] = round(max(gen), 1) if gen else None
        prompt_runs = [r for r in runs if r.get("prompt_eval_count", 0) > 50 and r.get("prompt_eval_duration")]
        if prompt_runs:
            r = prompt_runs[0]
            result["prompt_tps"] = round(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9), 1)

        ps = _call("/api/ps").get("models", [])
        loaded = next((m for m in ps if m.get("name") == name or m.get("model") == name), None)
        if loaded and loaded.get("size"):
            result["gpu_fraction"] = round(loaded.get("size_vram", 0) / loaded["size"], 2)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        log(f"  Test Ollama non riuscito: {e}")
        result["error"] = str(e)
    finally:
        try:  # libera la memoria della GPU
            _call("/api/generate", {"model": name, "keep_alive": 0}, timeout=30)
        except (urllib.error.URLError, OSError, ValueError):
            pass

    # Banda effettiva: per un modello denso ogni token legge tutti i pesi una volta
    if (result.get("gen_tps") and result.get("gpu_fraction", 0) >= 0.99
            and result.get("family") not in MOE_FAMILIES and result.get("model_size_bytes")):
        result["effective_bandwidth_gbs"] = round(result["gen_tps"] * result["model_size_bytes"] / 1e9, 1)
    return result
