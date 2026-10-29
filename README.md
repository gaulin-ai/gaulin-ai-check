# Gaulin AI Check

Verifica se un PC o un server è adatto a ospitare AI e LLM in locale, e produce un report HTML
leggibile anche da chi non è tecnico: verdetto complessivo, modelli che girano e a che velocità,
casi d'uso aziendali, compatibilità dei framework e consigli di upgrade.

Funziona su **Windows, Linux e macOS** con GPU **NVIDIA, AMD, Intel e Apple Silicon** (anche senza GPU).
Non invia nessun dato in rete.

Pagina del progetto, con spiegazione dei risultati e download: **[gaulin.ai/gaulin-ai-check](https://gaulin.ai/gaulin-ai-check.html)**.

## Cosa fa

| Fase | Come |
|---|---|
| Rilevamento GPU | `nvidia-smi` + NVML (banda calcolata dal bus di memoria), sysfs/`lspci` su Linux, registro su Windows, `system_profiler` su macOS; database integrato per le altre GPU |
| Benchmark GPU | Shader WebGPU (`wgpu`: Vulkan / Metal / DirectX 12) che misurano la **banda di memoria** e la **potenza FP32**. Su Windows gli shader crescono per gradi, così da non far scattare il watchdog del driver |
| Banda RAM | Copie multi-thread con numpy (serve quando un modello non entra in VRAM) |
| Test reale | Se Ollama è attivo, genera testo con un modello piccolo e misura i token/s reali, che **calibrano** tutte le stime |
| Analisi | Per ogni modello del catalogo e per ogni quantizzazione (Q4/Q8/FP16) calcola dove gira (GPU, GPU + RAM, CPU), la velocità e il contesto massimo |
| Report | HTML autonomo + JSON, aperto automaticamente nel browser |

### Come vengono stimate le velocità

La generazione di un LLM è limitata dalla banda di memoria: per ogni token si leggono tutti i pesi
attivi del modello. Di conseguenza:

```
tempo per token = byte letti / banda effettiva + costo fisso per token
```

- **Banda effettiva**: quella misurata dal benchmark (al massimo il 90% della teorica), altrimenti l'80% della banda da scheda tecnica.
- **Costo fisso**: ricavato dal test reale con Ollama; senza Ollama si usano 3 ms.
- **Modelli MoE** (gpt-oss, Qwen3-30B-A3B…): la memoria dipende dai parametri totali, la velocità solo da quelli attivi.
- **GPU + RAM**: la parte dei pesi che sta in RAM si legge alla velocità della RAM.
- **Mac / GPU integrate**: memoria unificata, la GPU può usarne il 67-75% (Apple) o circa il 50% (iGPU).

Verifica fatta su una RTX 3050 Laptop: calibrando su `llama3.2:3b`, la stima per `qwen2.5:3b` è stata
di 52,0 token/s contro 50,2 misurati.

## Uso

Eseguibile (dalla pagina Releases): doppio clic, oppure da terminale:

```bash
gaulin-ai-check                      # test completo, apre il report
gaulin-ai-check --pull               # se in Ollama non c'è un modello adatto, scarica llama3.2:3b (~2 GB)
gaulin-ai-check --ollama-model qwen3:8b
gaulin-ai-check --no-bench --no-ollama --out ./report   # solo rilevamento e stime
```

Da sorgente (Python 3.10+):

```bash
pip install -r requirements.txt      # tutte opzionali
python -m aicheck
```

## Sviluppo

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests            # test dell'analisi su configurazioni simulate
python build.py                      # crea dist/gaulin-ai-check-<os>-<arch>
```

GitHub Actions (`.github/workflows/build.yml`) compila per Linux, Windows e macOS (Apple Silicon)
a ogni push. Con un tag `v*` pubblica i tre eseguibili in una release.

### Aggiornare il catalogo modelli

`aicheck/data/models.json`: per ogni modello servono i parametri totali e attivi, la KV cache per token
(`2 × layer × kv_heads × head_dim × 2 byte`, ridotta se il modello usa sliding window), la classe
di qualità (1-5) e i casi d'uso.
Per i modelli con embedding per livello (Gemma 4 E2B/E4B) `params_b` comprende gli embedding e
`active_b` sono i parametri effettivi; con KV condivisa tra layer (E2B/E4B) o chiavi uguali ai valori
nei layer globali (Gemma 4 12B/26B/31B) la KV cache si conta una volta sola.

## Limiti noti

- Gli eseguibili non sono firmati: su Windows compare SmartScreen e su macOS Gatekeeper (Impostazioni di Sistema → Privacy e sicurezza → "Apri comunque").
  Per distribuirli al pubblico servono un certificato di code signing e la notarizzazione Apple.
- Il benchmark usa la GPU scelta da WebGPU come "high-performance"; con più GPU dedicate misura solo quella.
- AMD su Windows e Intel Arc sono rilevati dal registro/sysfs; la banda arriva dal database integrato.
- Le stime GPU + RAM sono validate solo in modo indiretto; i valori reali dipendono dal motore e dalle impostazioni.

## Autore

Sviluppato da [Gaulin.ai](https://gaulin.ai/), che progetta e installa server LLM on-premise per aziende italiane.
Per passare dal test all'uso: [guida all'LLM locale](https://gaulin.ai/local-llm-server.html) (installazione con Ollama, GPU e modelli).
