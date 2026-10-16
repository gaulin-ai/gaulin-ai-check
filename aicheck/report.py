"""Report HTML autonomo (un solo file, nessuna risorsa esterna) e riepilogo testuale."""

import datetime
import html

from . import __version__
from .analyze import GB, PLACEMENT_LABELS, best_model, speed_label

BRAND = "Gaulin AI"
BRAND_URL = "https://gaulin.ai"
CONTACT_URL = "https://gaulin.ai/contatti.html"

RATING = {"ottimo": ("Ottimo", "good"), "buono": ("Buono", "good"), "sufficiente": ("Sufficiente", "mid"),
          "limitato": ("Limitato", "low"), "no": ("Non adatto", "bad")}
FW_STATUS = {"ok": ("Supportato", "good"), "warn": ("Con limiti", "mid"), "cpu": ("Solo CPU", "low"),
             "no": ("Non supportato", "bad")}
PLACEMENT_CLASS = {"gpu": "good", "split": "mid", "cpu": "low", "no": "bad"}

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#1b1f24;--muted:#5d6670;--line:#e2e5e9;--accent:#0b6bcb;
--good:#137a3f;--good-bg:#e3f4ea;--mid:#8a5a00;--mid-bg:#fdf0d5;--low:#9a4a12;--low-bg:#fbe6d8;
--bad:#a3192d;--bad-bg:#fbe3e6;--meter:#dfe3e8}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--card:#1b1f24;--text:#e7eaee;--muted:#9aa4ae;
--line:#2c3239;--accent:#5aa9ff;--good:#6fd39a;--good-bg:#173524;--mid:#f0c060;--mid-bg:#3a2e12;
--low:#f2a070;--low-bg:#3b2416;--bad:#ff8a98;--bad-bg:#3d1a20;--meter:#2c3239}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:24px 16px 48px}
header{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:16px}
header h1{font-size:22px;margin:0}header .meta{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0}
h2{font-size:17px;margin:0 0 12px}h3{font-size:15px;margin:0 0 4px}
.verdict-title{font-size:28px;font-weight:700;margin:4px 0}
.meter{display:flex;gap:6px;margin:12px 0}.meter span{flex:1;height:10px;border-radius:5px;background:var(--meter)}
.meter span.on{background:var(--accent)}.meter-labels{display:flex;justify-content:space-between;
color:var(--muted);font-size:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:16px}
.kpi{border:1px solid var(--line);border-radius:10px;padding:12px}.kpi b{display:block;font-size:20px}
.kpi small{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px 6px;
border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600;font-size:13px}
.scroll{overflow-x:auto}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.good{color:var(--good);background:var(--good-bg)}.mid{color:var(--mid);background:var(--mid-bg)}
.low{color:var(--low);background:var(--low-bg)}.bad{color:var(--bad);background:var(--bad-bg)}
.cell small{display:block;color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.uc{border:1px solid var(--line);border-radius:10px;padding:14px}.uc p{margin:6px 0 0;color:var(--muted);font-size:13px}
.uc .top{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}
.muted{color:var(--muted)}ul{margin:0;padding-left:20px}li{margin:6px 0}
.cta{background:var(--accent);color:#fff;border:0}.cta a{color:#fff;font-weight:700}
.cta p{margin:6px 0 0}footer{color:var(--muted);font-size:12px;margin-top:24px}
@media print{body{background:#fff}.card{break-inside:avoid}}
"""


def _e(x):
    return html.escape(str(x)) if x is not None else "—"


def _gb(b, digits=1):
    return "—" if b is None else f"{b / GB:.{digits}f} GB"


def _badge(label, cls):
    return f'<span class="badge {cls}">{_e(label)}</span>'


def _quant_cell(q):
    cls = PLACEMENT_CLASS[q["placement"]]
    if q["placement"] == "no":
        return f'<td class="cell">{_badge("Non entra", cls)}<small>{q["memory_gb"]} GB necessari</small></td>'
    label = PLACEMENT_LABELS[q["placement"]]
    if q["placement"] == "split":
        label += f' ({int(q["gpu_fraction"] * 100)}% GPU)'
    speed = f"{q['tps']:.0f} tok/s"
    return (f'<td class="cell">{_badge(speed, cls)}'
            f'<small>{_e(label)} · {_e(speed_label(q["tps"]))}</small></td>')


def _hardware(system, gpus):
    rows = [("Sistema operativo", f'{system["os"]} ({system["arch"]})'),
            ("Processore", f'{system["cpu"]["name"]} — {system["cpu"]["cores_physical"] or "?"} core, '
                           f'{system["cpu"]["threads"]} thread' +
             (f' ({", ".join(system["cpu"]["flags"]).upper()})' if system["cpu"]["flags"] else "")),
            ("Memoria RAM", f'{_gb(system["ram_total"], 0)} (libera {_gb(system["ram_available"])})'),
            ("Spazio libero su disco", _gb(system.get("disk_free"), 0))]
    for i, g in enumerate(gpus, 1):
        parts = []
        if g["unified"]:
            parts.append("memoria condivisa con il sistema")
        elif g["vram_bytes"]:
            parts.append(f'{_gb(g["vram_bytes"])} VRAM')
        if g["bandwidth_gbs"]:
            parts.append(f'banda {g["bandwidth_gbs"]} GB/s ({g["bandwidth_source"]})')
        if g["compute_cap"]:
            parts.append(f'compute capability {g["compute_cap"]}')
        if g["driver"]:
            parts.append(f'driver {g["driver"]}')
        if g["cuda_version"]:
            parts.append(f'CUDA {g["cuda_version"]}')
        kind = "integrata" if g["integrated"] else "dedicata"
        rows.append((f"GPU {i} ({kind})", f'<b>{_e(g["name"])}</b><br><span class="muted">{_e(" · ".join(parts))}</span>'
                     + "".join(f'<br><span class="badge mid">{_e(n)}</span>' for n in g["notes"])))
    if not gpus:
        rows.append(("GPU", "Nessuna GPU rilevata"))
    body = "".join(f"<tr><th>{_e(k)}</th><td>{v if k.startswith('GPU') else _e(v)}</td></tr>" for k, v in rows)
    return f'<div class="card"><h2>Hardware rilevato</h2><table>{body}</table></div>'


def _tests(system, gpus, bench, ollama, prof):
    rows = []
    if bench and bench.get("bandwidth_gbs"):
        theo = prof["primary"]["bandwidth_gbs"] if prof["primary"] else None
        extra = f" — {bench['bandwidth_gbs'] / theo * 100:.0f}% del teorico ({theo} GB/s)" if theo else ""
        rows.append(("Banda memoria GPU", f'{bench["bandwidth_gbs"]} GB/s{extra}',
                     f'Misurata su {bench["adapter"]} ({bench["backend"]}). Determina la velocità di generazione.'))
        rows.append(("Calcolo GPU FP32", f'{bench["fp32_tflops"]} TFLOPS' if bench.get("fp32_tflops") else "—",
                     "Determina la velocità con cui vengono letti prompt e documenti lunghi."))
    elif bench and bench.get("skipped"):
        rows.append(("Benchmark GPU", "Non eseguito", "Disponibile solo un adattatore software."))
    else:
        rows.append(("Benchmark GPU", "Non eseguito", "Le prestazioni sono stimate dalle specifiche della GPU."))
    if system.get("ram_bandwidth_gbs"):
        rows.append(("Banda RAM", f'{system["ram_bandwidth_gbs"]} GB/s',
                     "Conta quando un modello non entra in VRAM e lavora in parte sulla CPU."))
    if ollama and ollama.get("gen_tps"):
        frac = ollama.get("gpu_fraction")
        where = "interamente in GPU" if frac and frac >= 0.99 else f"{int((frac or 0) * 100)}% in GPU"
        rows.append(("Test reale Ollama",
                     f'{ollama["gen_tps"]} token/s in generazione'
                     + (f', {ollama["prompt_tps"]:.0f} token/s in lettura' if ollama.get("prompt_tps") else ""),
                     f'Modello {ollama["model"]} ({ollama.get("parameter_size")}, {ollama.get("quantization")}), {where}. '
                     + ("Usato per calibrare tutte le stime." if prof["calibrated"] else "")))
    elif ollama and ollama.get("running"):
        rows.append(("Test reale Ollama", "Non eseguito", "Ollama è attivo ma senza un modello adatto al test."))
    else:
        rows.append(("Test reale Ollama", "Non eseguito",
                     "Ollama non è " + ("in esecuzione." if ollama and ollama.get("installed") else "installato.")))
    body = "".join(f"<tr><th>{_e(a)}</th><td><b>{_e(b)}</b><br><span class='muted'>{_e(c)}</span></td></tr>"
                   for a, b, c in rows)
    notes = "".join(f"<p class='muted'>{_e(n)}</p>" for n in prof["notes"])
    return f'<div class="card"><h2>Test eseguiti</h2><table>{body}</table>{notes}</div>'


def _verdict(result):
    prof, tier = result["profile"], result["tier"]
    meter = "".join(f'<span class="{"on" if i <= tier else ""}"></span>' for i in range(1, 6))
    best = best_model(result["models"])
    kpis = []
    if prof["primary"]:
        mem_label = "Memoria utilizzabile dalla GPU" if prof["unified"] else "VRAM utilizzabile"
        kpis.append((_gb(prof["usable_gpu_bytes"]), mem_label + (f' ({prof["gpu_count"]} GPU)' if prof["gpu_count"] > 1 else "")))
        if prof["gpu_bw"]:
            kpis.append((f'{prof["gpu_bw"]:.0f} GB/s', f'Banda effettiva ({prof["gpu_bw_source"]})'))
    else:
        kpis.append(("Nessuna", "GPU utilizzabile"))
    if best:
        kpis.append((best["model"]["name"], f'Modello migliore consigliato · {best["q4"]["tps"]:.0f} token/s'))
    conc = result["concurrency"]
    if conc:
        kpis.append((f'~{conc["sessions"]}', f'Conversazioni contemporanee da 8k token con {conc["model"]}'))
    kpi_html = "".join(f'<div class="kpi"><b>{_e(a)}</b><small>{_e(b)}</small></div>' for a, b in kpis)
    labels = "".join(f"<span>{x}</span>" for x in ("Base", "Iniziare", "Azienda", "Ottimo", "Server"))
    return (f'<div class="card"><div class="muted">Verdetto</div>'
            f'<div class="verdict-title">{_e(result["verdict"]["title"])}</div>'
            f'<div>{_e(result["verdict"]["text"])}</div>'
            f'<div class="meter">{meter}</div><div class="meter-labels">{labels}</div>'
            f'<div class="kpis">{kpi_html}</div></div>')


def _use_cases(result):
    cards = []
    for uc in result["use_cases"]:
        label, cls = RATING[uc["rating"]]
        if uc.get("model"):
            detail = f'Con {uc["model"]}: circa {uc["tps"]:.0f} token/s ({PLACEMENT_LABELS[uc["placement"]].lower()}).'
        else:
            detail = uc.get("note") or "Nessun modello adatto gira a una velocità sufficiente."
        cards.append(f'<div class="uc"><div class="top"><h3>{_e(uc["name"])}</h3>{_badge(label, cls)}</div>'
                     f'<p>{_e(uc["desc"])}</p><p><b>{_e(detail)}</b></p></div>')
    return f'<div class="card"><h2>Cosa puoi farci</h2><div class="grid">{"".join(cards)}</div></div>'


def _models(result):
    rows = []
    for e in result["models"]:
        m = e["model"]
        q4 = e["q4"]
        ctx = f'{q4["max_ctx"] // 1024}k' if q4.get("max_ctx") and q4["placement"] == "gpu" else "—"
        rows.append(f'<tr><td><b>{_e(m["name"])}</b><small class="muted"><br>{m["params_b"]:g}B parametri'
                    + (f', {m["active_b"]:g}B attivi' if m.get("active_b") else "")
                    + f' · ollama: {_e(m["ollama"])}</small></td>'
                    + "".join(_quant_cell(q) for q in e["quants"])
                    + f"<td>{ctx}</td></tr>")
    return (f'<div class="card"><h2>Modelli LLM: velocità stimata</h2>'
            f'<p class="muted">Token al secondo in generazione (un token ≈ ¾ di parola; si legge a circa 5-8 token/s). '
            f'Sopra i 15 token/s la risposta è fluida. Contesto massimo: lunghezza di conversazione/documenti che entra in VRAM.</p>'
            f'<div class="scroll"><table><tr><th>Modello</th><th>Q4 (consigliata)</th><th>Q8</th><th>FP16</th>'
            f'<th>Contesto max</th></tr>{"".join(rows)}</table></div></div>')


def _frameworks(result):
    rows = []
    for f in result["frameworks"]:
        label, cls = FW_STATUS[f["status"]]
        inst = "" if f["installed"] is None else ("Installato" if f["installed"] else "Non installato")
        rows.append(f'<tr><td><b>{_e(f["name"])}</b><br><small class="muted">{_e(f["desc"])}</small></td>'
                    f'<td>{_badge(label, cls)}</td><td>{_e(f["note"])}</td><td class="muted">{_e(inst)}</td></tr>')
    return (f'<div class="card"><h2>Framework e strumenti</h2><div class="scroll"><table>'
            f'<tr><th>Strumento</th><th>Compatibilità</th><th>Note</th><th></th></tr>{"".join(rows)}</table></div></div>')


def _recommendations(result):
    if not result["recommendations"]:
        return ""
    items = "".join(f"<li>{_e(r)}</li>" for r in result["recommendations"])
    return f'<div class="card"><h2>Consigli</h2><ul>{items}</ul></div>'


def render_html(system, gpus, bench, ollama, result):
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    prof = result["profile"]
    method = ("Le velocità sono stimate a partire dalla banda di memoria "
              + ("calibrata con il test reale di Ollama" if prof["calibrated"] else prof["gpu_bw_source"] or "della RAM")
              + f", con contesto di 8.192 token e quantizzazioni GGUF (Q4_K_M ≈ 4,85 bit per peso). "
                "I valori reali dipendono da driver, software e impostazioni: considerali indicativi (±25%). "
                f"Catalogo modelli aggiornato a {result['catalog_updated']}. Nessun dato è stato inviato in rete.")
    return f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Report idoneità AI</title><style>{CSS}</style></head>
<body><div class="wrap">
<header><h1>Report di idoneità all'AI locale</h1>
<div class="meta">{_e(BRAND)} Check v{__version__} · {now}</div></header>
{_verdict(result)}
{_use_cases(result)}
{_hardware(system, gpus)}
{_tests(system, gpus, bench, ollama, prof)}
{_models(result)}
{_frameworks(result)}
{_recommendations(result)}
<div class="card cta"><h2>Vuoi portare l'AI nella tua azienda?</h2>
<p>{_e(BRAND)} progetta e installa soluzioni AI on-premise: i dati restano nei tuoi server.
<a href="{CONTACT_URL}">Richiedi una valutazione gratuita</a> allegando questo report.</p></div>
<footer>{_e(method)}<br><a href="{BRAND_URL}">{BRAND_URL}</a></footer>
</div></body></html>"""


def render_text(system, gpus, bench, ollama, result):
    prof = result["profile"]
    lines = ["", "=" * 64, f"  VERDETTO: {result['verdict']['title'].upper()}  ({result['tier']}/5)",
             f"  {result['verdict']['text']}", "=" * 64]
    if prof["primary"]:
        lines.append(f"  GPU: {prof['primary']['name']} — {_gb(prof['usable_gpu_bytes'])} utilizzabili"
                     + (f", banda {prof['gpu_bw']:.0f} GB/s" if prof["gpu_bw"] else ""))
    else:
        lines.append("  GPU: nessuna GPU utilizzabile")
    if ollama and ollama.get("gen_tps"):
        lines.append(f"  Test reale: {ollama['model']} a {ollama['gen_tps']} token/s")
    lines.append("")
    lines.append("  Modelli (Q4):")
    for e in result["models"]:
        q = e["q4"]
        if q["placement"] == "no":
            continue
        lines.append(f"    {e['model']['name']:<30} {q['tps']:>6.0f} tok/s  {PLACEMENT_LABELS[q['placement']]}")
    lines.append("")
    lines.append("  Casi d'uso:")
    for uc in result["use_cases"]:
        lines.append(f"    {RATING[uc['rating']][0]:<12} {uc['name']}")
    return "\n".join(lines)
