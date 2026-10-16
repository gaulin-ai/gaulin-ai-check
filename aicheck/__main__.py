"""Gaulin AI Check: verifica se questo computer è adatto a ospitare AI e LLM in locale."""

import argparse
import datetime
import json
import os
import pathlib
import sys
import webbrowser

from . import __version__, analyze, bench_gpu, bench_ollama, gpu_detect, report, sysinfo


def log(msg):
    print(msg, flush=True)


def _output_dir(arg):
    if arg:
        return pathlib.Path(arg)
    # Accanto all'eseguibile se si può scrivere (doppio clic), altrimenti nella home
    candidates = [pathlib.Path.cwd(), pathlib.Path.home() / "Desktop", pathlib.Path.home()]
    for c in candidates:
        if c.is_dir() and os.access(c, os.W_OK):
            return c
    return pathlib.Path.home()


def _json_default(o):
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gaulin-ai-check",
                                 description="Verifica se il computer è adatto a far girare AI e LLM in locale.")
    ap.add_argument("--no-bench", action="store_true", help="salta il benchmark della GPU")
    ap.add_argument("--no-ollama", action="store_true", help="salta il test reale con Ollama")
    ap.add_argument("--ollama-model", help="modello Ollama da usare per il test (es. qwen3:8b)")
    ap.add_argument("--pull", action="store_true",
                    help="se in Ollama non c'è un modello adatto, scarica llama3.2:3b (~2 GB)")
    ap.add_argument("--out", help="cartella in cui salvare il report")
    ap.add_argument("--no-open", action="store_true", help="non aprire il report nel browser")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    log(f"Gaulin AI Check {__version__} — analisi del sistema per l'AI locale\n")
    log("[1/5] Raccolgo le informazioni di sistema...")
    system = sysinfo.collect()
    log("[2/5] Rilevo le GPU...")
    gpus = gpu_detect.detect()
    for g in gpus:
        log(f"  {g['name']}" + (f" — {g['vram_bytes'] / analyze.GB:.1f} GB" if g["vram_bytes"] and not g["unified"] else ""))
    if not gpus:
        log("  Nessuna GPU rilevata.")

    log("[3/5] Misuro la banda della RAM...")
    system["ram_bandwidth_gbs"] = sysinfo.measure_ram_bandwidth()
    if system["ram_bandwidth_gbs"]:
        log(f"  {system['ram_bandwidth_gbs']} GB/s")

    bench = None
    if args.no_bench:
        log("[4/5] Benchmark GPU saltato.")
    else:
        log("[4/5] Benchmark della GPU...")
        vram = next((g["vram_bytes"] for g in gpus if not g["integrated"] and g["vram_bytes"]), None)
        bench = bench_gpu.run(vram, log=log)
        if bench and bench.get("bandwidth_gbs"):
            log(f"  Banda {bench['bandwidth_gbs']} GB/s, calcolo {bench.get('fp32_tflops')} TFLOPS FP32")

    ollama = None
    if args.no_ollama:
        log("[5/5] Test Ollama saltato.")
    else:
        log("[5/5] Test reale con Ollama...")
        ollama = bench_ollama.run(args.ollama_model, args.pull, log=log)
        if ollama.get("gen_tps"):
            log(f"  {ollama['model']}: {ollama['gen_tps']} token/s")
        elif not ollama.get("installed") and not ollama.get("running"):
            log("  Ollama non installato: le velocità saranno stimate.")

    result = analyze.analyze(system, gpus, bench, ollama)
    log(report.render_text(system, gpus, bench, ollama, result))

    out_dir = _output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    html_path = out_dir / f"report-ai-{stamp}.html"
    json_path = out_dir / f"report-ai-{stamp}.json"
    html_path.write_text(report.render_html(system, gpus, bench, ollama, result), encoding="utf-8")
    json_path.write_text(json.dumps({"version": __version__, "system": system, "gpus": gpus, "gpu_benchmark": bench,
                                     "ollama": ollama, "analysis": result}, indent=2, ensure_ascii=False,
                                    default=_json_default), encoding="utf-8")
    log(f"\nReport salvato in:\n  {html_path.resolve()}\n  {json_path.resolve()}")
    if not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


def frozen_entry():
    """Avvio da eseguibile (doppio clic): tiene aperta la finestra alla fine."""
    try:
        code = main()
    except Exception as e:  # noqa: BLE001
        print(f"\nErrore imprevisto: {e}")
        code = 1
    if getattr(sys, "frozen", False) and sys.stdin and sys.stdin.isatty():
        try:
            input("\nPremi Invio per chiudere...")
        except EOFError:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
