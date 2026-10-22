"""Test dell'analisi su configurazioni hardware tipiche (nessuna GPU reale necessaria)."""

from aicheck import analyze, bench_ollama, gpu_db
from aicheck.analyze import GB


def system(ram_gb, os_family="Linux", cpu="AMD Ryzen 9 7950X", ram_bw=60.0):
    return {"os": os_family, "os_family": os_family, "arch": "x86_64",
            "cpu": {"name": cpu, "cores_physical": 16, "threads": 32, "flags": ["avx2"]},
            "ram_total": ram_gb * GB, "ram_available": ram_gb * GB // 2, "disk_free": 500 * GB,
            "software": {"ollama": True, "docker": True}, "ram_bandwidth_gbs": ram_bw}


def gpu(name, vendor, vram_gb, bw, cc=None, unified=False):
    return {"name": name, "vendor": vendor, "vram_bytes": int(vram_gb * GB) if vram_gb else None,
            "vram_used_bytes": 0, "integrated": unified, "unified": unified, "driver": "x",
            "cuda_version": "12.8" if vendor == "nvidia" else None, "compute_cap": cc,
            "bandwidth_gbs": bw, "bandwidth_source": "test", "gpu_cores": None, "notes": []}


def by_id(result, model_id):
    return next(e for e in result["models"] if e["model"]["id"] == model_id)


def test_rtx4090_runs_32b_in_gpu():
    r = analyze.analyze(system(64), [gpu("NVIDIA GeForce RTX 4090", "nvidia", 24, 1008, 8.9)])
    q = by_id(r, "qwen3-32b")["q4"]
    assert q["placement"] == "gpu"
    assert 20 < q["tps"] < 60
    assert r["tier"] == 4
    assert by_id(r, "llama3.3-70b")["q4"]["placement"] == "split"


def test_dual_a100_is_server_class():
    r = analyze.analyze(system(256), [gpu("NVIDIA A100 80GB PCIe", "nvidia", 80, 1935, 8.0)] * 2)
    assert r["profile"]["gpu_count"] == 2
    assert by_id(r, "llama3.3-70b")["q4"]["placement"] == "gpu"
    assert r["tier"] == 5


def test_mac_unified_memory_never_splits():
    r = analyze.analyze(system(64, "Darwin", "Apple M2 Max"),
                        [gpu("Apple M2 Max", "apple", None, 400, unified=True)])
    assert abs(r["profile"]["usable_gpu_bytes"] / GB - 64 * 0.75) < 0.1
    placements = {e["q4"]["placement"] for e in r["models"]}
    assert "split" not in placements
    assert by_id(r, "qwen3-32b")["q4"]["placement"] == "gpu"
    mlx = next(f for f in r["frameworks"] if f["name"] == "MLX")
    assert mlx["status"] == "ok"


def test_cpu_only_is_basic_at_most():
    r = analyze.analyze(system(16, "Windows", "Intel Core i5-1235U", ram_bw=40), [])
    assert r["profile"]["primary"] is None
    assert r["tier"] <= 1
    assert all(e["q4"]["placement"] in ("cpu", "no") for e in r["models"])


def test_small_laptop_gpu_is_not_overrated():
    r = analyze.analyze(system(32), [gpu("NVIDIA GeForce RTX 3050 Laptop GPU", "nvidia", 4, 176, 8.6)])
    assert r["tier"] <= 2
    rag = next(u for u in r["use_cases"] if u["id"] == "rag")
    assert rag["rating"] in ("no", "limitato", "sufficiente")


def test_ollama_calibration_sets_overhead():
    sys_ = system(32)
    g = [gpu("NVIDIA GeForce RTX 3050 Laptop GPU", "nvidia", 4, 176, 8.6)]
    ollama = {"gen_tps": 50.0, "model_size_bytes": 2_000_000_000, "effective_bandwidth_gbs": 100.0,
              "gpu_fraction": 1.0, "prompt_tps": 2000, "parameter_size": "3.2B"}
    prof = analyze.compute_profile(sys_, g, None, ollama)
    assert prof["calibrated"]
    assert 0.004 < prof["gpu_overhead_s"] < 0.01


def test_gpu_db_variants():
    assert gpu_db.lookup("NVIDIA GeForce RTX 4070 Ti SUPER")["bandwidth_gbs"] == 672
    assert gpu_db.lookup("NVIDIA GeForce RTX 4070 Ti")["bandwidth_gbs"] == 504
    assert gpu_db.lookup("NVIDIA GeForce RTX 4070 Laptop GPU")["bandwidth_gbs"] == 256
    assert gpu_db.lookup("AMD Radeon RX 7900 XTX")["bandwidth_gbs"] == 960
    assert gpu_db.lookup("Intel(R) Arc(TM) A770 Graphics")["bandwidth_gbs"] == 560
    assert gpu_db.lookup("Apple M3 Max", gpu_cores=30)["bandwidth_gbs"] == 300
    assert gpu_db.lookup("Apple M3 Max", gpu_cores=40)["bandwidth_gbs"] == 400
    assert gpu_db.lookup("NVIDIA A100-SXM4-40GB")["bandwidth_gbs"] == 1555
    assert gpu_db.lookup("Scheda sconosciuta") is None


def test_gemma4_on_24gb_gpu():
    r = analyze.analyze(system(64), [gpu("NVIDIA GeForce RTX 4090", "nvidia", 24, 1008, 8.9)])
    dense, moe = by_id(r, "gemma4-31b")["q4"], by_id(r, "gemma4-26b-a4b")["q4"]
    assert dense["placement"] == moe["placement"] == "gpu"
    assert moe["tps"] > 2 * dense["tps"]  # il MoE legge solo 3,8 miliardi di parametri per token
    assert by_id(r, "gemma4-12b")["q4"]["max_ctx"] == 262144


def test_calibration_skips_gemma_with_per_layer_embeddings():
    ok = bench_ollama._reads_all_weights
    assert ok("llama3.2:3b", "llama") and ok("gemma4:12b", "gemma4") and ok("gemma4:31b-it-qat", "gemma4")
    assert not ok("gemma4:latest", "gemma4") and not ok("gemma4:e2b", "gemma4")
    assert not ok("gemma4:26b", "gemma4") and not ok("gemma3n:e4b", "gemma3n")
    assert not ok("qwen3:30b", "qwen3moe")
