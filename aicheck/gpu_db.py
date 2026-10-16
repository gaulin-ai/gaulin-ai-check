"""Banda di memoria teorica (GB/s) e VRAM (GB) delle GPU più diffuse.

Si usa quando il driver non permette di calcolarla (su NVIDIA la si ricava da NVML).
La banda è il dato che più conta per la velocità di generazione di un LLM.
Formato: (nome normalizzato, è_laptop, banda GB/s, VRAM GB)
"""

import re

GPUS = [
    # NVIDIA GeForce desktop
    ("rtx 5090", False, 1792, 32), ("rtx 5080", False, 960, 16), ("rtx 5070 ti", False, 896, 16),
    ("rtx 5070", False, 672, 12), ("rtx 5060 ti", False, 448, 16), ("rtx 5060", False, 448, 8),
    ("rtx 4090", False, 1008, 24), ("rtx 4080 super", False, 736, 16), ("rtx 4080", False, 717, 16),
    ("rtx 4070 ti super", False, 672, 16), ("rtx 4070 ti", False, 504, 12),
    ("rtx 4070 super", False, 504, 12), ("rtx 4070", False, 504, 12),
    ("rtx 4060 ti", False, 288, 8), ("rtx 4060", False, 272, 8),
    ("rtx 3090 ti", False, 1008, 24), ("rtx 3090", False, 936, 24), ("rtx 3080 ti", False, 912, 12),
    ("rtx 3080", False, 760, 10), ("rtx 3070 ti", False, 608, 8), ("rtx 3070", False, 448, 8),
    ("rtx 3060 ti", False, 448, 8), ("rtx 3060", False, 360, 12), ("rtx 3050", False, 224, 8),
    ("rtx 2080 ti", False, 616, 11), ("rtx 2080 super", False, 496, 8), ("rtx 2080", False, 448, 8),
    ("rtx 2070", False, 448, 8), ("rtx 2060", False, 336, 6),
    ("gtx 1080 ti", False, 484, 11), ("gtx 1080", False, 320, 8), ("gtx 1070", False, 256, 8),
    ("gtx 1660", False, 192, 6), ("gtx 1650", False, 128, 4),
    # NVIDIA GeForce laptop
    ("rtx 5090", True, 896, 24), ("rtx 5080", True, 896, 16), ("rtx 5070 ti", True, 672, 12),
    ("rtx 5070", True, 384, 8), ("rtx 5060", True, 384, 8),
    ("rtx 4090", True, 576, 16), ("rtx 4080", True, 432, 12), ("rtx 4070", True, 256, 8),
    ("rtx 4060", True, 256, 8), ("rtx 4050", True, 192, 6),
    ("rtx 3080 ti", True, 512, 16), ("rtx 3080", True, 448, 8), ("rtx 3070 ti", True, 448, 8),
    ("rtx 3070", True, 448, 8), ("rtx 3060", True, 336, 6), ("rtx 3050 ti", True, 192, 4),
    ("rtx 3050", True, 192, 4),
    # NVIDIA professionali / datacenter
    ("rtx pro 6000", False, 1792, 96), ("rtx 6000 ada", False, 960, 48), ("rtx 5000 ada", False, 576, 32),
    ("rtx 4500 ada", False, 432, 24), ("rtx 4000 ada", False, 360, 20), ("rtx a6000", False, 768, 48),
    ("rtx a5000", False, 768, 24), ("rtx a4500", False, 640, 20), ("rtx a4000", False, 448, 16),
    ("rtx a2000", False, 288, 12), ("quadro rtx 8000", False, 672, 48), ("quadro rtx 6000", False, 672, 24),
    ("b200", False, 8000, 180), ("h200", False, 4800, 141), ("h100 sxm", False, 3350, 80),
    ("h100", False, 2000, 80), ("a100 80gb", False, 1935, 80), ("a100", False, 1555, 40),
    ("l40s", False, 864, 48), ("l40", False, 864, 48), ("l4", False, 300, 24), ("a40", False, 696, 48),
    ("a30", False, 933, 24), ("a10", False, 600, 24), ("t4", False, 320, 16), ("v100", False, 900, 32),
    ("gb10", False, 273, 128),  # DGX Spark
    # AMD Radeon / Instinct
    ("rx 9070 xt", False, 640, 16), ("rx 9070", False, 640, 16), ("rx 9060 xt", False, 320, 16),
    ("rx 7900 xtx", False, 960, 24), ("rx 7900 xt", False, 800, 20), ("rx 7900 gre", False, 576, 16),
    ("rx 7800 xt", False, 624, 16), ("rx 7700 xt", False, 432, 12), ("rx 7600 xt", False, 288, 16),
    ("rx 7600", False, 288, 8), ("rx 6950 xt", False, 576, 16), ("rx 6900 xt", False, 512, 16),
    ("rx 6800 xt", False, 512, 16), ("rx 6800", False, 512, 16), ("rx 6750 xt", False, 432, 12),
    ("rx 6700 xt", False, 384, 12), ("rx 6650 xt", False, 280, 8), ("rx 6600 xt", False, 256, 8),
    ("rx 6600", False, 224, 8), ("pro w7900", False, 864, 48), ("pro w7800", False, 576, 32),
    ("mi325x", False, 6000, 256), ("mi300x", False, 5300, 192), ("mi250x", False, 3277, 128),
    ("mi210", False, 1638, 64),
    ("radeon 8060s", False, 256, None),  # Ryzen AI Max (Strix Halo), memoria unificata
    # Intel Arc
    ("arc b580", False, 456, 12), ("arc b570", False, 380, 10), ("arc a770", False, 560, 16),
    ("arc a750", False, 512, 8), ("arc a580", False, 512, 8), ("arc a380", False, 186, 6),
    # Apple Silicon (memoria unificata: la VRAM dipende dalla RAM installata)
    ("m4 max", False, 546, None), ("m4 pro", False, 273, None), ("m4", False, 120, None),
    ("m3 ultra", False, 819, None), ("m3 max", False, 400, None), ("m3 pro", False, 150, None),
    ("m3", False, 100, None), ("m2 ultra", False, 800, None), ("m2 max", False, 400, None),
    ("m2 pro", False, 200, None), ("m2", False, 100, None), ("m1 ultra", False, 800, None),
    ("m1 max", False, 400, None), ("m1 pro", False, 200, None), ("m1", False, 68, None),
]

# Varianti Apple con banda ridotta in base al numero di core GPU
APPLE_CORE_VARIANTS = {"m4 max": (32, 410), "m3 max": (30, 300)}


def normalize(name):
    n = name.lower()
    n = re.sub(r"\(r\)|\(tm\)|nvidia|geforce|amd|radeon(?= rx| pro)|apple|intel", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def is_laptop(name):
    n = name.lower()
    return "laptop" in n or "mobile" in n or bool(re.search(r"\bmax-q\b", n))


def lookup(name, gpu_cores=None):
    """Cerca la GPU per nome. Restituisce {'bandwidth_gbs', 'vram_gb'} oppure None."""
    n = normalize(name)
    laptop = is_laptop(name)
    by_length = sorted(GPUS, key=lambda g: -len(g[0]))
    # Prima si cerca la variante giusta (desktop/laptop), poi qualunque variante
    for strict in (True, False):
        for key, lap, bw, vram in by_length:
            if strict and lap != laptop:
                continue
            if re.search(r"(^|[\s-])" + re.escape(key) + r"($|[\s-])", n):
                if key in APPLE_CORE_VARIANTS and gpu_cores:
                    max_cores, reduced_bw = APPLE_CORE_VARIANTS[key]
                    if gpu_cores <= max_cores:
                        bw = reduced_bw
                return {"bandwidth_gbs": bw, "vram_gb": vram, "match": key, "exact": strict}
    return None
