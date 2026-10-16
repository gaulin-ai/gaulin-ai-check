"""Rilevamento delle GPU su Windows, Linux e macOS (NVIDIA, AMD, Intel, Apple)."""

import ctypes
import glob
import json
import os
import platform
import re

from . import gpu_db
from .sysinfo import run_cmd

MB = 1024 ** 2
GB = 1024 ** 3

VENDOR_IDS = {"10de": "nvidia", "1002": "amd", "8086": "intel", "106b": "apple"}


def _gpu(name, vendor, **kw):
    g = {
        "name": name.strip(), "vendor": vendor, "vram_bytes": None, "vram_used_bytes": None,
        "integrated": False, "unified": False, "driver": None, "cuda_version": None,
        "compute_cap": None, "bandwidth_gbs": None, "bandwidth_source": None,
        "gpu_cores": None, "notes": [],
    }
    g.update(kw)
    return g


# ---------------------------------------------------------------- NVIDIA

def _nvml_memory_specs():
    """Larghezza del bus e clock massimo della memoria via NVML, per ogni GPU NVIDIA.

    Banda teorica = clock_MHz * 2 * bus_bit / 8 / 1000 GB/s (vale per GDDR6/6X/7 e HBM).
    """
    candidates = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    if os.name == "nt":
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        progfiles = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates = [os.path.join(sysroot, "System32", "nvml.dll"),
                      os.path.join(progfiles, "NVIDIA Corporation", "NVSMI", "nvml.dll"), "nvml.dll"]
    lib = None
    for c in candidates:
        try:
            lib = ctypes.CDLL(c)
            break
        except OSError:
            continue
    if lib is None:
        return []
    specs = []
    try:
        if lib.nvmlInit_v2() != 0:
            return []
        count = ctypes.c_uint()
        lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
        for i in range(count.value):
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)) != 0:
                specs.append(None)
                continue
            width, clock = ctypes.c_uint(), ctypes.c_uint()
            ok_w = lib.nvmlDeviceGetMemoryBusWidth(handle, ctypes.byref(width)) == 0
            ok_c = lib.nvmlDeviceGetMaxClockInfo(handle, 2, ctypes.byref(clock)) == 0  # NVML_CLOCK_MEM
            if ok_w and ok_c and width.value and clock.value:
                specs.append({"bus_width": width.value, "mem_clock_mhz": clock.value,
                              "bandwidth_gbs": round(clock.value * 2 * width.value / 8 / 1000)})
            else:
                specs.append(None)
        lib.nvmlShutdown()
    except (AttributeError, OSError):
        return specs
    return specs


def _detect_nvidia():
    fields = "name,memory.total,memory.used,driver_version,compute_cap"
    out = run_cmd(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    has_cc = out is not None
    if out is None:  # driver vecchi non conoscono compute_cap
        out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                       "--format=csv,noheader,nounits"])
    if not out:
        return []
    header = run_cmd(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", header)
    cuda = m.group(1) if m else None
    specs = _nvml_memory_specs()
    gpus = []
    for i, line in enumerate(l for l in out.strip().splitlines() if l.strip()):
        parts = [p.strip() for p in line.split(",")]
        g = _gpu(parts[0], "nvidia", driver=parts[3], cuda_version=cuda)
        try:
            g["vram_bytes"] = int(float(parts[1])) * MB
            g["vram_used_bytes"] = int(float(parts[2])) * MB
        except ValueError:
            pass
        if has_cc and len(parts) > 4:
            try:
                g["compute_cap"] = float(parts[4])
            except ValueError:
                pass
        spec = specs[i] if i < len(specs) else None
        if spec:
            g["bandwidth_gbs"] = spec["bandwidth_gbs"]
            g["bandwidth_source"] = f"driver (bus {spec['bus_width']} bit, {spec['mem_clock_mhz']} MHz)"
        gpus.append(g)
    return gpus


# ---------------------------------------------------------------- Linux

def _lspci_name(slot):
    out = run_cmd(["lspci", "-mm", "-s", slot])
    if not out:
        return None
    fields = re.findall(r'"([^"]*)"', out)
    # "classe" "vendor" "dispositivo" ...
    if len(fields) >= 3:
        name = fields[2]
        m = re.search(r"\[(.+)\]", name)  # es. "Navi 31 [Radeon RX 7900 XTX]"
        return m.group(1) if m else name
    return None


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _detect_linux_sysfs(skip_nvidia):
    gpus, seen = [], set()
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        real = os.path.realpath(dev)
        if real in seen:
            continue
        seen.add(real)
        vendor = VENDOR_IDS.get((_read(f"{dev}/vendor") or "").replace("0x", ""))
        cls = _read(f"{dev}/class") or ""
        if not vendor or not cls.startswith("0x03"):  # 0x03xxxx = controller video
            continue
        if vendor == "nvidia" and skip_nvidia:
            continue
        uevent = _read(f"{dev}/uevent") or ""
        m = re.search(r"PCI_SLOT_NAME=(\S+)", uevent)
        slot = m.group(1) if m else None
        driver = os.path.basename(os.path.realpath(f"{dev}/driver")) if os.path.exists(f"{dev}/driver") else None
        name = (_lspci_name(slot) if slot else None) or f"GPU {vendor.upper()}"
        g = _gpu(name, vendor, driver=driver)
        vram = _read(f"{dev}/mem_info_vram_total")  # driver amdgpu
        if vram and vram.isdigit():
            g["vram_bytes"] = int(vram)
            used = _read(f"{dev}/mem_info_vram_used")
            if used and used.isdigit():
                g["vram_used_bytes"] = int(used)
        if vendor == "intel":
            g["integrated"] = not re.search(r"arc(\(tm\))? [ab]\d", name.lower())
        if vendor == "amd":
            # Le APU espongono solo una piccola porzione riservata come VRAM
            g["integrated"] = (g["vram_bytes"] or 0) <= 2 * GB or not re.search(r"\b(rx|pro|instinct|mi\d)", name.lower())
        if vendor == "nvidia":
            g["notes"].append("Driver NVIDIA proprietario non attivo: senza di esso la GPU non è usabile per l'AI.")
        if g["integrated"]:
            g["unified"] = True
        gpus.append(g)
    return gpus


# ---------------------------------------------------------------- Windows

def _detect_windows(skip_nvidia):
    try:
        import winreg
    except ImportError:
        return []
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return []
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(root, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        try:
            key = winreg.OpenKey(root, sub)
        except OSError:
            continue

        def val(name):
            try:
                return winreg.QueryValueEx(key, name)[0]
            except OSError:
                return None

        name = val("DriverDesc") or ""
        dev_id = (val("MatchingDeviceId") or "").lower()
        m = re.search(r"ven_([0-9a-f]{4})", dev_id)
        vendor = VENDOR_IDS.get(m.group(1)) if m else None
        if not vendor or "basic" in name.lower() or "remote" in name.lower():
            continue
        if vendor == "nvidia" and skip_nvidia:
            continue
        g = _gpu(name, vendor, driver=val("DriverVersion"))
        mem = val("HardwareInformation.qwMemorySize")
        if mem is None:
            raw = val("HardwareInformation.MemorySize")
            if isinstance(raw, bytes):
                mem = int.from_bytes(raw[:8], "little")
            elif isinstance(raw, int):
                mem = raw
        if mem:
            g["vram_bytes"] = int(mem)
        low = name.lower()
        if vendor == "intel":
            g["integrated"] = not re.search(r"arc(\(tm\))? [ab]\d", low)
        if vendor == "amd":
            g["integrated"] = not re.search(r"\b(rx|pro w|instinct)\b", low)
        g["unified"] = g["integrated"]
        if not any(x["name"] == g["name"] for x in gpus):
            gpus.append(g)
    return gpus


# ---------------------------------------------------------------- macOS

def _detect_macos():
    out = run_cmd(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30)
    if not out:
        return []
    try:
        items = json.loads(out).get("SPDisplaysDataType", [])
    except json.JSONDecodeError:
        return []
    chip = (run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip()
    gpus = []
    for it in items:
        name = it.get("sppci_model") or it.get("_name") or "GPU"
        vendor_raw = (it.get("spdisplays_vendor") or "").lower()
        if "apple" in vendor_raw or name.lower().startswith("apple"):
            if chip.lower().startswith("apple"):
                name = chip  # es. "Apple M2 Pro"
            cores = it.get("sppci_cores")
            g = _gpu(name, "apple", integrated=True, unified=True, driver="Metal",
                     gpu_cores=int(cores) if str(cores).isdigit() else None)
        else:
            vendor = "amd" if "amd" in vendor_raw or "ati" in vendor_raw else \
                     "nvidia" if "nvidia" in vendor_raw else "intel" if "intel" in vendor_raw else "other"
            g = _gpu(name, vendor, driver="Metal")
            vram = it.get("spdisplays_vram") or it.get("_spdisplays_vram") or ""
            m = re.match(r"(\d+)\s*(GB|MB)", vram)
            if m:
                g["vram_bytes"] = int(m.group(1)) * (GB if m.group(2) == "GB" else MB)
            g["integrated"] = vendor == "intel"
            g["unified"] = g["integrated"]
        gpus.append(g)
    return gpus


# ---------------------------------------------------------------- pubblico

def detect():
    system = platform.system()
    gpus = []
    if system == "Darwin":
        gpus = _detect_macos()
    else:
        nvidia = _detect_nvidia()
        gpus.extend(nvidia)
        if system == "Linux":
            gpus.extend(_detect_linux_sysfs(skip_nvidia=bool(nvidia)))
        elif system == "Windows":
            gpus.extend(_detect_windows(skip_nvidia=bool(nvidia)))

    for g in gpus:
        info = gpu_db.lookup(g["name"], g.get("gpu_cores"))
        if info:
            if not g["bandwidth_gbs"]:
                g["bandwidth_gbs"] = info["bandwidth_gbs"]
                g["bandwidth_source"] = "scheda tecnica" + ("" if info["exact"] else " (variante simile)")
            if not g["vram_bytes"] and info["vram_gb"] and not g["unified"]:
                g["vram_bytes"] = info["vram_gb"] * GB
        if g["vendor"] == "nvidia" and g["compute_cap"] is None and g["driver"] and "nouveau" not in str(g["driver"]):
            g["notes"].append("Compute capability non rilevata (driver datato?).")

    # GPU dedicate prima, poi per VRAM decrescente
    gpus.sort(key=lambda g: (g["integrated"], -(g["vram_bytes"] or 0)))
    return gpus
