"""Informazioni di sistema: sistema operativo, CPU, RAM, disco, software AI installato."""

import ctypes
import os
import platform
import shutil
import subprocess
import time

try:
    import psutil
except ImportError:  # psutil è opzionale
    psutil = None

GB = 1024 ** 3


def run_cmd(args, timeout=15):
    """Esegue un comando e restituisce stdout, oppure None se fallisce o non esiste."""
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kwargs)
        if out.returncode != 0:
            return None
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_name():
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        out = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    elif system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "Sconosciuta"


def _cpu_flags():
    """Estensioni vettoriali utili all'inferenza su CPU (solo Linux le espone in modo affidabile)."""
    if platform.system() != "Linux":
        return []
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    return [x for x in ("avx2", "avx512f", "avx512_bf16", "amx_tile") if x in flags]
    except OSError:
        pass
    return []


def _ram():
    """Restituisce (totale, disponibile) in byte."""
    if psutil:
        vm = psutil.virtual_memory()
        return vm.total, vm.available
    system = platform.system()
    if system == "Linux":
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return info.get("MemTotal", 0), info.get("MemAvailable", 0)
    if system == "Darwin":
        out = run_cmd(["sysctl", "-n", "hw.memsize"])
        total = int(out) if out else 0
        return total, total // 2
    if system == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys, stat.ullAvailPhys
    return 0, 0


def _os_name():
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if system == "Windows":
        return f"Windows {platform.release()} ({platform.version()})"
    if system == "Linux":
        try:
            with open("/etc/os-release", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return f"Linux {platform.release()}"
    return system


def _software():
    """Strumenti AI e di containerizzazione già presenti."""
    found = {}
    for name, cmds in {
        "ollama": ["ollama"],
        "docker": ["docker"],
        "nvidia_container_toolkit": ["nvidia-ctk", "nvidia-container-cli"],
        "lm_studio": ["lms"],
        "cuda_toolkit": ["nvcc"],
        "rocm": ["rocminfo", "amd-smi"],
        "wsl": ["wsl"] if platform.system() == "Windows" else [],
    }.items():
        found[name] = any(shutil.which(c) for c in cmds)
    return found


def measure_ram_bandwidth(log=None):
    """Misura grossolana della banda RAM con copie multi-thread (serve numpy).

    Restituisce GB/s oppure None. Sottostima un po' i server multi-canale, ma è
    molto più realistica di un valore fisso.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    from concurrent.futures import ThreadPoolExecutor

    threads = max(2, min(16, os.cpu_count() or 4))
    n = 64 * 1024 * 1024  # 256 MB per buffer (float32)
    try:
        src = np.ones(n, dtype=np.float32)
        dst = np.empty_like(src)
    except MemoryError:
        return None
    chunks = [(i * n // threads, (i + 1) * n // threads) for i in range(threads)]

    def copy(chunk):
        a, b = chunk
        np.copyto(dst[a:b], src[a:b])

    best = 0.0
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(copy, chunks))  # riscaldamento
        for _ in range(5):
            t0 = time.perf_counter()
            list(pool.map(copy, chunks))
            dt = time.perf_counter() - t0
            best = max(best, 2 * src.nbytes / dt / 1e9)  # lettura + scrittura
    return round(best, 1)


def collect():
    total, avail = _ram()
    home = os.path.expanduser("~")
    try:
        disk = shutil.disk_usage(home)
        disk_free = disk.free
    except OSError:
        disk_free = None
    physical = psutil.cpu_count(logical=False) if psutil else None
    return {
        "os": _os_name(),
        "os_family": platform.system(),  # Windows / Linux / Darwin
        "arch": platform.machine(),
        "cpu": {
            "name": _cpu_name(),
            "cores_physical": physical,
            "threads": os.cpu_count(),
            "flags": _cpu_flags(),
        },
        "ram_total": total,
        "ram_available": avail,
        "disk_free": disk_free,
        "software": _software(),
    }
