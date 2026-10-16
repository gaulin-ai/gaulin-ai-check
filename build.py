"""Crea l'eseguibile per il sistema corrente: python build.py → dist/gaulin-ai-check[.exe]"""
import os
import platform

import PyInstaller.__main__

suffix = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}[platform.system()]
PyInstaller.__main__.run([
    "run_aicheck.py",
    "--onefile",
    "--console",
    f"--name=gaulin-ai-check-{suffix}-{platform.machine().lower()}",
    f"--add-data=aicheck/data{os.pathsep}aicheck/data",
    "--collect-all=wgpu",
    "--noconfirm",
    "--clean",
])
