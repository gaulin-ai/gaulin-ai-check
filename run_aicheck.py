"""Punto di ingresso per PyInstaller."""
import sys

from aicheck.__main__ import frozen_entry

sys.exit(frozen_entry())
