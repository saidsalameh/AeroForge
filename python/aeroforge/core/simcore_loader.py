# python/aeroforge/core/simcore_loader.py

import sys
import importlib
from pathlib import Path

# This file lives in: python/aeroforge/core/simcore_loader.py
# So:
#   parents[0] -> /.../python/aeroforge/core
#   parents[1] -> /.../python/aeroforge
#   parents[2] -> /.../python
#   parents[3] -> /.../AeroForge   (project root)
ROOT = Path(__file__).resolve().parents[3]

BINDINGS_DIR = ROOT / "build" / "src" / "bindings" / "python"
sys.path.append(str(BINDINGS_DIR))

aeroforge_core = importlib.import_module("aeroforge_core")
SimCore = aeroforge_core.SimCore
