import sys, importlib, pathlib
path_to_load = Path("build/src/bindings/python").resolve()
sys.path.append(str(path_to_load))

aeroforge_core = importlib.import_module("aeroforge_core")  
# from aeroforge_core import *

SimCore = aeroforge_core.SimCore
