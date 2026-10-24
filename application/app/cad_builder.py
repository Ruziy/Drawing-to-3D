"""Run FreeCAD in its own compatible Python process, never in FastAPI's ABI."""
import os
from pathlib import Path
import shutil

from .config import ROOT


def find_freecad():
    configured = os.environ.get("FREECAD_PYTHON") or os.environ.get("FREECAD_CMD")
    if configured:
        path = Path(configured)
        return path.resolve() if path.is_file() else None
    bases = [ROOT/".runtime",Path(os.environ.get("ProgramFiles","C:/Program Files")),
             Path(os.environ.get("LOCALAPPDATA","C:/Users/Default/AppData/Local"))/"Programs"]
    for base in bases:
        for pattern in ("FreeCAD*/bin/python.exe","FreeCAD*/bin/FreeCADCmd.exe","FreeCAD*/bin/freecadcmd.exe"):
            hits = sorted(base.glob(pattern)) if base.exists() else []
            if hits:
                return hits[-1].resolve()
    for name in ("FreeCADCmd","freecadcmd"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def cad_command(intent_path: Path, output_dir: Path):
    exe = find_freecad()
    if not exe:
        raise RuntimeError("FreeCAD не найден. Укажите FREECAD_PYTHON (bin/python.exe) или FREECAD_CMD (FreeCADCmd.exe).")
    env = os.environ.copy()
    env.update(DRAWING_CAD_INTENT=str(intent_path.resolve()), DRAWING_CAD_OUTPUT=str(output_dir.resolve()),
               PYTHONIOENCODING="utf-8",PYTHONUTF8="1")
    env.pop("PYTHONHOME",None)
    env.pop("PYTHONPATH",None)
    env["PATH"] = str(exe.parent)+os.pathsep+str(exe.parent/"Library"/"bin")+os.pathsep+env.get("PATH","")
    return [str(exe),str(ROOT/"app"/"freecad_worker.py")],env
