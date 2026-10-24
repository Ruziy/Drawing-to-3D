import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# A deliberately small .env reader: no interpolation, commands or external dependency.
if (ROOT / '.env').is_file():
    for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            if key.strip().startswith(('LLM_', 'DRAWING_', 'FREECAD_')):
                os.environ.setdefault(key.strip(), value.strip().strip('\"\''))
DATA = Path(os.environ.get("DRAWING_DATA_DIR", ROOT / "data")).resolve()
MAX_UPLOAD = 25 * 1024 * 1024
MAX_PAGES = 20
MAX_PAGE_PIXELS = 96_000_000
MAX_DOCUMENT_PIXELS = 220_000_000
DPI = 300


def ensure_dirs():
    for name in ("uploads", "analysis", "models"):
        (DATA / name).mkdir(parents=True, exist_ok=True)
