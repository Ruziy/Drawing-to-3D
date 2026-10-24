import json
import os
import re
from pathlib import Path
from uuid import uuid4

from .config import DATA


def filesystem_path(path):
    path=Path(path)
    if os.name=='nt':
        name=str(path.absolute())
        if not name.startswith('\\\\?\\'):
            name='\\\\?\\UNC\\'+name[2:] if name.startswith('\\\\') else '\\\\?\\'+name
        return Path(name)
    return path


def atomic_json(path: Path, value):
    path=filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / ('.'+uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(filesystem_path(path).read_text(encoding="utf-8"))


def upload_dir(drawing_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", drawing_id):
        raise ValueError("Некорректный ID чертежа")
    return DATA / "uploads" / drawing_id
