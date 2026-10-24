"""Content-addressed immutable candidate history; no drawing-number lookup."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from .config import ROOT
from .storage import atomic_json, read_json


class BestCandidateRegistry:
    def __init__(self, root=None):
        self.root=Path(root or os.environ.get('DRAWING_REGISTRY_DIR',ROOT/'data/best-candidates'))

    def candidates(self, pdf_hash):
        if len(pdf_hash)!=64 or any(c not in '0123456789abcdef' for c in pdf_hash):return []
        return sorted(({**read_json(p),'build':str(p.parent.resolve()),'intent':str((p.parent/'intent.json').resolve())} for p in (self.root/pdf_hash).glob('*/candidate.json')),
                      key=lambda r:(r['quality'] in ('A','B'),r.get('score',0)),reverse=True)

    @staticmethod
    def algorithm_hash():
        return hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'app').glob('*.py')))).hexdigest()

    def store(self, pdf_hash, intent, build, quality, model, provenance):
        build=Path(build)
        if quality not in ('A','B','C') or not (build/'validation.json').is_file():return None
        validation=read_json(build/'validation.json')
        if validation.get('overall_status')=='failed':return None
        def fingerprint(x):
            if isinstance(x,dict):return {k:fingerprint(v) for k,v in x.items() if k!='drawing_id'}
            if isinstance(x,list):return [fingerprint(v) for v in x]
            if isinstance(x,str) and len(x)<1000:
                try:
                    path=Path(x)
                    if path.is_absolute() and path.is_file():return {'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                except OSError:pass
            return x
        identity=hashlib.sha256((json.dumps(fingerprint(intent),sort_keys=True)+quality+self.algorithm_hash()).encode()).hexdigest()[:20]
        target=self.root/pdf_hash/identity
        if (target/'candidate.json').is_file():return read_json(target/'candidate.json')
        target.mkdir(parents=True,exist_ok=True)
        # Preserve every referenced source image as well as the actual exports.
        def relocate(value):
            if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
            if isinstance(value,list):return [relocate(v) for v in value]
            if isinstance(value,str) and len(value)<1000:
                try:
                    path=Path(value)
                    if path.is_absolute() and path.is_file():
                        name=hashlib.sha256(str(path).encode()).hexdigest()[:12]+'-'+path.name
                        dest=target/'evidence'/name;dest.parent.mkdir(exist_ok=True)
                        shutil.copy2(path,dest);return str(dest.resolve())
                except OSError:pass
            return value
        saved=relocate(intent)
        atomic_json(target/'intent.json',saved)
        for p in build.iterdir():
            if p.name!='intent.json' and p.is_file() and p.suffix.lower() in ('.json','.step','.fcstd','.stl','.png','.log'):
                shutil.copy2(p,target/p.name)
        files={p.relative_to(target).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file()}
        record={'pdf_sha256':pdf_hash,'quality':quality,'model':model,'provenance':provenance,
                'algorithm_sha256':self.algorithm_hash(),
                'intent':str((target/'intent.json').resolve()),'build':str(target.resolve()),
                'score':sum(c.get('status')=='passed' for c in validation.get('dimensions',[])),
                'files_sha256':files,'validation':validation}
        atomic_json(target/'candidate.json',record);return record

    def load_intent(self,record):
        root=Path(record['build'])
        def rebase(x):
            if isinstance(x,dict):return {k:rebase(v) for k,v in x.items()}
            if isinstance(x,list):return [rebase(v) for v in x]
            if isinstance(x,str) and len(x)<1000:
                try:
                    replacement=root/'evidence'/Path(x).name
                    if replacement.is_file():return str(replacement)
                except OSError:pass
            return x
        return rebase(read_json(root/'intent.json'))

    def intact(self, record):
        root=Path(record['build'])
        return all((root/k).is_file() and hashlib.sha256((root/k).read_bytes()).hexdigest()==v
                   for k,v in record['files_sha256'].items())
