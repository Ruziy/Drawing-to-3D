"""Versioned CV/OCR cache; original bytes/pixels are never modified."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from .config import ROOT,DPI
from .storage import atomic_json,read_json

FILES=('drawing_analyzer.py','pipeline_cache.py','image_geometry.py','layout.py','ocr.py','pdf_reader.py',
       'dimension_parser.py','radius_ocr.py','annotation_context.py','non_geometry.py','contour_graph.py','schemas.py')


def cache_path(pdf_hash):
    versions={p:hashlib.sha256((ROOT/'app'/p).read_bytes()).hexdigest() for p in FILES}
    weights={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'.runtime/tessdata').glob('*.traineddata')}
    key=hashlib.sha256(json.dumps({'source':pdf_hash,'code':versions,'ocr_weights':weights,'dpi':DPI},sort_keys=True).encode()).hexdigest()
    return ROOT/'data/cache/cv'/key


def enabled():return os.environ.get('DRAWING_CACHE','true').lower()=='true'


def load(pdf_hash,output,drawing_id,filename):
    if not enabled():return None
    cache=cache_path(pdf_hash)
    if not (cache/'complete.json').is_file():return None
    info=read_json(cache/'complete.json')
    for rel,digest in info['files'].items():
        p=cache/rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:return None
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for folder in cache.glob('page-*'):shutil.copytree(folder,output/folder.name,dirs_exist_ok=True)
    value=read_json(cache/'cv-analysis.json');old_id=value['drawing_id'];old_output=info['original_output']
    def rewrite(x):
        if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
        if isinstance(x,list):return [rewrite(v) for v in x]
        if isinstance(x,str):return x.replace(old_id,drawing_id).replace(old_output,str(output.resolve()))
        return x
    value=rewrite(value);value['filename']=filename;value['metadata']['cv_cache_hit']=True
    return value


def store(value,output):
    if not enabled():return
    cache=cache_path(value['sha256']);cache.mkdir(parents=True,exist_ok=True)
    if (cache/'complete.json').is_file():return
    for folder in Path(output).glob('page-*'):shutil.copytree(folder,cache/folder.name,dirs_exist_ok=True)
    atomic_json(cache/'cv-analysis.json',value)
    files={p.relative_to(cache).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.rglob('*') if p.is_file() and p.name!='complete.json'}
    atomic_json(cache/'complete.json',{'files':files,'original_output':str(Path(output).resolve())})
