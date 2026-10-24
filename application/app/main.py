import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .cad_builder import cad_command, find_freecad
from .config import DATA, MAX_UPLOAD, ROOT, ensure_dirs
from .schemas import CADIntent
from .storage import atomic_json, read_json, upload_dir
from .ocr import available as ocr_available
from .validation import validate_model
from .ollama_client import LLMSettings, llm_status

tasks = set()
analysis_slots = asyncio.Semaphore(1)
cad_slots = asyncio.Semaphore(1)


def stamp():
    return datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def lifespan(app):
    ensure_dirs()
    for path in (DATA/"uploads").glob("*/status.json"):
        state = read_json(path)
        changed = False
        for kind in ("analysis","cad"):
            if state[kind]["state"] in ("queued","running"):
                state[kind] = {"state":"failed","error":"Сервер был перезапущен. Повторите операцию."}
                changed = True
        if changed:
            atomic_json(path,state)
    yield
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*list(tasks),return_exceptions=True)


app = FastAPI(title="Drawing-to-3D",version="1.0.0",lifespan=lifespan)
app.mount("/static",StaticFiles(directory=ROOT/"static"),name="static")


@app.exception_handler(StarletteHTTPException)
async def http_error(request,exc):
    return JSONResponse({"detail":exc.detail},status_code=exc.status_code)


@app.exception_handler(Exception)
async def server_error(request,exc):
    import logging
    logging.exception("Request failed")
    return JSONResponse({"detail":"Внутренняя ошибка. Подробности в журнале сервера."},status_code=500)


@app.middleware("http")
async def limits_and_local_origin(request: Request,call_next):
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length)>MAX_UPLOAD+1024*1024):
        return JSONResponse({"detail":"Лимит загрузки: 25 МБ"},status_code=413)
    origin = request.headers.get("origin")
    if request.method not in ("GET","HEAD","OPTIONS") and origin and origin != str(request.base_url).rstrip("/"):
        return JSONResponse({"detail":"Запросы разрешены только со страницы локального приложения"},status_code=403)
    return await call_next(request)


def get_dir(drawing_id):
    try:
        directory = upload_dir(drawing_id)
    except ValueError:
        raise HTTPException(404,"Чертёж не найден")
    if not (directory/"metadata.json").is_file():
        raise HTTPException(404,"Чертёж не найден")
    return directory


def update_state(directory,kind,**values):
    path = directory/"status.json"
    state = read_json(path)
    state[kind] = {**values,"updated_at":stamp()}
    atomic_json(path,state)


async def process(command,logfile,timeout=600,env=None):
    options = {"creationflags":subprocess.CREATE_NO_WINDOW} if os.name=="nt" else {}
    if env is None:
        env = {**os.environ,"PYTHONUTF8":"1","PYTHONIOENCODING":"utf-8"}
    with logfile.open("wb") as log:
        proc = await asyncio.create_subprocess_exec(*command,cwd=ROOT,env=env,stdout=log,stderr=log,**options)
        try:
            result = await asyncio.wait_for(proc.wait(),timeout)
        except (asyncio.TimeoutError,asyncio.CancelledError):
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
    if result:
        detail = logfile.read_text(encoding="utf-8",errors="replace")[-3000:]
        raise RuntimeError(detail or f"Процесс завершился с кодом {result}")


def keep_task(coro):
    task = asyncio.create_task(coro)
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def run_analysis(drawing_id,directory,mode=None,model=None):
    output = DATA/"analysis"/drawing_id
    output.mkdir(parents=True,exist_ok=True)
    try:
        async with analysis_slots:
            update_state(directory,"analysis",state="running")
            meta = read_json(directory/"metadata.json")
            settings=LLMSettings.from_env(mode,model)
            options=['--mode',settings.analysis_mode,'--model',settings.model]
            atomic_json(output/'progress.json',{'stage':'cv_ocr','mode':settings.analysis_mode,'model':settings.model})
            await process([sys.executable,"-m","app.worker","analyze","--input",str(directory/"original.pdf"),
                "--output",str(output),"--id",drawing_id,"--filename="+meta["filename"],"--sha256",meta["sha256"]]+options,output/"worker.log",
                timeout=1200+meta.get('page_count',1)*settings.max_iterations*(settings.timeout*(settings.max_retries+1)+540))
            statistics = read_json(output/"statistics.json")
            update_state(directory,"analysis",state="completed",statistics=statistics)
        intent=read_json(output/'cad-intent.json')
        interpretation=read_json(output/'interpretation.json')
        reconstruction=output/'reconstruction/diagnostics.json'
        if settings.analysis_mode=='hybrid' and reconstruction.is_file():
            engine=read_json(reconstruction)
            selected=engine.get('selected_build') or engine.get('visual_preview')
            if selected:
                build_id=uuid4().hex;dest=DATA/'models'/drawing_id/build_id
                shutil.copytree(selected,dest)
                preview=read_json(dest/'preview.json');quality=engine['quality']
                formats=[name for name,file in [('step','model.step'),('fcstd','model.FCStd'),('stl','model.stl')] if (dest/file).is_file()]
                preview.update(quality_level=quality,parametric_cad=quality in ('A','B','C'),available_formats=formats)
                atomic_json(dest/'preview.json',preview)
                values={'state':'completed','build_id':build_id,'quality_level':quality,'result':{k:v for k,v in preview.items() if k not in ('vertices','triangles','edges')},
                        'reconstruction_status':preview['reconstruction_status'],'simplified_available':bool(engine.get('visual_preview'))}
                update_state(directory,'cad',**values);atomic_json(directory/'current-model.json',values)
                return
        if intent.get('operations'):
            if find_freecad():
                alternatives=read_json(output/'cad-candidates.json').get('alternatives',[]) if (output/'cad-candidates.json').is_file() else []
                candidates=[{'intent':intent,'interpretation':None}]+alternatives
                attempts=[]
                for index,candidate in enumerate(candidates):
                    build_id=uuid4().hex
                    update_state(directory,'cad',state='queued',build_id=build_id,reconstruction_status=interpretation['status'])
                    passed=await run_build(drawing_id,directory,build_id,CADIntent.model_validate(candidate['intent']))
                    attempts.append({'candidate_index':index,'build_id':build_id,'passed':passed})
                    atomic_json(output/'cad-attempts.json',attempts)
                    if passed:
                        atomic_json(output/'cad-intent.json',candidate['intent'])
                        if candidate['interpretation']:
                            selected=candidate['interpretation'];interpretation['selected_page']=selected['page']
                            interpretation['pages']=[selected if p.get('page')==selected['page'] else p for p in interpretation['pages']]
                        interpretation['cad_selection']={'candidate_index':index,'build_id':build_id,'attempt_count':len(attempts)}
                        atomic_json(output/'interpretation.json',interpretation)
                        break
                if not any(a['passed'] for a in attempts):
                    interpretation.update(status='failed',message='Все подтверждённые гипотезы отклонены проверкой CAD или сравнением с чертежом.',cad_attempts=attempts)
                    atomic_json(output/'interpretation.json',interpretation)
            else:update_state(directory,'cad',state='failed',error='Автоматический CAD Intent готов, но FreeCAD не найден.')
        else:
            details=[x.get('reason') for x in interpretation['pages'] if x.get('reason')]
            clarifications=[c for p in interpretation['pages'] for c in p.get('clarifications',[])]
            update_state(directory,'cad',state='needs_review',reconstruction_status='needs_review',clarifications=clarifications,
                         error=' '.join(details) or interpretation['message'])
    except asyncio.CancelledError:
        update_state(directory,"analysis",state="failed",error="Операция прервана при остановке сервера")
        raise
    except Exception as exc:
        update_state(directory,"analysis",state="failed",error=str(exc) or "Превышено время анализа")


async def run_build(drawing_id,directory,build_id,intent):
    output = DATA/"models"/drawing_id/build_id
    output.mkdir(parents=True,exist_ok=True)
    try:
        async with cad_slots:
            update_state(directory,"cad",state="running",build_id=build_id)
            atomic_json(output/"intent.json",intent.model_dump(mode="json"))
            cmd,env = cad_command(output/"intent.json",output)
            try:
                await process(cmd,output/"worker.log",timeout=180,env=env)
            except RuntimeError:
                if (output/"result.json").is_file():
                    raise RuntimeError(read_json(output/"result.json").get("error","Ошибка FreeCAD"))
                raise
            result = read_json(output/"result.json")
            if not result["ok"]:
                raise RuntimeError(result["error"])
            validation=await asyncio.to_thread(validate_model,output,intent.model_dump(mode='json'))
            if validation['overall_status']=='failed':
                raise RuntimeError('Модель не прошла сравнение с чертежом или проверку размеров. См. Validation JSON.')
            result['validation_status']=validation['overall_status']
            update_state(directory,"cad",state="completed",build_id=build_id,result=result,
                         reconstruction_status=intent.reconstruction_status)
            atomic_json(directory/'current-model.json',read_json(directory/'status.json')['cad'])
            return True
    except asyncio.CancelledError:
        update_state(directory,"cad",state="failed",build_id=build_id,error="Построение прервано")
        raise
    except Exception as exc:
        previous=directory/'current-model.json'
        if previous.is_file():
            kept=read_json(previous);kept.pop('updated_at',None)
            update_state(directory,'cad',**kept,last_attempt_error=str(exc),failed_build_id=build_id)
        else:update_state(directory,"cad",state="failed",build_id=build_id,error=str(exc) or "Превышено время CAD-построения")
        return False


@app.get("/")
def index():
    return FileResponse(ROOT/"static"/"index.html")


@app.get("/api/health")
def health():
    freecad = find_freecad()
    return {"status":"ok","python":sys.version.split()[0],"freecad_available":bool(freecad),
            "freecad_executable":str(freecad) if freecad else None,"render_dpi":300,"ocr_enabled":ocr_available(),
            "neural_models":"local_tesseract_and_optional_ollama_vision","cloud_inference":False,
            "llm":llm_status()}


@app.get('/api/llm/status')
def local_llm_status(model:str|None=None):
    return llm_status(LLMSettings.from_env(model=model))


@app.get('/api/samples')
def sample_drawings():
    folder=ROOT.parent/'чертежи'
    return [{'id':p.stem,'filename':p.name} for p in sorted(folder.glob('*.pdf'),key=lambda p:(len(p.stem),p.stem))
            if re.fullmatch(r'[1-9]\d?',p.stem)]


@app.post('/api/samples/{number}',status_code=201)
async def load_sample(number:int):
    if not 1<=number<=99:raise HTTPException(404,'Чертёж не найден')
    path=ROOT.parent/'чертежи'/f'{number}.pdf'
    if not path.is_file():raise HTTPException(404,'Чертёж не найден')
    with path.open('rb') as stream:
        return await upload(UploadFile(file=stream,filename=path.name))


@app.get('/api/progress/{drawing_id}')
def analysis_progress(drawing_id:str):
    get_dir(drawing_id)
    path=DATA/'analysis'/drawing_id/'progress.json'
    return read_json(path) if path.is_file() else {'stage':'idle'}


@app.get('/api/llm/{drawing_id}')
def llm_diagnostics(drawing_id:str):
    get_dir(drawing_id)
    path=DATA/'analysis'/drawing_id/'llm'/'diagnostics.json'
    if not path.is_file():raise HTTPException(409,'LLM-анализ ещё не завершён')
    return read_json(path)


@app.post("/api/upload",status_code=201)
async def upload(file: UploadFile=File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(415,"Требуется файл PDF")
    drawing_id = uuid4().hex
    directory = upload_dir(drawing_id)
    directory.mkdir(parents=True)
    size = 0
    try:
        with (directory/"original.pdf").open("wb") as target:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413,"Лимит загрузки: 25 МБ")
                target.write(chunk)
        if size == 0:
            raise HTTPException(422,"Пустой файл")
        async with analysis_slots:
            await process([sys.executable,"-m","app.worker","inspect","--input",str(directory/"original.pdf"),
                           "--output",str(directory)],directory/"inspection.log",timeout=120)
        inspection = read_json(directory/"inspection.json")
        filename = (file.filename or "drawing.pdf").replace("\\","/").split("/")[-1][:200]
        meta = {"id":drawing_id,"filename":filename,"size_bytes":size,"uploaded_at":stamp(),**inspection}
        atomic_json(directory/"metadata.json",meta)
        atomic_json(directory/"status.json",{"analysis":{"state":"idle"},"cad":{"state":"idle"}})
        return meta
    except BaseException as exc:
        shutil.rmtree(directory,ignore_errors=True)
        if isinstance(exc,HTTPException) or isinstance(exc,asyncio.CancelledError):
            raise
        raise HTTPException(422,"PDF не удалось открыть: "+str(exc)[-1000:])
    finally:
        await file.close()


@app.get("/api/drawing/{drawing_id}")
def drawing(drawing_id:str):
    directory = get_dir(drawing_id)
    return {**read_json(directory/"metadata.json"),**read_json(directory/"status.json")}


@app.get("/api/drawing/{drawing_id}/pdf")
def original_pdf(drawing_id:str):
    return FileResponse(get_dir(drawing_id)/"original.pdf",media_type="application/pdf")


@app.get("/api/drawing/{drawing_id}/page/{page}")
def page_preview(drawing_id:str,page:int):
    directory = get_dir(drawing_id)
    if not 1<=page<=read_json(directory/"metadata.json")["page_count"]:
        raise HTTPException(404,"Страница не найдена")
    return FileResponse(directory/f"preview-{page}.png",media_type="image/png")


@app.post("/api/analyze/{drawing_id}",status_code=202)
async def analyze(drawing_id:str,mode:str|None=None,model:str|None=None):
    if mode and mode not in ('classic','hybrid'):raise HTTPException(422,'Режим: classic или hybrid')
    settings=LLMSettings.from_env(mode,model)
    try:settings.validate_local()
    except ValueError as exc:raise HTTPException(422,str(exc))
    directory = get_dir(drawing_id)
    state = read_json(directory/"status.json")["analysis"]
    if state["state"] in ("queued","running"):
        return state
    if read_json(directory/'status.json')['cad']['state'] in ('queued','running'):
        raise HTTPException(409,'Дождитесь завершения CAD-построения перед повторным анализом')
    if len(tasks)>=8:
        raise HTTPException(429,"Очередь заполнена. Повторите запрос позже.")
    if settings.analysis_mode=='hybrid' and settings.enabled:
        availability=await asyncio.to_thread(llm_status,settings)
        if not availability['available']:raise HTTPException(422,availability['reason'])
    update_state(directory,"analysis",state="queued",requested_model=settings.model)
    update_state(directory,'cad',state='idle')
    keep_task(run_analysis(drawing_id,directory,settings.analysis_mode,settings.model))
    return {"state":"queued","status_url":f"/api/drawing/{drawing_id}"}


def completed_analysis(drawing_id):
    directory = get_dir(drawing_id)
    if read_json(directory/"status.json")["analysis"]["state"] != "completed":
        raise HTTPException(409,"Анализ ещё не завершён")
    return DATA/"analysis"/drawing_id/"analysis.json"


@app.get("/api/analysis/{drawing_id}")
def analysis(drawing_id:str):
    return FileResponse(completed_analysis(drawing_id),media_type="application/json")


@app.get("/api/analysis/{drawing_id}/download")
def download_analysis(drawing_id:str):
    return FileResponse(completed_analysis(drawing_id),media_type="application/json",filename=f"drawing-{drawing_id}-analysis.json")


@app.get("/api/artifacts/{drawing_id}/{page_dir}/{filename}")
def artifact(drawing_id:str,page_dir:str,filename:str):
    completed_analysis(drawing_id)
    if not re.fullmatch(r"page-[1-9]\d?",page_dir) or not re.fullmatch(r"(?:original|overlay|physical-strokes|binary-detection|ROI\d{3}|embedded-\d+)\.(?:png|jpg|jpeg|jpx|jb2|tiff|bmp)|(?:diagnostics|layout|ocr)\.json",filename):
        raise HTTPException(404,"Файл не найден")
    path = DATA/"analysis"/drawing_id/page_dir/filename
    if not path.is_file():
        raise HTTPException(404,"Файл не найден")
    return FileResponse(path)


def model_file(drawing_id,name,variant='main'):
    directory = get_dir(drawing_id)
    if variant=='simplified':
        path=DATA/'analysis'/drawing_id/'reconstruction/visual'/name
        if not path.is_file():raise HTTPException(409,'Упрощённая геометрия в этом формате отсутствует')
        return path
    state = read_json(directory/"status.json")["cad"]
    if state["state"] != "completed":
        raise HTTPException(409,"Готовая CAD-модель отсутствует")
    path=DATA/"models"/drawing_id/state["build_id"]/name
    if not path.is_file():raise HTTPException(409,'Этот формат недоступен для выбранной модели')
    return path


@app.get("/api/model/{drawing_id}/preview")
def preview(drawing_id:str,variant:str='main'):
    if variant not in ('main','simplified'):raise HTTPException(422,'Неизвестный вариант модели')
    return FileResponse(model_file(drawing_id,"preview.json",variant),media_type="application/json")


@app.get("/api/model/{drawing_id}/step")
def step(drawing_id:str,variant:str='main'):
    if variant not in ('main','simplified'):raise HTTPException(422,'Неизвестный вариант модели')
    return FileResponse(model_file(drawing_id,"model.step",variant),filename="model.step",media_type="application/step")


@app.get("/api/model/{drawing_id}/fcstd")
def fcstd(drawing_id:str,variant:str='main'):
    if variant not in ('main','simplified'):raise HTTPException(422,'Неизвестный вариант модели')
    return FileResponse(model_file(drawing_id,"model.FCStd",variant),filename="model.FCStd")


@app.get('/api/model/{drawing_id}/stl')
def stl(drawing_id:str,variant:str='main'):
    if variant not in ('main','simplified'):raise HTTPException(422,'Неизвестный вариант модели')
    path=model_file(drawing_id,'model.stl',variant)
    if not path.is_file():raise HTTPException(409,'Перестройте модель для экспорта STL')
    return FileResponse(path,filename='model.stl',media_type='model/stl')


@app.get('/api/interpretation/{drawing_id}')
def interpretation(drawing_id:str):
    completed_analysis(drawing_id)
    return FileResponse(DATA/'analysis'/drawing_id/'interpretation.json',media_type='application/json',filename='geometry-interpretation.json')


@app.get('/api/layout/{drawing_id}')
def layout_json(drawing_id:str):
    completed_analysis(drawing_id)
    return FileResponse(DATA/'analysis'/drawing_id/'layout.json',media_type='application/json',filename='drawing-layout.json')


@app.get('/api/cad-intent/{drawing_id}')
def automatic_intent(drawing_id:str):
    completed_analysis(drawing_id)
    return FileResponse(DATA/'analysis'/drawing_id/'cad-intent.json',media_type='application/json',filename='cad-intent.json')


@app.get('/api/validation/{drawing_id}')
def validation_json(drawing_id:str):
    directory=get_dir(drawing_id);state=read_json(directory/'status.json')['cad']
    if not state.get('build_id'):raise HTTPException(409,'Валидация ещё не выполнялась')
    path=DATA/'models'/drawing_id/state['build_id']/'validation.json'
    if not path.is_file():raise HTTPException(409,'Отчёт валидации ещё не готов')
    return FileResponse(path,media_type='application/json',filename='validation.json')


@app.get('/api/model/{drawing_id}/diagnostic/{name}')
def model_diagnostic(drawing_id:str,name:str):
    if name not in ('projection-xy.png','projection-xz.png','projection-yz.png','section-comparison.png'):
        raise HTTPException(404,'Диагностика не найдена')
    path=model_file(drawing_id,name)
    if not path.is_file():raise HTTPException(404,'Диагностика не найдена')
    return FileResponse(path)
