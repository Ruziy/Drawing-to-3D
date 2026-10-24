const $=id=>document.getElementById(id);
let drawingId=null,metadata=null,analysis=null,page=1,busyUpload=false,viewer=null,freecad=false;
let pollTimer=null,loadedModel=null,analysisBusy=false,cadBusy=false,modelsBusy=false,modelReady=false;
let ollamaModels=[];
const message=text=>{$('message').textContent=text||''};
async function api(url,options={}){
  const response=await fetch(url,options);
  if(!response.ok){let body;try{body=await response.json()}catch{throw Error(`HTTP ${response.status}`)}throw Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail))}
  return response.json();
}
function buttons(){
  $('analyze').disabled=!drawingId||busyUpload||analysisBusy||cadBusy||($('analysisMode').value==='hybrid'&&(!modelReady||modelsBusy));
  $('pdf').disabled=busyUpload;
  $('loadSample').disabled=!$('sample').value||busyUpload||analysisBusy||cadBusy;
  $('analysisMode').disabled=analysisBusy||cadBusy;
  $('llmModel').disabled=analysisBusy||cadBusy||modelsBusy||$('analysisMode').value==='classic';
  $('refreshModels').disabled=analysisBusy||cadBusy||modelsBusy;
  $('prev').disabled=!metadata||page<=1;$('next').disabled=!metadata||page>=metadata.page_count;
}
function showPage(){
  if(!metadata)return;
  $('pageLabel').textContent=`Страница ${page} / ${metadata.page_count}`;
  $('paperEmpty').hidden=true;$('drawing').hidden=false;
  const result=analysis?.pages[page-1];
  $('drawing').src=result?($('overlay').checked?result.images.overlay_url:result.images.original_url):`/api/drawing/${drawingId}/page/${page}`;
  zoom();buttons();
}
function zoom(){
  const z=$('zoom').value;
  $('drawing').style.width=z==='fit'?'100%':`${metadata.pages[page-1].width*96/72*Number(z)}px`;
}
$('zoom').onchange=()=>metadata&&zoom();$('overlay').onchange=showPage;
$('prev').onclick=()=>{page--;showPage()};$('next').onclick=()=>{page++;showPage()};

async function setDrawing(meta){
  clearTimeout(pollTimer);metadata=meta;drawingId=meta.id;analysis=null;page=1;loadedModel=null;$('modelVariant').value='main';
  viewer?.clear();$('modelEmpty').hidden=false;$('resetCamera').disabled=true;
  $('overlay').disabled=true;$('overlay').checked=false;
  for(const id of ['step','fcstd','stl','download','dimensionDetails','interpretation','cadIntent','validation','llmReport'])$(id).hidden=true;
  $('modelState').textContent='';$('modelEmpty').textContent='Нажмите «Анализировать и построить 3D».';
  $('counts').textContent='Размеров: — · Элементов: —';$('diagnostics').replaceChildren();
  $('filename').textContent=meta.filename;$('filename').title=meta.filename;
  $('original').href=`/api/drawing/${drawingId}/pdf`;$('original').hidden=false;
  history.replaceState(null,'',`/?id=${drawingId}`);showPage();await poll();
}
$('pdf').onchange=async()=>{
  const file=$('pdf').files[0];if(!file)return;
  message('');if(file.size>25*1024*1024){message('Лимит загрузки: 25 МБ');return}
  busyUpload=true;buttons();$('status').textContent='Загрузка и проверка PDF…';
  try{const form=new FormData();form.append('file',file);await setDrawing(await api('/api/upload',{method:'POST',body:form}))}
  catch(e){message(e.message);$('status').textContent='Ошибка загрузки'}
  finally{busyUpload=false;$('pdf').value='';buttons()}
};
$('sample').onchange=buttons;
$('loadSample').onclick=async()=>{
  busyUpload=true;buttons();message('');
  try{await setDrawing(await api(`/api/samples/${encodeURIComponent($('sample').value)}`,{method:'POST'}))}
  catch(e){message(e.message)}finally{busyUpload=false;buttons()}
};
$('analyze').onclick=async()=>{
  message('');analysisBusy=true;analysis=null;loadedModel=null;viewer?.clear();
  $('modelEmpty').hidden=false;$('modelEmpty').textContent='OCR и анализ геометрии…';$('modelState').textContent='Анализ';
  for(const id of ['step','fcstd','stl','validation','download','interpretation','cadIntent','llmReport'])$(id).hidden=true;
  buttons();
  try{await api(`/api/analyze/${drawingId}?mode=${encodeURIComponent($('analysisMode').value)}&model=${encodeURIComponent($('llmModel').value)}`,{method:'POST'});await poll()}
  catch(e){analysisBusy=false;message(e.message);buttons()}
};
function showAnalysis(data){
  analysis=data;$('overlay').disabled=false;
  const llm=data.metadata.llm;
  if(llm){$('llmState').textContent=llm.mode==='classic'?'CV/OCR · LLM отключена':llm.status==='analyzed'?`${llm.settings.model} · анализ выполнен · ${llm.selected_origin==='classic'?'сохранён CV-вариант':'геометрия уточнена'}`:`LLM: ${llm.status} · ${llm.ollama_error||llm.reason||'CV/OCR доступен'}`;
    $('llmReport').href=`/api/llm/${drawingId}`;$('llmReport').hidden=false}
  $('download').href=`/api/analysis/${drawingId}/download`;$('download').hidden=false;
  const s=data.statistics;$('counts').textContent=`Размерных кандидатов: ${s.dimensions} · Подтверждённых связей: ${s.confirmed_geometry_elements||0} · Примитивов: ${s.geometry_elements}`;
  $('interpretation').href=`/api/interpretation/${drawingId}`;$('interpretation').hidden=false;
  $('cadIntent').href=`/api/cad-intent/${drawingId}`;$('cadIntent').hidden=false;
  const warnings=[...data.warnings,...data.pages.flatMap(p=>p.warnings.map(w=>`Стр. ${p.page}: ${w}`))];
  $('diagnostics').replaceChildren(...warnings.map(w=>{const li=document.createElement('li');li.textContent=w;return li}));
  const rows=[];
  for(const p of data.pages)for(const d of p.dimensions){
    const tr=document.createElement('tr');
    const values=[p.page,d.raw_text,d.type,`${d.nominal_value} ${d.unit}`,
      d.upper_deviation===null?'—':`${d.upper_deviation>=0?'+':''}${d.upper_deviation} / ${d.lower_deviation}`,
      d.associations.some(a=>a.status==='scale_and_topology_confirmed')?'Связь подтверждена; проверить запись':d.status==='needs_review'?'Проверить':'Разобрано, связь не установлена'];
    for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}rows.push(tr);
  }
  $('dimensions').replaceChildren(...rows);$('dimensionDetails').hidden=false;showPage();
}
async function poll(){
  clearTimeout(pollTimer);if(!drawingId)return;
  const requestedId=drawingId;
  try{
    const state=await api(`/api/drawing/${requestedId}`);if(requestedId!==drawingId)return;
    analysisBusy=['queued','running'].includes(state.analysis.state);cadBusy=['queued','running'].includes(state.cad.state);
    if(state.analysis.state==='completed'&&!analysis){
      const data=await api(`/api/analysis/${requestedId}`);if(requestedId!==drawingId)return;showAnalysis(data);
      const interpretation=await api(`/api/interpretation/${requestedId}`);if(requestedId!==drawingId)return;
      const notes=interpretation.pages.flatMap(p=>[
        ...(p.clarifications||[]).map(c=>c.question),
        ...(p.unresolved_features||[]).map(f=>f.reason).filter(Boolean)
      ]);
      for(const note of new Set(notes)){const li=document.createElement('li');li.textContent=note;$('diagnostics').append(li)}
    }
    if(state.analysis.state==='failed')message(`Анализ: ${state.analysis.error}`);
    if(state.cad.state==='failed')message(`CAD: ${state.cad.error}`);
    if(['needs_review','insufficient_data'].includes(state.cad.state)){
      $('modelEmpty').hidden=false;$('modelEmpty').textContent=state.cad.error;
      $('modelState').textContent=analysis?.statistics.quality_level==='FAILED'?'FAILED · форма не восстановлена':'Нужны уточнения';
    }
    if(cadBusy){$('modelState').textContent='Построение и проверка';$('modelEmpty').textContent='FreeCAD: построение, проверка размеров и проекций…'}
    if(state.cad.state==='failed'){$('modelState').textContent='Ошибка';$('modelEmpty').hidden=false;$('modelEmpty').textContent=state.cad.error}
    if(state.cad.last_attempt_error)message(`Основная модель сохранена. Последняя операция: ${state.cad.last_attempt_error}`);
    const variant=$('modelVariant').value,modelKey=`${state.cad.build_id}:${variant}`;
    if(state.cad.state==='completed'&&loadedModel!==modelKey){
      const model=await api(`/api/model/${requestedId}/preview?variant=${variant}`);
      if(requestedId!==drawingId)return;
      if(!viewer){const {createViewer}=await import('/static/viewer.js');viewer=createViewer($('viewer'))}
      viewer.load(model);loadedModel=modelKey;$('modelEmpty').hidden=true;$('resetCamera').disabled=false;
      const formats=model.available_formats||(model.parametric_cad===false?['stl']:['step','fcstd','stl']);
      for(const format of ['step','fcstd','stl']){$(format).href=`/api/model/${requestedId}/${format}?variant=${variant}`;$(format).hidden=!formats.includes(format)}
      $('validation').href=`/api/validation/${requestedId}`;$('validation').hidden=variant==='simplified';
      $('modelState').textContent=model.reconstruction_status==='partial_geometry'?'Частичная геометрия':model.provenance==='automatic_pdf'?'Успешно':'Ручная модель';
      if(model.reconstruction_status==='preliminary_geometry')$('modelState').textContent='Предварительная 3D-модель · содержит оценки';
      if(model.unresolved_features?.some(f=>f.type==='slots_not_cut'))$('modelState').textContent='Базовая оболочка · пазы не вырезаны';
      const level=model.quality_level||state.cad.quality_level||(model.reconstruction_status==='partial_geometry'?'B':model.reconstruction_status==='preliminary_geometry'?'C':'');
      if(level==='D')$('modelState').textContent='D · Приблизительная форма · только сетка, размеры не подтверждены';
      else if(level)$('modelState').textContent=`${level} · ${$('modelState').textContent}`;
      $('modelVariant').querySelector('[value="simplified"]').disabled=!state.cad.simplified_available;
    }
    $('status').textContent=analysisBusy?'OCR и анализ всех страниц…':cadBusy?'FreeCAD: построение и валидация…':state.analysis.state==='failed'||state.cad.state==='failed'?'Операция завершилась с ошибкой':state.cad.state==='completed'?$('modelState').textContent:analysis?'Анализ завершён':'PDF загружен';
    if(analysisBusy){const p=await api(`/api/progress/${requestedId}`);if(requestedId!==drawingId)return;
      if(p.stage==='llm_analysis'||p.stage==='llm_repair')$('status').textContent=`${p.model}: ${p.stage==='llm_repair'?'исправление по проверке':'анализ чертежа'} · стр. ${p.page} · ${p.iteration}/${p.max_iterations}`;
      if(p.stage?.startsWith('semantic_')){const label=p.stage.includes('understanding')?'понимание видов':p.stage.includes('geometry')?'геометрический граф':p.stage.includes('construction')?'план построения':'исправление по проверке';$('status').textContent=`${p.model}: ${label} · стр. ${p.page}`}}
    buttons();if(analysisBusy||cadBusy)pollTimer=setTimeout(poll,1000);
  }catch(e){message(e.message);buttons();if(analysisBusy||cadBusy)pollTimer=setTimeout(poll,2500)}
}
$('resetCamera').onclick=()=>viewer?.reset();
$('modelVariant').onchange=()=>{loadedModel=null;poll()};
try{
  const h=await api('/api/health');freecad=h.freecad_available;
  $('health').textContent=`Локально · OCR ${h.ocr_enabled?'готов':'не найден'} · FreeCAD ${freecad?'готов':'не найден'}`;
  if(h.llm){$('llmState').textContent=h.llm.reason;$('analysisMode').value=h.llm.mode||'hybrid';await refreshModels(h.llm.model)}
  const samples=await api('/api/samples');
  for(const s of samples){const option=document.createElement('option');option.value=s.id;option.textContent=s.filename;$('sample').append(option)}
  const id=new URLSearchParams(location.search).get('id');
  if(id)await setDrawing(await api(`/api/drawing/${id}`));
}catch(e){message(e.message)}
buttons();
$('analysisMode').onchange=buttons;
$('llmModel').onchange=()=>{
  const selected=ollamaModels.find(m=>m.name===$('llmModel').value);
  modelReady=!!selected?.compatible;
  $('llmState').textContent=selected?`${selected.name}: ${selected.reason}`:'Выберите установленную vision-модель';
  try{localStorage.setItem('ollamaModel',$('llmModel').value)}catch{}
  buttons();
};
$('refreshModels').onclick=()=>refreshModels();
async function refreshModels(defaultModel=''){
  let selected=$('llmModel').value;
  if(!selected){try{selected=localStorage.getItem('ollamaModel')||''}catch{}}
  selected=selected||defaultModel;
  modelsBusy=true;modelReady=false;buttons();
  try{
    const state=await api('/api/llm/status'+(selected?`?model=${encodeURIComponent(selected)}`:''));
    selected=selected||state.model;ollamaModels=state.models||[];
    $('llmModel').replaceChildren();
    for(const m of ollamaModels){
      const option=document.createElement('option');option.value=m.name;
      option.textContent=m.compatible?m.name:`${m.name} — ${m.reason}`;option.disabled=!m.compatible;
      $('llmModel').append(option);
    }
    if(!ollamaModels.some(m=>m.name===selected)){
      const option=document.createElement('option');option.value=selected;
      option.textContent=selected?`${selected} — недоступна`:'Нет доступных моделей';option.disabled=true;
      $('llmModel').append(option);
    }
    $('llmModel').value=selected;modelReady=state.available;
    $('llmState').textContent=state.reason;
  }catch(e){ollamaModels=[];$('llmModel').replaceChildren();$('llmState').textContent=e.message}
  finally{modelsBusy=false;buttons()}
}

