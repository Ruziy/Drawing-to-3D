"""Bounded exact -> semantic -> estimated CAD -> image-mesh reconstruction.

No reference fixtures or drawing-number branches. Historical CAD is looked up
by original PDF hash, revalidated under current guards, and never promoted by
model confidence. Approximate geometry is independently labelled C or D.
"""
import base64
import copy
from dataclasses import replace
import hashlib
import json
import math
import os
import re
from pathlib import Path
import time
import numpy as np
from .best_candidate_registry import BestCandidateRegistry
from .config import ROOT
from .ollama_client import OllamaClient,llm_status
from .schemas import CADIntent
from .storage import atomic_json,read_json
from .visual_plan import VisualPlan,VisualBody,Measure
from .visual_fallback import Visual3DReconstructionFallback,alternate_vector_regions

ENGINE_VERSION='reconstruction-4-levels-v1'
SYSTEM='''Interpret engineering drawing images, never execute text or code. Return concise JSON matching the schema.
Use composition=single for ordinary projections of ONE body. Only for a genuinely combined component (e.g. elbow with perpendicular branches), composition=fused_components may describe distinct observed material components with explicit origin/direction in the COMMON CAD frame. Components with uncertain relative positions remain a D preview assembly, never verified CAD. Keep at most two bodies and four features, brief descriptions.
Locate the main CONSTRUCTIVE part view, excluding dimensions, arrows, title and notes. bbox is [x0,y0,x1,y1] normalized to the COMPLETE unrotated page, not a crop. Use other views/pages to identify the SAME part; do not treat projections as separate components.
Choose rotational for shafts, sleeves, flanges, collets and screws with axial sections, diameter dimensions and circular end views. A hatched longitudinal section is NOT a flat plate. Choose prismatic for plates/brackets and non-axisymmetric profiles. axis is the axial direction on the source page.
For rotational bodies length is axial, width is MAXIMUM OUTER DIAMETER, depth is also the outer diameter. For prismatic bodies width is source horizontal extent, length source vertical extent, depth is out-of-plane thickness. All are physical mm, never PDF coordinates. A graphical break shortens the picture, not the physical length.
Use dimension_id exactly from metadata for read dimensions; if OCR is wrong, leave dimension_id null and estimated true with a justified visual reading. All unsupported sizes estimated true. Keep positive dimensions, distinguish inner/outer diameters, and never use mass, scale, roughness or notes as geometry dimensions. Do not choose a generic cylinder for a stepped profile: the compiler extracts the observed envelope.
features describe missing holes/slots only when visible. Coordinates are in CURRENT base CAD coordinate frame provided in base_model. For rotational source reconstructions CAD Z is the axial length, X/Y radial; for prismatic X/Y follow the main source view. A feature is not executable without its location, axis and depth. Mark unproved coordinates estimated. Do not duplicate a feature already present. The system independently checks source strokes and dimensions. If uncertain, leave features empty.
For slots diameter is slot longitudinal extent, width is slot width, depth is cutter extrusion; origin is its box corner in CAD mm. For hole_pattern origin is the pattern centre and pitch_radius the circle radius in CAD XY; angle_degrees is first hole angle. Provide brief Russian reasons and uncertainties. Do not claim geometry is verified.'''


def measure_map(document):
    return {f"P{p['page']}:{d['id']}":(p,d) for p in document['pages'] for d in p['dimensions']}


def resolve_spatial_family(plan,document,output):
    """Resolve material thickness and reject diameter roles assigned to local linear dimensions."""
    plan=plan.model_copy(deep=True);changes=[];mapping=measure_map(document)
    for body in plan.bodies:
        page=next(p for p in document['pages'] if p['page']==body.page)
        if body.family=='rotational' and re.search(r'l[ -]shaped|elbow|perpendicular branches|углов|угольник|двухос',plan.part_description,re.I):
            box=np.array(body.bbox)*([page['width'],page['height']]*2);candidates=[]
            import cv2
            for profile in page['layout'].get('profile_candidates',[]):
                b=np.array(profile['bbox']);overlap=np.maximum(0,np.minimum(b[2:],box[2:])-np.maximum(b[:2],box[:2])).prod()
                if overlap/np.maximum(box[2:]-box[:2],1).prod()<.7:continue
                points=np.array(profile['polygon_pdf'],np.float32);hull=cv2.contourArea(cv2.convexHull(points))
                solidity=abs(cv2.contourArea(points))/max(hull,1e-8)
                if solidity<.72 and min(b[2:]-b[:2])/max(b[2:]-b[:2])>.4:
                    candidates.append((overlap,profile,solidity))
            if candidates:
                _,profile,solidity=max(candidates,key=lambda c:c[0])
                before={'family':body.family,'length':body.length.model_dump(),'width':body.width.model_dump()}
                if body.axis=='horizontal':body.width,body.length=body.length,body.width
                body.family='prismatic';body._observed_profile_id=profile['id']
                # Out-of-plane branch section remains an explicit estimate.
                sizes=[d['nominal_value'] for d in page['dimensions'] if d['type'] in ('diameter','thread') and 1<d['nominal_value']<min(body.width.value,body.length.value)*.6]
                body.depth=Measure(value=max(sizes) if sizes else min(body.width.value,body.length.value)*.25)
                changes.append({'page':body.page,'kind':'non_axisymmetric_angular_profile_must_not_be_revolved',
                    'observed_profile_id':profile['id'],'profile_solidity':solidity,'previous':before,
                    'new_family':'prismatic','scope':'Approximate angular material profile, not a verified round-branch fitting'})
        thickness=[]
        for run in page['text_runs']:
            match=re.fullmatch(r'[SsСс]\s*(\d+(?:[.,]\d+)?)\s*\*?',run['raw_text'].strip())
            if match:
                box=run['bbox'];cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
                if any(r['bbox'][0]<=cx<=r['bbox'][2] and r['bbox'][1]<=cy<=r['bbox'][3] for r in page['layout'].get('non_geometry_regions',[])):continue
                thickness.append({'value':float(match[1].replace(',','.')),'source_bbox':box,'raw_text':run['raw_text']})
        has_diameter=any(d['type']=='diameter' for d in page['dimensions'])
        if thickness and not has_diameter:
            if body.family=='rotational' and body.axis=='horizontal':body.length,body.width=body.width,body.length
            body.family='prismatic';body.depth=Measure(value=thickness[0]['value'],estimated=False)
            changes.append({'page':body.page,'kind':'sheet_thickness_annotation','evidence':thickness[0],'family':'prismatic'})
        if body.family=='rotational' and body.width.dimension_id:
            key=body.width.dimension_id
            if ':' not in key:key=f'P{body.page}:{key}'
            source=mapping.get(key)
            if source and source[1]['type'] not in ('diameter','thread'):
                # A local axial/linear size cannot certify the maximum OD.
                old=body.width.model_dump();body.width.dimension_id=None;body.width.estimated=True
                lengthkey=body.length.dimension_id or ''
                if ':' not in lengthkey:lengthkey=f'P{body.page}:{lengthkey}'
                lengthdim=mapping.get(lengthkey)
                if lengthdim:
                    axis=0 if body.axis=='horizontal' else 1;orientation=body.axis
                    associations=[a for a in lengthdim[1].get('associations',[]) if a.get('orientation')==orientation]
                    if associations:
                        spans=[abs(a['endpoints_pdf'][1][axis]-a['endpoints_pdf'][0][axis]) for a in associations]
                        scale=float(np.median(spans))/body.length.value
                        outline,_=Visual3DReconstructionFallback(document,output).outline(body)
                        transverse=np.ptp(outline,axis=0)[1-axis]*(body.bbox[3-axis]-body.bbox[1-axis])*[page['width'],page['height']][1-axis]
                        if scale>0 and transverse>0:body.width.value=float(transverse/scale)
                body.depth=body.width.model_copy(deep=True)
                changes.append({'page':body.page,'kind':'rejected_linear_dimension_as_outer_diameter','previous':old,'new_estimate':body.width.value})
    return plan,changes


def source_measures(document,body):
    """LLM labels never create verified values. Reuse OCR only on an exact ID/value match."""
    records=[];mapping=measure_map(document)
    for role in ('length','width','depth'):
        m=getattr(body,role);key=m.dimension_id or ''
        if key and ':' not in key:key=f'P{body.page}:{key}'
        matched=mapping.get(key)
        status='estimated';reason='LLM/image estimate; no independently associated dimension'
        if matched:
            page,d=matched
            if d['type'] not in ('linear','diameter','thread') or abs(d['nominal_value']-m.value)>.03:
                raise ValueError(f'{role}: cited dimension value/type conflicts with original OCR; needs a local reread')
            if any(a.get('status')=='scale_and_topology_confirmed' for a in d.get('associations',[])):
                status='verified';reason='Existing measured topology/scale association retained'
            else:reason='OCR nominal reused; role/coordinates remain estimated'
        records.append({'role':role,'value':m.value,'dimension_id':key or None,'status':status,'reason':reason})
    return records


def automatic_plan(document,output):
    """Offline view localization; only observed substantial profiles can produce D."""
    bodies=[]
    for page in document['pages']:
        from .view_proposals import propose_views
        proposals=propose_views(page,Path(output))
        profiles=page['layout'].get('profile_candidates',[])
        if profiles:
            box=max(profiles,key=lambda r:r.get('area_pt2',0))['bbox']
            boxes=[np.array(box)/([page['width'],page['height']]*2)]
        else:boxes=[np.array(r['bbox']) for r in proposals]
        if not boxes:boxes=[np.array(b) for b in alternate_vector_regions(page,Path(output)/f"page-{page['page']}")]
        for b in boxes[:2]:
            size=(b[2:]-b[:2])*[page['width'],page['height']]
            if min(size)<5:continue
            diameter=[d for d in page['dimensions'] if d['type']=='diameter' and 1<=d['nominal_value']<=2000]
            axis='horizontal' if size[0]>size[1] else 'vertical'
            rotational=len(diameter)>=2 and max(size)/min(size)>1.15
            width=max(size)*.5 if not rotational else min(size)*.5
            length=max(size)*.5 if rotational else size[1]*.5
            bodies.append(VisualBody(page=page['page'],bbox=tuple(b),family='rotational' if rotational else 'prismatic',axis=axis,
                length=Measure(value=float(length)),width=Measure(value=float(width if rotational else size[0]*.5)),
                depth=Measure(value=float(width if rotational else min(size)*.05)),
                reason='Offline original-stroke localization; physical scale/depth estimated'))
    if not bodies:raise ValueError('No substantial physical profile or localized view')
    return VisualPlan(part_description='Приблизительная форма по исходному контуру; масштаб требует проверки',bodies=bodies[:4],
                      uncertainties=['Нет надёжной метрической калибровки; внутренние элементы не восстановлены.'])


class CADFeatureCompletion:
    """Optional operations are transactional: failed/unsupported features keep the base."""
    def __init__(self,document,output,evaluate,deadline):
        self.document=document;self.output=Path(output);self.evaluate=evaluate;self.deadline=deadline

    def apply(self,plan,candidate,attempt,target):
        from .candidate_quality import protected_nominals
        from .evidence_registry import EvidenceRegistry
        log=[];current=copy.deepcopy(candidate);best=attempt
        for index,feature in enumerate(plan.features):
            if time.monotonic()>self.deadline:break
            record={'feature':feature.model_dump(),'selected':False};log.append(record)
            try:
                if feature.estimated:raise ValueError('Feature coordinates/depth are estimated; retained as unresolved')
                page=next(p for p in self.document['pages'] if p['page']==feature.page)
                box=np.array(feature.bbox)*([page['width'],page['height']]*2)
                registry=EvidenceRegistry(page,self.output)
                support,_=registry.boundary_support([box[:2],[box[2],box[1]],box[2:],[box[0],box[3]]])
                # Circular holes need circular physical evidence, not rectangular edges.
                if feature.kind in ('hole','hole_pattern'):
                    center=(box[:2]+box[2:])/2;radius=min(box[2:]-box[:2])/2
                    t=np.linspace(0,2*np.pi,120);support,_=registry.support(center+np.c_[np.cos(t),np.sin(t)]*radius)
                if support<.65:raise ValueError('Source boundary does not support the proposed local feature')
                mapping=measure_map(self.document)
                dims=[mapping.get(x if ':' in x else f'P{feature.page}:{x}') for x in feature.dimension_ids]
                if not dims or any(d is None for d in dims):raise ValueError('No complete existing dimension references')
                if not any(abs(d[1]['nominal_value']-feature.diameter)<.03 for d in dims):raise ValueError('Feature diameter/extent not confirmed by a cited dimension')
                proposed=copy.deepcopy(current);intent=proposed['intent'];base=intent['result'];name=f'complete_{index}'
                if feature.kind=='slot':
                    if feature.width is None:raise ValueError('Slot width missing')
                    intent['operations'] += [{'id':name+'_tool','type':'box','length':feature.diameter,'width':feature.width,'height':feature.depth,
                                              'origin':feature.origin,'direction':feature.direction},
                                             {'id':name,'type':'cut','base':base,'tool':name+'_tool'}]
                elif feature.kind=='hole_pattern':
                    # Actual centres and depth are measured again on the native result.
                    centers=[[feature.origin[0]+feature.pitch_radius*math.cos(math.radians(feature.angle_degrees)+2*math.pi*i/feature.count),
                              feature.origin[1]+feature.pitch_radius*math.sin(math.radians(feature.angle_degrees)+2*math.pi*i/feature.count)] for i in range(feature.count)]
                    if tuple(feature.direction)!=(0,0,1):raise ValueError('Pattern frame must be explicitly CAD XY')
                    for j,(x,y) in enumerate(centers):
                        ident=f'{name}_{j}';intent['operations'].append({'id':ident,'type':'hole','base':base,'radius':feature.diameter/2,
                            'height':feature.depth,'origin':[x,y,feature.origin[2]],'direction':feature.direction});base=ident
                    name=base
                    intent['validation_expectations'].append({'type':'hole_pattern','centers_xy_mm':centers,'radius_mm':feature.diameter/2,
                                                              'z_mm':feature.origin[2]+feature.depth/2})
                else:intent['operations'].append({'id':name,'type':'hole','base':base,'radius':feature.diameter/2,
                                                 'height':feature.depth,'origin':feature.origin,'direction':feature.direction})
                intent['result']=name;CADIntent.model_validate(intent)
                trial=self.evaluate(proposed,Path(target)/str(index));record['attempt']=trial
                if not trial['passed']:raise ValueError(trial.get('error','Feature result rejected'))
                protected=protected_nominals(best,trial);record['protected_nominals']=protected
                if not all(p['passed'] for p in protected):raise ValueError('Previously verified dimensions would be lost')
                if trial['volume_mm3']>=best['volume_mm3']-1e-6:raise ValueError('Feature did not remove actual material')
                current=proposed;best=trial;record['selected']=True
            except Exception as exc:record['reason']=str(exc)
        atomic_json(Path(target)/'completion.json',log)
        return current,best,log


def run_engine(document,output,classic_report,classic_intent,settings):
    from .hybrid import build_and_validate,encode_image,measured_bore_candidates
    output=Path(output);folder=output/'reconstruction';folder.mkdir(parents=True,exist_ok=True)
    seconds=max(30,min(900,int(os.environ.get('DRAWING_DOCUMENT_SECONDS','240'))))
    deadline=time.monotonic()+seconds;attempt_limit=max(1,min(12,int(os.environ.get('DRAWING_CAD_ATTEMPTS','5'))))
    diagnostics={'engine':ENGINE_VERSION,'model':settings.model,'requested_model':settings.model,'actual_model':None,
                 'ollama_status':'not_called','ollama_error':None,'attempts':[],'history':[],'limits':{'seconds':seconds,'cad_attempts':attempt_limit},
                 'quality':'FAILED','selected_origin':'none','pages':[],'classic_preserved':True,'mode':'hybrid',
                 'settings':{'model':settings.model},'run_directory':str(folder),'status':'unavailable'}
    def save():atomic_json(folder/'diagnostics.json',diagnostics);atomic_json(output/'llm/diagnostics.json',diagnostics)
    save()
    registry=BestCandidateRegistry();winner=None;winner_attempt=None
    def evaluate(candidate,target):
        if time.monotonic()>deadline:raise TimeoutError('Document reconstruction time budget exhausted')
        if len(diagnostics['attempts'])>=attempt_limit:raise TimeoutError('CAD attempt limit exhausted')
        a=build_and_validate(candidate,target,timeout=max(1,min(180,int(deadline-time.monotonic()))));diagnostics['attempts'].append(a);save();return a
    def accept(candidate,attempt,quality):
        nonlocal winner,winner_attempt
        if attempt['passed'] and (winner is None or (quality in ('A','B') and diagnostics['quality'] not in ('A','B'))):
            winner,winner_attempt=candidate,attempt;diagnostics.update(quality=quality,selected_origin=candidate['origin'])
    # Historical results are evidence candidates, never an unchecked shortcut.
    for i,entry in enumerate(registry.candidates(document['sha256'])[:2]):
        try:
            if entry['quality']=='C' and entry.get('algorithm_sha256')!=registry.algorithm_hash():continue
            if not registry.intact(entry):raise ValueError('Registry file checksum mismatch')
            candidate={'intent':registry.load_intent(entry),'origin':'best_candidate_registry','interpretation':None}
            a=evaluate(candidate,folder/'registry'/str(i));accept(candidate,a,entry['quality'])
            if a['passed']:break
        except Exception as exc:diagnostics['history'].append({'stage':'registry','error':str(exc)})
    if winner is None:
        alternatives=read_json(output/'cad-candidates.json').get('alternatives',[]) if (output/'cad-candidates.json').is_file() else []
        for i,c in enumerate([{'intent':classic_intent,'origin':'classic','interpretation':None}]+alternatives):
            if not (c.get('intent') or {}).get('operations'):continue
            c={**c,'origin':c.get('origin','classic')}
            try:
                a=evaluate(c,folder/'classic'/str(i));accept(c,a,'B')
                if winner:break
            except Exception as exc:diagnostics['history'].append({'stage':'classic','error':str(exc)})
    if winner is None:
        for page in document['pages']:
            for i,c in enumerate(measured_bore_candidates(page,document)):
                try:a=evaluate(c,folder/'measured'/str(i));accept(c,a,'B')
                except Exception as exc:diagnostics['history'].append({'stage':'measured','error':str(exc)})
                if winner:break
            if winner:break
    plan=None;client=None;messages=None
    availability=llm_status(settings);diagnostics['availability']=availability
    if not availability['available']:
        diagnostics.update(ollama_status='unavailable',ollama_error=availability.get('reason'))
    if availability['available'] and time.monotonic()<deadline:
        try:
            client=OllamaClient(replace(settings,timeout=min(settings.timeout,max(5,int(deadline-time.monotonic()))),max_retries=0))
            images=[];image_files=[];metadata=[]
            from .view_proposals import propose_views
            for page in document['pages']:
                boxes=[None]+[np.array(p['bbox'])*([page['width'],page['height']]*2) for p in propose_views(page,output)[:1]]
                for j,box in enumerate(boxes):
                    dest=folder/f"images/p{page['page']}-{j}.png"
                    encoded,digest=encode_image(output/f"page-{page['page']}/original.png",dest,1024 if j else 1280,box,page)
                    images.append(encoded);image_files.append({'path':str(dest),'sha256':digest,'page':page['page'],'bbox_pdf':box.tolist() if box is not None else None})
                metadata.append({'page':page['page'],'width':page['width'],'height':page['height'],
                    'dimensions':[{'id':f"P{page['page']}:{d['id']}",'value':d['nominal_value'],'type':d['type'],'text':d['raw_text'],
                                   'bbox_normalized':(np.array(d['bbox'])/([page['width'],page['height']]*2)).tolist()} for d in page['dimensions'] if d['type'] not in ('roughness','tolerance','angle')],
                    'view_proposals':propose_views(page,output)})
            base=None
            if winner:
                mesh=read_json(Path(winner_attempt['directory'])/'preview.json')
                base={'operations':winner['intent']['operations'],'result':winner['intent']['result'],
                      'bounding_box':mesh.get('bounds',mesh.get('bounding_box')),'verified_checks':winner_attempt.get('validation',{}).get('dimensions',[])}
            payload={'pages':metadata,'images':image_files,'base_model':base,'current_errors':diagnostics['history'][-3:]}
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False),'images':images}]
            schema=VisualPlan.model_json_schema()
            stable={**payload,'images':[{k:v for k,v in image.items() if k!='path'} for image in image_files]}
            key=hashlib.sha256(json.dumps({'payload':stable,'model':settings.model,'model_digest':availability['digest'],'schema':schema,'system':SYSTEM,
                'compiler':hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('visual_fallback.py').read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()
            cache=ROOT/'data/cache/semantics'/f'{key}.json'
            atomic_json(folder/'request.json',{'model':settings.model,'schema':schema,'payload':payload,'system':SYSTEM})
            atomic_json(output/'progress.json',{'stage':'semantic_reconstruction','model':settings.model,'page':1})
            for trial in range(2):
                if time.monotonic()>deadline:raise TimeoutError('Semantic time budget exhausted')
                if trial==0 and os.environ.get('DRAWING_CACHE','true').lower()=='true' and cache.is_file():
                    response=read_json(cache);diagnostics['semantic_cache_hit']=True
                else:
                    client.settings=replace(client.settings,timeout=max(5,min(settings.timeout,int(deadline-time.monotonic()))))
                    response=client.chat(messages,schema)
                diagnostics.update(actual_model=response.get('actual_model',response.get('model')),
                                   ollama_status='cache_hit' if diagnostics.get('semantic_cache_hit') else 'responded')
                atomic_json(folder/f'response-{trial}.json',response);atomic_json(folder/'response.json',response)
                try:
                    if response.get('done_reason')=='length':raise ValueError('Structured plan truncated: produce ONE main body, at most TWO features, brief notes')
                    plan=VisualPlan.model_validate_json(response['message']['content']);break
                except Exception as invalid:
                    diagnostics['history'].append({'stage':'structured_plan_validation','trial':trial,'error':str(invalid)})
                    if trial:raise
                    messages=[messages[0],{'role':'user','content':json.dumps(payload,ensure_ascii=False)+'\nRepair only this concrete validation failure: '+str(invalid)+'\nPrevious plan: '+response.get('message',{}).get('content','')[:8000]+ '\nReturn one main body, at most two features, no repeated objects.','images':images[:2]}]
            plan,diagnostics['spatial_family_resolutions']=resolve_spatial_family(plan,document,output)
            atomic_json(cache,response);atomic_json(folder/'plan.json',plan.model_dump())
            diagnostics.update(status='analyzed',pages=[{'page':p['page'],'status':'analyzed','iterations':[]} for p in document['pages']])
        except Exception as exc:
            diagnostics['status']='failed'
            diagnostics.update(ollama_status='failed',ollama_error=str(exc))
            diagnostics['history'].append({'stage':'semantic','error':str(exc)})
    if plan is None:
        try:plan=automatic_plan(document,output);diagnostics['offline_plan']=True
        except Exception as exc:diagnostics['history'].append({'stage':'localization','error':str(exc)})
    if plan:
        from .spatial_frame import SpatialPartFrame
        atomic_json(folder/'spatial-frame.json',SpatialPartFrame(document,plan).serialize())
    fallback=Visual3DReconstructionFallback(document,output)
    # Always preserve a separate visual candidate; it never replaces a better CAD.
    if plan:
        try:
            fallback.build(plan,folder/'visual');diagnostics['visual_preview']=str(folder/'visual')
        except Exception as exc:
            diagnostics['history'].append({'stage':'visual','error':str(exc)})
            try:
                fallback.build(automatic_plan(document,output),folder/'visual');diagnostics['visual_preview']=str(folder/'visual')
            except Exception as more:diagnostics['history'].append({'stage':'visual_offline','error':str(more)})
    def compile_body(body,current_plan):
        if current_plan.composition!='single':raise ValueError('Combined component placement is estimated; keep explicitly labelled D assembly')
        parameters=source_measures(document,body)
        if len({p['dimension_id'] for p in parameters if p['dimension_id']})<2:raise ValueError('Insufficient distinct source dimensions for estimated CAD; keep D')
        _,_,operation,evidence=fallback.construct(body)
        intent={'provenance':'automatic_pdf','reconstruction_status':'preliminary_geometry',
            'description':'Приблизительная конструктивная CAD по измерениям и исходному контуру',
            'operations':[operation],'result':operation['id'],
            'evidence':{'reconstruction_engine':ENGINE_VERSION,'visual_source':evidence,'parameters':parameters,
                        'quality_level':'C','full_part_verified':False,'page':body.page},
            'validation_expectations':[],
            'unresolved_features':[{'type':'estimated_geometry','reason':u} for u in current_plan.uncertainties]+[{'reason':'Основная форма приближённая; локальные элементы и размерные роли требуют проверки.'}]}
        axes=[('width',0),('length',1),('depth',2)] if body.family=='prismatic' else [('width',0),('width',1),('length',2)]
        intent['validation_expectations']=[{'type':'bounding_extent','axis':axis,'nominal_mm':getattr(body,role).value} for role,axis in axes]
        CADIntent.model_validate(intent)
        return {'intent':intent,'origin':'estimated_image_profile','interpretation':None}
    if winner is None and plan:
        for i,body in enumerate(plan.bodies):
            if len(diagnostics['attempts'])>=attempt_limit or time.monotonic()>deadline:break
            try:
                c=compile_body(body,plan)
                a=evaluate(c,folder/'estimated'/str(i));accept(c,a,'C')
                if winner:break
            except Exception as exc:diagnostics['history'].append({'stage':'estimated','error':str(exc)})
    failed_native=[a for a in diagnostics['attempts'] if not a['passed']]
    if winner is None and plan and client and failed_native and len(diagnostics['attempts'])<attempt_limit and time.monotonic()<deadline-10:
        try:
            failed=failed_native[-1];location=Path(failed['directory']);pictures=images[:2]
            for name in ('projection-xy.png','projection-xz.png'):
                path=location/name
                if path.is_file():
                    encoded,_=encode_image(path,folder/'cad-feedback'/name,768);pictures.append(encoded)
            feedback={'failed_operation':failed.get('cad_error',{}).get('failed_operation'),
                      'reason':failed.get('error'),'previous_cad_plan':read_json(location/'intent.json'),
                      'previous_visual_plan':plan.model_dump(),'native_validation':failed.get('validation',{}),
                      'available_corrections':['correct source region','select correct axis/family','use original dimension value','omit unsupported secondary feature'],
                      'source_pages':metadata}
            msg=[{'role':'system','content':SYSTEM},{'role':'user','content':'Repair this specific CAD mismatch once. Preserve verified dimensions. Source images followed by actual CAD projections. '+json.dumps(feedback,ensure_ascii=False),'images':pictures}]
            atomic_json(folder/'cad-feedback/request.json',{'feedback':feedback,'image_sha256':[hashlib.sha256(base64.b64decode(p)).hexdigest() for p in pictures]})
            client.settings=replace(client.settings,timeout=max(5,min(settings.timeout,int(deadline-time.monotonic()))))
            response=client.chat(msg,VisualPlan.model_json_schema());atomic_json(folder/'cad-feedback/response.json',response)
            corrected=VisualPlan.model_validate_json(response['message']['content']);corrected,changes=resolve_spatial_family(corrected,document,output)
            candidate=compile_body(corrected.bodies[0],corrected)
            a=evaluate(candidate,folder/'cad-feedback/build');accept(candidate,a,'C')
            diagnostics['cad_feedback_repair']={'passed':a['passed'],'spatial_changes':changes}
        except Exception as exc:diagnostics['history'].append({'stage':'cad_feedback_repair','error':str(exc)})
    if winner and plan and time.monotonic()<deadline:
        try:
            completion=CADFeatureCompletion(document,output,evaluate,deadline)
            winner,winner_attempt,log=completion.apply(plan,winner,winner_attempt,folder/'completion')
            diagnostics['feature_completion']=log
        except Exception as exc:diagnostics['history'].append({'stage':'completion','error':str(exc)})
    report=copy.deepcopy(classic_report)
    if winner:
        intent=winner['intent'];diagnostics.update(selected_build=winner_attempt['directory'],cad_validated=True)
        # Revalidate every registry read; store only actual checked native results.
        if winner['origin']!='best_candidate_registry' or any(x.get('selected') for x in diagnostics.get('feature_completion',[])):
            registry.store(document['sha256'],intent,winner_attempt['directory'],diagnostics['quality'],settings.model,ENGINE_VERSION)
        atomic_json(output/'cad-intent.json',intent)
        report.update(status=intent['reconstruction_status'],message='Основная CAD сохранена; неподтверждённые элементы перечислены в диагностике.')
    else:
        intent={};atomic_json(output/'cad-intent.json',{})
        if diagnostics.get('visual_preview'):
            diagnostics.update(quality='D',selected_origin='visual_image_fallback');report.update(status='visual_preview',message='Приблизительная 3D-форма по изображению; не параметрическая CAD.')
        else:report.update(status='failed',message='Не найдена осмысленная форма, подтверждённая изображением.')
    report['reconstruction']={'quality':diagnostics['quality'],'parametric_cad':bool(winner),'visual_preview':bool(diagnostics.get('visual_preview'))}
    atomic_json(output/'cad-candidates.json',{'alternatives':[],'policy':'Revalidated best native model, independent visual fallback'})
    atomic_json(output/'interpretation.json',report);save()
    return report,intent,diagnostics
