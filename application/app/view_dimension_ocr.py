"""Dynamic high-resolution OCR bands around independently located views."""
import cv2
import numpy as np
from .ocr import recognize
from .dimension_parser import parse_dimensions
from .annotation_context import filter_annotation_context
from .layout import associate_dimensions
from .storage import atomic_json
from .local_dimension_ocr import reread_dimension


def discover_view_dimensions(page,view,output,target):
    image=cv2.imdecode(np.fromfile(output/f"page-{page['page']}"/'original.png',np.uint8),1)
    if image is None:return []
    box=np.array(view.bbox)*[page['width'],page['height'],page['width'],page['height']]
    x0,y0,x1,y1=box;pad=min(90,max(35,max(x1-x0,y1-y0)*.35))
    boxes=[[x0-pad,y0,x0+2,y1],[x1-2,y0,x1+pad,y1],
           [x0,y0-pad,x1,y0+2],[x0,y1-2,x1,y1+pad],[x0,y0,x1,y1]]
    dimensions=[];manifest=[];scale=np.array([image.shape[1]/page['width'],image.shape[0]/page['height']])
    for index,proposed in enumerate(boxes):
        b=np.maximum(proposed,[page['width']*.08,page['height']*.035,0,0])
        b[2:]=np.minimum(b[2:],[page['width']*.97,page['height']*.78])
        lo=np.floor(b[:2]*scale).astype(int);hi=np.ceil(b[2:]*scale).astype(int)
        crop=image[lo[1]:hi[1],lo[0]:hi[0]].copy()
        if not crop.size:continue
        # Remove long dimension/extension strokes only on this OCR copy.
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);ink=(gray<180).astype(np.uint8)*255
        length=max(70,round(max(ink.shape)*.35))
        lines=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,length),np.uint8))
        lines|=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((length,1),np.uint8))
        crop[cv2.dilate(lines,np.ones((2,2),np.uint8))>0]=255
        folder=target/f'band-{index+1}'
        runs,metadata=recognize(crop,b[2]-b[0],b[3]-b[1],folder,refine=False)
        for run in runs:
            run.id=f"LOCAL_P{page['page']}_B{index+1}_{run.id}"
            run.bbox=tuple(np.array(run.bbox)+[b[0],b[1],b[0],b[1]])
            run.recognition.update(method='dynamic_view_band_original_resolution',source_bbox_pdf=b.tolist())
        found,_=parse_dimensions(runs)
        found,_=filter_annotation_context(found,page['width'],page['height'],page.get('text_runs',[]))
        for d in found:
            if d.type=='roughness':continue
            if d.type=='linear' and any(c.isalpha() for c in d.raw_text):continue
            center=(np.array(d.bbox[:2])+d.bbox[2:])/2
            if any(existing['type'] in ('diameter','radius') and d.type=='linear' and
                   np.linalg.norm((np.array(existing['bbox'][:2])+existing['bbox'][2:])/2-center)<4 for existing in page['dimensions']):continue
            if any(existing['type']==d.type and abs(existing['nominal_value']-d.nominal_value)<1e-6 and
                   np.linalg.norm((np.array(existing['bbox'][:2])+existing['bbox'][2:])/2-center)<3 for existing in page['dimensions']):continue
            if any(e.type==d.type and abs(e.nominal_value-d.nominal_value)<1e-6 and np.linalg.norm((np.array(e.bbox[:2])+e.bbox[2:])/2-center)<3 for e in dimensions):continue
            d.id=f"NEW_P{page['page']}_{len(dimensions)+1:03d}"
            if d.direction[1] or d.nominal_value>500:
                reread=reread_dimension(page,output,d.bbox,target/'local-rechecks'/d.id)
                if len(reread['consensus_values'])==1:
                    value=reread['consensus_values'][0]
                    d.alternatives.append({'original_reading':d.raw_text,'original_value':d.nominal_value,'recheck':reread,
                                           'action':'local_ocr_italic_shear_refinement'})
                    d.nominal_value=value;d.normalized_text=str(value)
                    reading=next((r['raw_text'] for r in reread['reads'] if r.get('eligible_numeric_vote') and r['values']==[value]),None)
                    if reading:d.raw_text=reading
            dimensions.append(d)
        manifest.append({'band':index+1,'bbox_pdf':b.tolist(),'ocr':metadata,'runs':[r.model_dump(mode='json') for r in runs]})
    associate_dimensions(dimensions,page['layout'])
    result=[d.model_dump(mode='json') for d in dimensions]
    atomic_json(target/'discovered-dimensions.json',{'dimensions':result,'bands':manifest,'origin':'dynamic_view_band_tesseract_not_llm_numeric_guess'})
    return result
