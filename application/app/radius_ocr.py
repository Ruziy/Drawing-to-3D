"""Re-read ambiguous engineering R glyphs; retain every original OCR record."""
from collections import Counter
import re
import numpy as np
from .local_dimension_ocr import reread_dimension
from .dimension_parser import parse_dimensions
from .storage import atomic_json


def recover_radii(page,runs,dimensions,output,target):
    recovered=[];new_runs=[];diagnostics=[]
    for run in runs:
        if not re.match(r'^\s*[КKРPRr]\s*[З3\d]',run.raw_text):continue
        center=(np.array(run.bbox[:2])+run.bbox[2:])/2
        if any(d.type=='radius' and np.linalg.norm((np.array(d.bbox[:2])+d.bbox[2:])/2-center)<3 for d in dimensions):continue
        reread=reread_dimension(page,output,run.bbox,target/run.id.replace(':','_'))
        choices=[]
        for index,reading in enumerate(reread['reads']):
            candidate=run.model_copy(update={'raw_text':reading['raw_text'],'id':f'REREAD_{run.id}_{index}',
                                            'source':'ocr_tesseract','font':'Tesseract LSTM','confidence':None,'recognition':{'method':'local_Latin_engineering_glyph_reread'}})
            parsed,_=parse_dimensions([candidate])
            for dimension in parsed:
                if dimension.type=='radius':choices.append((dimension,candidate))
        votes=Counter(d.nominal_value for d,_ in choices)
        if not votes:continue
        value,count=votes.most_common(1)[0]
        if count<2 or sum(n==count for n in votes.values())>1:continue
        dimension,candidate=next((d,r) for d,r in choices if d.nominal_value==value)
        dimension.id=f"RECOVERED_R_P{page['page']}_{len(recovered)+1:03d}"
        dimension.alternatives.append({'original_raw_text':run.raw_text,'method':'independent_local_OCR_reread','evidence':reread})
        recovered.append(dimension);new_runs.append(candidate)
        diagnostics.append({'original_run':run.id,'dimension':dimension.model_dump(mode='json'),'reread':reread})
    atomic_json(target/'radius-corrections.json',diagnostics)
    return recovered,new_runs
