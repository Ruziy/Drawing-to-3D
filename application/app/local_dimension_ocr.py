"""Targeted independent re-read. Never feed the desired value to Tesseract."""
from collections import Counter
from pathlib import Path
import re
import subprocess
import os
import cv2
import numpy as np
from .ocr import find_tesseract,tessdata
from .storage import atomic_json


def reread_dimension(page,output,bbox,target):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    image=cv2.imdecode(np.fromfile(Path(output)/f"page-{page['page']}"/'original.png',np.uint8),0)
    if image is None or not find_tesseract():return {'consensus_values':[],'reads':[]}
    sx,sy=image.shape[1]/page['width'],image.shape[0]/page['height']
    x0,y0,x1,y1=bbox
    crop=image[max(0,int((y0-2)*sy)):min(image.shape[0],int((y1+2)*sy)),max(0,int((x0-2)*sx)):min(image.shape[1],int((x1+2)*sx))]
    if not crop.size:return {'consensus_values':[],'reads':[]}
    reads=[];votes=Counter()
    rotations=(90,270) if crop.shape[0]>crop.shape[1]*1.4 else (0,)
    env={**os.environ,'OMP_THREAD_LIMIT':'1'}
    for rotation in rotations:
        rotated=np.rot90(crop,rotation//90).copy()
        enlarged=cv2.resize(rotated,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        for shear,mode in [(s,m) for s in (0,.25,.4) for m in ('gray','otsu')]:
            h,w=enlarged.shape
            deskewed=cv2.warpAffine(enlarged,np.array([[1,shear,0],[0,1,0]],np.float32),(int(w+shear*h),h),borderValue=255) if shear else enlarged
            work=deskewed if mode=='gray' else cv2.threshold(deskewed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            work=cv2.copyMakeBorder(work,14,14,14,14,cv2.BORDER_CONSTANT,value=255)
            path=target/f'{rotation}-{shear}-{mode}.png';cv2.imencode('.png',work)[1].tofile(str(path))
            command=[find_tesseract(),str(path),'stdout','--tessdata-dir',str(tessdata()),'-l','eng','--oem','1','--psm','7',
                     '-c','load_system_dawg=0','-c','load_freq_dawg=0']
            options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
            process=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=25,**options)
            raw=process.stdout.decode('utf-8',errors='replace').strip()
            values={float(s.replace(',','.')) for s in re.findall(r'(?<![\d.,])\d+(?:[.,]\d+)?(?![\d.,])',raw)}
            eligible=raw.count('(')==raw.count(')') and '/' not in raw
            if eligible:
                for value in values:votes[value]+=1
            reads.append({'rotation':rotation,'mode':mode,'italic_shear':shear,'raw_text':raw,'values':sorted(values),'eligible_numeric_vote':eligible,'exit_code':process.returncode,'file':str(path)})
    maximum=max(votes.values(),default=0)
    result={'bbox_pdf':list(bbox),'consensus_values':[v for v,n in votes.items() if n>=2 and n==maximum],
            'value_votes':dict(votes),'reads':reads,'method':'unbiased_original_crop_tesseract_orientation_italic_shear_consensus'}
    atomic_json(target/'reads.json',result);return result
