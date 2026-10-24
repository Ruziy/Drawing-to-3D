"""Local OCR with reversible coordinates, raw evidence and explicit engine scores."""
import csv
import io
import os
from pathlib import Path
import shutil
import subprocess
import re

import cv2
import numpy as np

from .config import ROOT
from .schemas import TextRun


def text_image(page,rgb):
    """An OCR-only copy. Filled vector glyphs retain exact shapes, not PDF encoding."""
    import pymupdf as fitz
    paths=page.get_drawings()
    filled=[p for p in paths if p.get('fill') is not None and p['rect'].width<80 and p['rect'].height<80]
    if len(filled)>30 and not page.get_text().strip():
        doc=fitz.open();out=doc.new_page(width=page.rect.width,height=page.rect.height)
        for path in filled:
            shape=out.new_shape()
            for item in path['items']:
                if item[0]=='l':shape.draw_line(item[1],item[2])
                elif item[0]=='c':shape.draw_bezier(*item[1:])
                elif item[0]=='re':shape.draw_rect(item[1])
                elif item[0]=='qu':shape.draw_quad(item[1])
            shape.finish(fill=path['fill'],color=path.get('color'),width=path.get('width') or 0,
                         closePath=path.get('closePath',False),even_odd=path.get('even_odd',True))
            shape.commit()
        pix=out.get_pixmap(dpi=300,alpha=False)
        result=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3).copy()
        doc.close()
        return result,'isolated_filled_vector_glyphs'
    if not page.get_text().strip():
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        ink=(gray<180).astype(np.uint8)*255
        length=max(70,round(max(ink.shape)*.025))
        lines=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,length),np.uint8))
        lines|=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((length,1),np.uint8))
        work=rgb.copy();work[cv2.dilate(lines,np.ones((2,2),np.uint8))>0]=255
        return work,'long_orthogonal_lines_removed_copy'
    return rgb,'original_render_pdf_text_primary'


def find_tesseract():
    candidates = [os.environ.get('TESSERACT_CMD',''),shutil.which('tesseract') or '',
                  r'C:\Program Files\Tesseract-OCR\tesseract.exe']
    return next((str(Path(p).resolve()) for p in candidates if p and Path(p).is_file()),None)


def tessdata():
    return Path(os.environ.get('DRAWING_TESSDATA', ROOT/'.runtime'/'tessdata'))


def available():
    return bool(find_tesseract() and all((tessdata()/f'{l}.traineddata').is_file() for l in ('eng','rus')))


def inverse_point(x,y,rotation,width,height):
    # np.rot90 rotates counterclockwise; all boxes return to the original raster.
    if rotation==90:
        return width-y,x
    if rotation==180:
        return width-x,height-y
    if rotation==270:
        return y,height-x
    return x,y


def overlap(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    return ix*iy/max(1e-9,min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])))


def recognize(rgb, pdf_width, pdf_height, directory:Path, pdf_runs=(), refine=True):
    if not available():
        return [],{'available':False,'warning':'Tesseract или локальные модели eng/rus не найдены; см. setup-ocr.ps1'}
    directory.mkdir(parents=True,exist_ok=True)
    h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    runs=[];passes=[]
    env={**os.environ,'OMP_THREAD_LIMIT':'1'}
    orientations=(0,90,270)
    for rotation in orientations:
        image=np.ascontiguousarray(np.rot90(gray,rotation//90))
        # No modifications to original image: OCR input stored separately.
        image_path=directory/f'input-{rotation}.png'
        cv2.imencode('.png',image)[1].tofile(str(image_path))
        command=[find_tesseract(),str(image_path),'stdout','--tessdata-dir',str(tessdata()),
                 '-l','eng+rus','--oem','1','--psm','11','--dpi','300',
                 '-c','load_system_dawg=0','-c','load_freq_dawg=0','-c','tessedit_create_tsv=1']
        options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100,env=env,**options)
        output=result.stdout.decode('utf-8',errors='replace')
        (directory/f'raw-{rotation}.tsv').write_text(output,encoding='utf-8')
        (directory/f'raw-{rotation}.log').write_bytes(result.stderr)
        if result.returncode:
            passes.append({'rotation':rotation,'error':result.stderr.decode('utf-8',errors='replace')});continue
        grouped={}
        for word in csv.DictReader(io.StringIO(output),delimiter='\t',quoting=csv.QUOTE_NONE):
            if word.get('level')!='5' or not word.get('text','').strip():continue
            if float(word['conf']) < 15:continue
            key=(word['block_num'],word['par_num'],word['line_num'])
            grouped.setdefault(key,[]).append(word)
        found=0
        for key,words in grouped.items():
            text=' '.join(word['text'] for word in words)
            x0=min(int(a['left']) for a in words);y0=min(int(a['top']) for a in words)
            x1=max(int(a['left'])+int(a['width']) for a in words);y1=max(int(a['top'])+int(a['height']) for a in words)
            points=[inverse_point(x,y,rotation,w,h) for x,y in ((x0,y0),(x1,y0),(x0,y1),(x1,y1))]
            box=(min(p[0] for p in points)*pdf_width/w,min(p[1] for p in points)*pdf_height/h,
                 max(p[0] for p in points)*pdf_width/w,max(p[1] for p in points)*pdf_height/h)
            direction={0:(1,0),90:(0,1),270:(0,-1)}[rotation]
            confidence=sum(float(a['conf'])*len(a['text']) for a in words)/max(1,sum(len(a['text']) for a in words))/100
            # Keep OCR even when overlapping PDF text, but flag as an independent alternative.
            suppressed=any(overlap(box,r.bbox)>.55 for r in pdf_runs)
            runs.append(TextRun(id=f'O{rotation}-{found+1:04d}',raw_text=text,bbox=box,block_id=f'OB{rotation}-{key[0]}',
                font='Tesseract LSTM eng+rus',size=(y1-y0)*pdf_height/h,direction=direction,source='ocr_tesseract',
                confidence=round(confidence,4),recognition={'rotation':rotation,'engine':'tesseract_lstm',
                'psm':11,'overlaps_pdf_text':suppressed,'word_evidence':words,
                'confidence_semantics':'uncalibrated_engine_word_score_not_probability'}))
            found+=1
        passes.append({'rotation':rotation,'text_lines':found,'psm':11})
    if not pdf_runs and refine:
        runs.extend(discover_diameters(gray,pdf_width,pdf_height,directory,env,runs))
        refine_annotations(gray,runs,directory,pdf_width,pdf_height,env)
        refine_punctuation(gray,runs,pdf_width,pdf_height)
    return runs,{'available':True,'engine':'Tesseract LSTM','languages':['eng','rus'],'passes':passes,
                 'raw_evidence_directory':'ocr','confidence_calibrated':False}


def diameter_component(ink,component):
    x,y,w,h,area=component
    if h<12 or w<8 or not .35<w/h<1.6:return False
    contours,hierarchy=cv2.findContours(ink[y:y+h,x:x+w],cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return False
    holes=[c for c,hier in zip(contours,hierarchy[0]) if hier[3]>=0 and cv2.contourArea(c)>w*h*.025]
    if len(holes)!=2:return False
    centers=[]
    for c in holes:
        m=cv2.moments(c);centers.append((m['m10']/max(m['m00'],1),m['m01']/max(m['m00'],1)))
    # Engineering italic Ø has two *lateral* holes. Their vertical centroids
    # can almost coincide; the old 12%-height test discarded real diameters.
    # An italic 8 has vertically stacked holes, so require lateral separation.
    dx=centers[0][0]-centers[1][0];dy=centers[0][1]-centers[1][1]
    # For the / diameter stroke, the two counter centres lie on the opposite
    # diagonal (top-left / bottom-right). Italic 8 counters lean the other way.
    # Both upright and italic engineering diameter fonts must be accepted.
    return abs(dx)>w*.20 and abs(dy)<h*.60 and dx*dy>=-w*h*.015


def observed_punctuation(text,image):
    """Restore only punctuation evidenced by connected pixels, never digit guesses."""
    ink=(image<175).astype(np.uint8)*255
    _,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
    comps=sorted([tuple(int(v) for v in p) for p in stats[1:] if p[4]>=4],key=lambda p:p[0])
    if not comps:return text,[]
    height=max(p[3] for p in comps)
    def cross(p):
        x,y,w,h,_=p
        if not .5<w/max(h,1)<1.8:return False
        a=ink[y:y+h,x:x+w]>0;rows=a.sum(axis=1);columns=a.sum(axis=0)
        peak=int(rows.argmax())/max(h,1)
        return .2<peak<.7 and rows.max()>w*.75 and columns.max()>h*.55
    crosses=[p for p in comps if cross(p)]
    major=[p for p in comps if p[3]>=height*.52 and p not in crosses]
    letters=[i for i,c in enumerate(text) if c.isalnum() or c in ('Ø','⌀')]
    evidence=[];result=text
    # A punctuation component must lie between two OCR characters, at their
    # baseline. Character count agreement prevents positional guessing.
    if len(major)==len(letters) and len(major)>1:
        insertions=[]
        for i,(a,b) in enumerate(zip(major,major[1:])):
            if not text[letters[i]].isdigit() or not text[letters[i+1]].isdigit():continue
            if any(c in text[letters[i]+1:letters[i+1]] for c in ',.'):continue
            baseline=max(a[1]+a[3],b[1]+b[3])
            # Italic glyph bounding boxes overlap: a comma can lie under the
            # right overhang of the preceding digit. Use glyph centres.
            small=[p for p in comps if a[0]+a[2]*.5<=p[0]+p[2]/2<=b[0]+b[2]*.5 and
                2<=p[3]<=height*.35 and .12<p[2]/max(p[3],1)<1.8 and
                baseline-height*.25<=p[1]+p[3]/2<=baseline+height*.25]
            if len(small)==1:
                insertions.append(letters[i]+1);evidence.append({'type':'decimal_separator','component_bbox_px':list(small[0][:4]),'after_character_index':letters[i]})
        for index in reversed(insertions):result=result[:index]+','+result[index:]
    # Plus/minus is identified only when a second horizontal bar is visible.
    if any(c in result for c in ('+','*','£')) and len(crosses)==1:
        x,y,w,h,_=crosses[0]
        bars=[p for p in comps if p[1]>=y+h and p[1]-(y+h)<height*.28 and
              p[3]<h*.3 and p[2]>w*.55 and abs((p[0]+p[2]/2)-(x+w/2))<w*.25]
        if len(bars)==1:
            result=re.sub(r'[+*£]','±',result,count=1);evidence.append({'type':'plus_minus','cross_bbox_px':list(crosses[0][:4]),'lower_bar_bbox_px':list(bars[0][:4])})
    return result,evidence


def refine_punctuation(gray,runs,pdf_width,pdf_height):
    h,w=gray.shape
    for run in runs:
        raw=run.raw_text
        if len(raw)>30 or not any(c.isdigit() for c in raw):continue
        if run.recognition.get('structured_text'):continue
        if not __import__('re').match(r'^(?:[MМ]\d|\d)',raw):continue
        x0,y0,x1,y1=run.bbox
        a,b,c,d=[round(v) for v in (x0/pdf_width*w,y0/pdf_height*h,x1/pdf_width*w,y1/pdf_height*h)]
        crop=gray[max(0,b-3):min(h,d+3),max(0,a-3):min(w,c+3)]
        crop=np.ascontiguousarray(np.rot90(crop,run.recognition['rotation']//90))
        revised,evidence=observed_punctuation(raw,crop)
        if evidence:
            run.recognition['structured_text']=revised
            run.recognition['punctuation_image_evidence']=evidence
            run.recognition['original_ocr_text']=raw


def discover_diameters(gray,pdf_width,pdf_height,directory,env,existing):
    """CV discovers diameter glyphs even if the page OCR omitted the whole label."""
    h,w=gray.shape;found=[]
    for rotation in (0,270):
        work=np.ascontiguousarray(np.rot90(gray,rotation//90));ink=(work<175).astype(np.uint8)*255
        _,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
        components=[tuple(int(v) for v in row) for row in stats[1:] if row[4]>=12]
        for component in components:
            if not diameter_component(ink,component):continue
            x,y,cw,ch,_=component
            neighbors=sorted([p for p in components if p[0]>x+cw*.65 and p[0]<x+cw+ch*5 and
                abs((p[1]+p[3]/2)-(y+ch/2))<ch*.5 and p[3]>ch*.38],key=lambda p:p[0])
            if not neighbors:continue
            group=[];last=x+cw
            for p in neighbors:
                if p[0]-last>ch*.5:break
                group.append(p);last=p[0]+p[2]
            if not group:continue
            x1=last;y0=min(y,min(p[1] for p in group));y1=max(y+ch,max(p[1]+p[3] for p in group))
            points=[inverse_point(a,b,rotation,w,h) for a,b in ((x,y0),(x1,y1))]
            bbox=(min(p[0] for p in points)*pdf_width/w,min(p[1] for p in points)*pdf_height/h,
                  max(p[0] for p in points)*pdf_width/w,max(p[1] for p in points)*pdf_height/h)
            if bbox[1]>pdf_height*.70:continue
            crop=cv2.copyMakeBorder(work[max(0,y0-3):y1+3,max(0,x-3):x1+3],12,12,12,12,cv2.BORDER_CONSTANT,value=255)
            path=directory/f'discovered-{rotation}-{len(found)}.png';cv2.imencode('.png',crop)[1].tofile(str(path))
            command=[find_tesseract(),str(path),'stdout','--tessdata-dir',str(tessdata()),'-l','eng','--psm','7','-c','tessedit_create_tsv=1']
            options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
            result=subprocess.run(command,capture_output=True,env=env,timeout=15,**options)
            tsv=result.stdout.decode('utf-8',errors='replace');(directory/f'discovered-{rotation}-{len(found)}.tsv').write_text(tsv,encoding='utf-8')
            words=[row for row in csv.DictReader(io.StringIO(tsv),delimiter='\t',quoting=csv.QUOTE_NONE) if row.get('level')=='5']
            raw=' '.join(row.get('text','') for row in words)
            if not any(c.isdigit() for c in raw):continue
            confidence=sum(max(0,float(row['conf'])) for row in words)/max(1,len(words))/100
            run=TextRun(id=f'OCV{len(found)+1:04d}',raw_text=raw,bbox=bbox,block_id=f'OCVB{len(found)+1}',
                font='Tesseract targeted glyph ROI',size=ch*pdf_height/h,direction=(1,0) if rotation==0 else (0,-1),
                source='ocr_tesseract',confidence=round(confidence,4),recognition={'rotation':rotation,'engine':'tesseract_lstm',
                'psm':7,'word_evidence':words,'overlaps_pdf_text':False,'discovery':'split_circle_glyph_cv',
                'confidence_semantics':'uncalibrated_engine_score'})
            found.append(run)
    return found


def refine_annotations(gray,runs,directory,pdf_width,pdf_height,env):
    """Read nominal and superscripts separately, without correcting numeric guesses."""
    h,w=gray.shape
    for i,run in enumerate(runs):
        if run.recognition['rotation'] not in (0,270) or not re.search(r'\d',run.raw_text):continue
        if len(run.raw_text)>35 or run.bbox[1]>pdf_height*.72:continue
        x0,y0,x1,y1=run.bbox
        a,b,c,d=[round(v) for v in (x0/pdf_width*w,y0/pdf_height*h,x1/pdf_width*w,y1/pdf_height*h)]
        crop=gray[max(0,b-3):min(h,d+3),max(0,a-3):min(w,c+3)]
        crop=np.ascontiguousarray(np.rot90(crop,run.recognition['rotation']//90))
        if crop.size==0:continue
        ink=(crop<175).astype(np.uint8)*255
        _,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
        comps=sorted([tuple(int(v) for v in p) for p in stats[1:] if p[4]>=5],key=lambda p:p[0])
        if not comps:continue
        # Long dimension lines remaining in a crop are excluded, not interpreted as characters.
        comps=[p for p in comps if p[2]<crop.shape[1]*.9 or p[3]>crop.shape[0]*.35]
        if not comps:continue
        dia=diameter_component(ink,comps[0])
        if not dia:continue
        first=comps[0]
        # Italic Ø overhangs the next digit's bounding box. Requiring its
        # entire box to end first silently dropped the first 6 in Ø6,6.
        after=first[0]+first[2]*.65 if dia else 0
        def plus_component(p):
            x,y,cw,ch,_=p
            if not .55<cw/max(ch,1)<1.7:return False
            component=ink[y:y+ch,x:x+cw]>0
            row=component.sum(axis=1);col=component.sum(axis=0)
            peak=int(row.argmax())/max(ch,1)
            return .25<peak<.65 and row.max()>cw*.75 and col.max()>ch*.55
        stop=min((p[0] for p in comps if p[0]>=after and plus_component(p)),default=crop.shape[1]+1)
        suffix_components=[p for p in comps if p!=first and p[0]>=after]
        if not suffix_components:continue
        glyph_height=max(p[3] for p in suffix_components)
        # Ø is deliberately taller than digits in several CAD fonts. Compare
        # nominal/deviation sizes to the suffix, not to the diameter symbol.
        candidates=[p for p in suffix_components if p[3]>=glyph_height*.78]
        candidates=[p for p in candidates if p[0]<stop]
        if not candidates:continue
        # Do not reinterpret R/M/Ra or a letter prefix as a bare linear size.
        left=min(p[0] for p in candidates);right=max(p[0]+p[2] for p in candidates)
        top=min(p[1] for p in candidates);bottom=max(p[1]+p[3] for p in candidates)
        def read_part(image,label,whitelist,shear=0,enlarge=False):
            if shear:
                ih,iw=image.shape
                matrix=np.float32([[1,shear,0],[0,1,0]])
                image=cv2.warpAffine(image,matrix,(round(iw+shear*ih),ih),borderValue=255)
            image=cv2.copyMakeBorder(image,12,12,12,12,cv2.BORDER_CONSTANT,value=255)
            # The engineering font is thin and italic. At native 300 DPI the
            # engine drops narrow leading 1s; keep a documented 2x ROI reading.
            if enlarge or image.shape[0]<65:image=cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            path=directory/f'refine-{i}-{label}.png';cv2.imencode('.png',image)[1].tofile(str(path))
            cmd=[find_tesseract(),str(path),'stdout','--tessdata-dir',str(tessdata()),'-l','eng','--psm','7',
                 '-c','tessedit_char_whitelist='+whitelist,'-c','load_system_dawg=0','-c','load_freq_dawg=0']
            options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
            result=subprocess.run(cmd,capture_output=True,env=env,timeout=15,**options)
            text=result.stdout.decode('utf-8',errors='replace').strip()
            (directory/f'refine-{i}-{label}.txt').write_text(text,encoding='utf-8')
            return text
        nominal_image=crop[max(0,top-2):bottom+2,max(0,left-2):right+2]
        # A dropped digit must not become a new nominal. Compare observed tall
        # glyphs with the complete alphanumeric reading (fit included).
        tall=[p for p in candidates if p[3]>=max(p[3] for p in candidates)*.55]
        alphabet='0123456789.,HhABCDEFGJKLMNPRSTUVWXYZabcdefgmnprstuvxyz'
        reading=read_part(nominal_image,'nominal','0123456789.,')
        if len(tall)!=sum(c.isalnum() for c in reading):
            reading=read_part(nominal_image,'nominal-deslanted',alphabet,shear=.30,enlarge=True)
        elif re.search(r'[HhА-Яа-яA-Ca-cf-g]',run.raw_text):
            fitted=read_part(nominal_image,'nominal-with-fit',alphabet)
            if re.fullmatch(r'\d+(?:[.,]\d+)?[A-Za-z]{1,2}\d{1,2}',fitted) and len(tall)==sum(c.isalnum() for c in fitted):reading=fitted
        reading,punctuation=observed_punctuation(reading,nominal_image)
        match=re.fullmatch(r'(\d+(?:[.,]\d+)?)([A-Za-z]{1,2}\d{1,2})?',reading)
        if not match or len(tall)!=sum(c.isalnum() for c in reading):continue
        nominal=match[1]
        text=('Ø' if dia else '')+reading
        tolerance='';tol_original='';tol_evidence=[]
        if right+3<crop.shape[1] and np.count_nonzero(ink[:,right+3:])>5:
            tolerance=read_part(crop[:,right+3:],'deviation','0123456789.,+-±')
            tol_original=tolerance
            tolerance,tol_evidence=observed_punctuation(tolerance,crop[:,right+3:])
        run.recognition['segmented_reading']={'nominal':nominal,'diameter_symbol':bool(dia),
            'deviation_raw':tol_original,'deviation_reading':tolerance,'punctuation_image_evidence':tol_evidence,
            'method':'connected_glyph_sizes_and_split_circle_prefix',
            'raw_page_reading':run.raw_text,'numerical_correction_applied':False}
        run.recognition['segmented_reading']['nominal_punctuation_evidence']=punctuation
        run.recognition['structured_text']=text+(' ('+tolerance+')' if re.fullmatch(r'[+-]\d+[.,]\d+',tolerance) else '')
        run.recognition['structured_nominal_only']=True


def merge_runs(pdf_runs,ocr_runs):
    # Only one primary reading in overlapping locations; retain alternatives separately.
    accepted=[]
    alternatives=[]
    for run in pdf_runs:
        duplicate=next((p for p in accepted if p.raw_text==run.raw_text and overlap(p.bbox,run.bbox)>.95),None)
        if duplicate:
            alternatives.append({'primary_text_run_id':duplicate.id,'alternative_text_run_id':run.id,
                'raw_text':run.raw_text,'bbox':run.bbox,'reason':'duplicate_pdf_text_paint'})
        else:accepted.append(run)
    def order(run):
        structured=run.recognition.get('structured_text','')
        return (not bool(structured),not any(c in structured for c in ('(','±')), -(run.confidence or 0))
    for run in sorted(ocr_runs,key=order):
        collision=next((r for r in accepted if abs(r.direction[0]*run.direction[0]+r.direction[1]*run.direction[1])>.95 and overlap(run.bbox,r.bbox)>.55),None)
        if collision:
            alternatives.append({'primary_text_run_id':collision.id,'alternative_text_run_id':run.id,
                                 'raw_text':run.raw_text,'primary_text':collision.raw_text,'bbox':run.bbox,
                                 'confidence':run.confidence,'reason':'overlapping_independent_reading'})
        else:
            accepted.append(run)
    return accepted,alternatives
