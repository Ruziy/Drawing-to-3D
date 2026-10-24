from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import numpy as np
import pymupdf as fitz

from .config import DPI
from .dimension_parser import parse_dimensions
from .image_geometry import find_regions, raster_geometry, vector_geometry
from .pdf_reader import embedded_images, text_content, validate_document
from .schemas import DrawingAnalysis, PageAnalysis,Dimension
from .storage import atomic_json
from .ocr import recognize, merge_runs, text_image
from .layout import analyze_layout, associate_dimensions
from .reconstruction import reconstruct


def extract_pdf(path: Path, output: Path, drawing_id: str, filename: str, sha256: str):
    validate_document(path)
    start = time.perf_counter()
    pages = []
    with fitz.open(path) as document:
        for number,page in enumerate(document,1):
            begun = time.perf_counter()
            rotation = page.rotation
            page.set_rotation(0)
            width,height = page.rect.width,page.rect.height
            directory = output / f"page-{number}"
            directory.mkdir(parents=True,exist_ok=True)
            url = f"/api/artifacts/{drawing_id}/page-{number}"
            pix = page.get_pixmap(dpi=DPI,alpha=False,colorspace=fitz.csRGB)
            pix.save(directory/"original.png")
            rgb = np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3).copy()
            pdf_runs,blocks = text_content(page)
            ocr_input,preprocessing=text_image(page,rgb)
            ocr_runs,ocr_info=recognize(ocr_input,width,height,directory/'ocr',pdf_runs)
            ocr_info['preprocessing']=preprocessing
            ocr_info['evidence_url']=f'{url}/ocr.json'
            runs,ocr_alternatives=merge_runs(pdf_runs,ocr_runs)
            atomic_json(directory/'ocr.json',{'metadata':ocr_info,'runs':[r.model_dump(mode='json') for r in ocr_runs],
                        'alternatives':ocr_alternatives})
            vector = vector_geometry(page)
            raster,binary,diagnostics = raster_geometry(rgb,width,height)
            views = find_regions(binary,runs,width,height)
            for region in views:
                x0,y0,x1,y1 = region.bbox
                crop = rgb[max(0,int(y0/height*pix.height)):min(pix.height,int(y1/height*pix.height)+1),
                           max(0,int(x0/width*pix.width)):min(pix.width,int(x1/width*pix.width)+1)]
                if crop.size:
                    cv2.imencode(".png",cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))[1].tofile(str(directory/f"{region.id}.png"))
                    region.crop_url = f"{url}/{region.id}.png"
            excluded = [r.bbox for r in views if r.type == "title_block"]
            # Long prose blocks are technical notes, not dimension annotations.
            excluded += [tuple(b["bbox"]) for b in blocks if any(len(line)>40 and sum(c.isalpha() for c in line)>20 for line in b["raw_text"].splitlines())]
            dims,ambiguous = parse_dimensions(runs,excluded)
            from .radius_ocr import recover_radii
            recovered,recovered_runs=recover_radii({'page':number,'width':width,'height':height},runs,dims,output,directory/'radius-rechecks')
            dims+=recovered;runs+=recovered_runs
            from .annotation_context import filter_annotation_context
            dims,rejected_annotations=filter_annotation_context(dims,width,height,runs)
            diagnostics['rejected_annotation_candidates']=rejected_annotations
            has_text = bool(pdf_runs)
            has_vector = any(vector.values())
            image_records = embedded_images(page,directory,url)
            warnings = []
            if not has_text:
                warnings.append("Текстовый слой отсутствует. Использован локальный OCR; распознанные значения требуют геометрической проверки.")
            if not ocr_info['available']:
                warnings.append(ocr_info['warning'])
            ambiguous.extend(ocr_alternatives)
            if ambiguous:
                warnings.append("Есть неоднозначные надписи. Проверьте ambiguous_elements и исходное изображение.")
            if diagnostics["truncated"]:
                warnings.append("Часть растровых кандидатов отсечена лимитом; точные векторные пути сохранены отдельно.")
            warnings.append("Примитивы включают рамку, штриховку и символы. Подтверждённые связи выбранной геометрии перечислены в Geometry Interpretation.")
            geometry = {k:vector[k]+raster[k] for k in vector}
            # Avoid ID collisions across pages while preserving source links.
            prefix = f"P{number}-"
            for elements in geometry.values():
                for element in elements:
                    element.id = prefix+element.id
                    if "line_ids" in element.properties:
                        element.properties["line_ids"] = [prefix+x for x in element.properties["line_ids"]]
            layout=analyze_layout(page,geometry,runs,rgb,directory)
            from .non_geometry import title_regions
            non_geometry=title_regions({'width':width,'height':height,'geometry':geometry,'text_runs':runs,'layout':layout})
            layout['non_geometry_regions']=non_geometry
            protected=[]
            for d in dims:
                cx,cy=(d.bbox[0]+d.bbox[2])/2,(d.bbox[1]+d.bbox[3])/2
                if any(r['bbox'][0]<=cx<=r['bbox'][2] and r['bbox'][1]<=cy<=r['bbox'][3] for r in non_geometry):
                    diagnostics['rejected_annotation_candidates'].append({**d.model_dump(mode='json'),'exclusion_reason':'confirmed_title_table'})
                else:protected.append(d)
            dims=protected
            associate_dimensions(dims,layout)
            atomic_json(directory/'layout.json',layout)
            # Overlay is a separate file. Green: text dimensions, blue: vector/raster lines, orange: unknown ROIs.
            overlay = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            sx,sy = pix.width/width,pix.height/height
            def box(b,color,thickness=2):
                cv2.rectangle(overlay,(int(b[0]*sx),int(b[1]*sy)),(int(b[2]*sx),int(b[3]*sy)),color,thickness)
            for line in geometry["lines"]:
                a,b = line.coordinates
                cv2.line(overlay,(int(a[0]*sx),int(a[1]*sy)),(int(b[0]*sx),int(b[1]*sy)),(220,130,40),1)
            for dim in dims:
                box(dim.bbox,(40,170,30),3)
            for run in ocr_runs:
                box(run.bbox,(190,80,170),1)
            for region in views:
                box(region.bbox,(0,150,245),3)
            cv2.imencode(".png",overlay)[1].tofile(str(directory/"overlay.png"))
            cv2.imencode(".png",binary)[1].tofile(str(directory/"binary-detection.png"))
            diagnostics.update(duration_seconds=round(time.perf_counter()-begun,3),
                text_run_count=len(runs),dimension_count=len(dims),
                parsed_annotation_count=sum(d.status=="parsed" for d in dims),
                unknown_geometry_roles=True,exclusion_regions=[list(b) for b in excluded])
            atomic_json(directory/"diagnostics.json",{"warnings":warnings,**diagnostics})
            pages.append(PageAnalysis(page=number,width=width,height=height,width_mm=width*25.4/72,height_mm=height*25.4/72,
                orientation="landscape" if width>height else "portrait",original_rotation=rotation,
                has_text_layer=has_text,content_type="mixed" if has_vector and image_records else "vector" if has_vector else "raster" if image_records else "empty",
                coordinate_system={"unit":"pdf_point","origin":"top_left_unrotated_cropbox","x_axis":"right","y_axis":"down",
                    "pdf_to_pixel":[sx,0,0,sy,0,0],"pixel_to_pdf":[1/sx,0,0,1/sy,0,0],
                    "original_rotation_degrees":rotation,"cropbox":list(page.cropbox),"mediabox":list(page.mediabox),
                    "note":"Sheet coordinates are not physical part dimensions. Image is rendered unrotated to match extracted text and paths."},
                images={"original_url":f"{url}/original.png","overlay_url":f"{url}/overlay.png","binary_url":f"{url}/binary-detection.png",
                        "dpi":DPI,"width_px":pix.width,"height_px":pix.height,"original_unchanged":True},
                embedded_images=image_records,text_blocks=blocks,text_runs=pdf_runs+ocr_runs,dimensions=dims,geometry=geometry,
                views=views,ambiguous_elements=ambiguous,warnings=warnings,diagnostics=diagnostics))
            pages[-1].ocr=ocr_info
            pages[-1].layout=layout
    analysis = DrawingAnalysis(drawing_id=drawing_id,filename=filename,sha256=sha256,
        created_at=datetime.now(timezone.utc).isoformat(),pages=pages,
        metadata={"engine":"local_tesseract_opencv_vector_geometry","pymupdf_version":fitz.VersionBind,
                  "opencv_version":cv2.__version__,"render_dpi":DPI,"accuracy_metrics":None,
                  "dimension_confidence_policy":"PDF: null; OCR: uncalibrated source engine score, not a probability of dimensional correctness",
                  "source_pdf_url":f"/api/drawing/{drawing_id}/pdf"},
        warnings=["Единицы размеров мм предполагаются по инженерной практике; координаты листа указаны в PDF points.",
                  "Точность распознавания не измерена: эталонная разметка отсутствует.",
                  "B — проверенная частичная CAD; C — приближённая CAD; D — визуальная сетка. Оценочные размеры и ограничения сохраняются отдельно."],
        statistics={"pages":len(pages),"dimensions":sum(len(p.dimensions) for p in pages),
                    "dimensions_needing_review":sum(d.status=="needs_review" for p in pages for d in p.dimensions),
                    "geometry_elements":sum(len(g) for p in pages for g in p.geometry.values()),
                    "duration_seconds":round(time.perf_counter()-start,3)})
    return analysis


def analyze_pdf(path: Path, output: Path, drawing_id: str, filename: str, sha256: str, mode=None, model=None):
    from . import pipeline_cache
    start=time.perf_counter()
    cached=pipeline_cache.load(sha256,output,drawing_id,filename)
    analysis=DrawingAnalysis.model_validate(cached) if cached else extract_pdf(path,output,drawing_id,filename,sha256)
    document=analysis.model_dump(mode='json')
    if not cached:pipeline_cache.store(document,output)
    interpretation,intent=reconstruct(document,output)
    from .hybrid import run_hybrid
    from .ollama_client import LLMSettings
    settings=LLMSettings.from_env(mode,model)
    if settings.analysis_mode=='hybrid':
        from .reconstruction_engine import run_engine
        interpretation,intent,llm=run_engine(document,output,interpretation,intent,settings)
    else:interpretation,intent,llm=run_hybrid(document,output,interpretation,intent,settings)
    analysis.metadata['llm']={k:llm.get(k) for k in ('mode','status','selected_origin','classic_preserved','settings','reason','requested_model','actual_model','ollama_status','ollama_error')}
    analysis.metadata['llm_diagnostics_url']=f'/api/llm/{drawing_id}'
    for page,serialized in zip(analysis.pages,document['pages']):
        page.layout=serialized['layout']
        page.dimensions=[Dimension.model_validate(d) for d in serialized['dimensions']]
        atomic_json(output/f'page-{page.page}'/'layout.json',page.layout)
    analysis.statistics['reconstruction_status']=interpretation['status']
    analysis.statistics['quality_level']=llm.get('quality')
    analysis.metadata['reconstruction']=interpretation.get('reconstruction',{})
    analysis.statistics['duration_seconds']=round(time.perf_counter()-start,3)
    analysis.statistics['dimensions']=sum(len(p.dimensions) for p in analysis.pages)
    analysis.statistics['dimensions_needing_review']=sum(d.status=='needs_review' for p in analysis.pages for d in p.dimensions)
    confirmed=0
    for report in interpretation['pages']:
        if 'page' not in report:continue
        page_result=analysis.pages[report['page']-1]
        for binding in report.get('dimension_bindings',[]):
            if binding['status']!='scale_and_topology_confirmed':continue
            dimension=next((d for d in page_result.dimensions if d.id==binding['dimension_id']),None)
            if dimension:
                dimension.associated_geometry_id=binding['geometry_id']
                dimension.associations=[binding]+[x for x in dimension.associations if x['geometry_id']!=binding['geometry_id']]
                confirmed+=1
    analysis.statistics['confirmed_geometry_elements']=confirmed
    analysis.metadata['interpretation_url']=f'/api/interpretation/{drawing_id}'
    analysis.metadata['cad_intent_url']=f'/api/cad-intent/{drawing_id}'
    atomic_json(output/"analysis.json",analysis.model_dump(mode="json"))
    atomic_json(output/"statistics.json",analysis.statistics)
    atomic_json(output/'layout.json',{'drawing_id':drawing_id,'pages':[p.layout for p in analysis.pages]})
    return analysis.statistics
