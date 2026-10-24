"""Keep technical notes/title-block values out of geometric dimension candidates."""
import re


def filter_annotation_context(dimensions,width,height,runs=()):
    kept=[];rejected=[]
    headers=[]
    for run in runs:
        r=run.model_dump(mode='json') if hasattr(run,'model_dump') else run
        if r['bbox'][1]>height*.72 and re.search(r'масс|масштаб|листов',r['raw_text'],re.I):headers.append(r['bbox'])
    for dimension in dimensions:
        d=dimension.model_dump(mode='json') if hasattr(dimension,'model_dump') else dimension
        x0,y0,x1,y1=d['bbox'];raw=d['raw_text'];reason=None
        cx,cy=(x0+x1)/2,(y0+y1)/2
        under_header=any(abs(cx-(b[0]+b[2])/2)<max(30,width*.045) and b[3]<=cy<b[3]+height*.13 for b in headers)
        fallback=(not headers and cx>width*.65 and cy>height*(.94 if width>height else .83))
        if under_header or fallback:reason='title_block_mass_scale_or_revision_not_geometry'
        elif cx<width*.13 and cy>height*.70:reason='revision_margin_not_a_geometry_dimension'
        elif re.search(r'(?:размеры\s+для\s+спра|раз[вб]ернутая\s+длина|маркиро|неуказан|тв[её]рдост|общие\s+допус)',raw,re.I):
            reason='technical_requirement_or_note_number_not_dimension'
        elif d['type']=='linear' and re.fullmatch(r'\s*\d+\s*[:/]\s*\d+\s*',raw):reason='drawing_scale_not_dimension'
        elif d['nominal_value']<=0:reason='nonpositive_size_not_a_geometry_parameter'
        if reason:rejected.append({**d,'exclusion_reason':reason})
        else:kept.append(dimension)
    return kept,rejected
