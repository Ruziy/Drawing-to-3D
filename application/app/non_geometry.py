"""Locate drawing title tables from actual heading text and horizontal grid lines."""
import re


def record(value):return value.model_dump(mode='json') if hasattr(value,'model_dump') else value


def title_regions(page):
    width,height=page['width'],page['height'];headings=[]
    grid=grid_title_regions(page)
    for raw in page.get('text_runs',[]):
        run=record(raw);b=run['bbox'];text=run['raw_text']
        if b[1]>height*.72 and b[0]>width*.15 and re.search(r'масс|масштаб|лист(?:ов)?\b|разраб|провер|изм\.',text,re.I):headings.append(b)
    if not headings:return grid
    top=min(b[1] for b in headings);centers=[(b[0]+b[2])/2 for b in headings]
    lines=[]
    for raw in page.get('geometry',{}).get('lines',[]):
        line=record(raw);points=line.get('coordinates',[])
        if len(points)!=2:continue
        a,b=points
        if abs(a[1]-b[1])>.7 or abs(a[0]-b[0])<width*.25:continue
        lo,hi=sorted((a[0],b[0]));y=(a[1]+b[1])/2
        if top-95<y<=top+3 and sum(lo-2<=x<=hi+2 for x in centers)>=max(1,len(centers)*.6):lines.append((y,lo,hi))
    if not lines:return grid
    y,x0,x1=min(lines)
    return grid+[{'type':'title_table','bbox':[x0,max(y,height*.70),x1,height],
             'method':'actual_title_headings_and_enclosing_horizontal_grid','heading_count':len(headings)}]


def grid_title_regions(page):
    """Detect a repeated, intersecting title grid even when headings are unreadable.

    A lower-page rectangle alone is insufficient: require four complete rows,
    five distinct vertical dividers and repeated crossings. This avoids a blind
    bottom crop that would erase lower views on a multi-sheet plate drawing.
    """
    width,height=page['width'],page['height'];horizontal=[];vertical=[]
    raw_lines=[]
    for raw in page.get('geometry',{}).get('lines',[]):
        points=record(raw).get('coordinates',[])
        if len(points)==2:raw_lines.append(points)
    raw_lines += [[s['a'],s['b']] for s in page.get('layout',{}).get('dimension_line_candidates',[])]
    for a,b in raw_lines:
        if abs(a[1]-b[1])<1 and abs(a[0]-b[0])>width*.30 and min(a[1],b[1])>height*.68:
            value=((a[1]+b[1])/2,min(a[0],b[0]),max(a[0],b[0]))
            if not any(abs(value[0]-v[0])<1 and abs(value[1]-v[1])<3 and abs(value[2]-v[2])<3 for v in horizontal):horizontal.append(value)
        elif abs(a[0]-b[0])<1 and abs(a[1]-b[1])>height*.015 and max(a[1],b[1])>height*.75:
            vertical.append(((a[0]+b[0])/2,min(a[1],b[1]),max(a[1],b[1])))
    candidates=[]
    for top,left,right in horizontal:
        rows=sorted({round(y,1) for y,lo,hi in horizontal if y>=top-1 and
            max(0,min(hi,right)-max(lo,left))>(right-left)*.85})
        # Merge the two painted edges of one scanned stroke.
        rows=[y for i,y in enumerate(rows) if not i or y-rows[i-1]>1.2]
        if len(rows)<4 or rows[-1]-rows[0]<height*.06:continue
        dividers=[];strong=[]
        for x,lo,hi in vertical:
            if not left-2<=x<=right+2:continue
            crossings=sum(lo-2<=y<=hi+2 for y in rows)
            if crossings>=2 and not any(abs(x-p)<1.5 for p in dividers):dividers.append(x)
            if crossings>=3 and not any(abs(x-p)<1.5 for p in strong):strong.append(x)
        if len(dividers)>=5 and len(strong)>=3:
            candidates.append({'type':'title_table','bbox':[left,rows[0],right,height],
                'method':'observed_repeated_grid_crossings_independent_of_OCR','horizontal_rows':len(rows),
                'vertical_dividers':len(dividers),'strong_dividers':len(strong)})
    if not candidates:return []
    return [min(candidates,key=lambda r:r['bbox'][1])]


def overlap_fraction(box,region):
    a,b,c,d=box;x0,y0,x1,y1=region
    overlap=max(0,min(c,x1)-max(a,x0))*max(0,min(d,y1)-max(b,y0))
    return overlap/max((c-a)*(d-b),1e-8)
