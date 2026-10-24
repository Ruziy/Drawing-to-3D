"""Physical strokes, dimension lines and section candidates with explicit evidence."""
import math
import re
from pathlib import Path

import cv2
import numpy as np

from .contour_graph import stroke_graph


def box_overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def analyze_layout(page,geometry,runs,rgb,directory:Path):
    h,w=rgb.shape[:2];pw,ph=page.rect.width,page.rect.height;sx,sy=w/pw,h/ph
    mask=np.zeros((h,w),np.uint8)
    vector=[g for group in geometry.values() for g in group if g.source=='vector']
    widths=[g.properties.get('stroke_width_pt') or 0 for g in vector if (g.properties.get('stroke_width_pt') or 0)>0]
    threshold=max(.7,float(np.percentile(widths,85))*.78) if widths else None
    textboxes=[r.bbox for r in runs if r.source=='pdf_text']
    roles=[];segments=[];hatching=[];filled_groups={}
    for g in vector:
        width=g.properties.get('stroke_width_pt') or 0
        coords=np.array(g.coordinates,dtype=float)
        if g.properties.get('fill') is not None and g.type=='line':
            filled_groups.setdefault(g.properties.get('path_index',g.id),[]).extend(coords.tolist())
        center=np.mean(coords,axis=0)
        text_hit=any(all(b[0]-1<=p[0]<=b[2]+1 and b[1]-1<=p[1]<=b[3]+1 for p in coords) for b in textboxes)
        if width>=threshold and not text_hit:
            role='physical_boundary_candidate'
            if g.type=='circle':
                c=g.properties['center'];r=g.properties['radius_pt'];cv2.circle(mask,(round(c[0]*sx),round(c[1]*sy)),round(r*sx),255,max(2,round(width*sx)))
            else:
                points=np.round(coords*[sx,sy]).astype(np.int32)
                cv2.polylines(mask,[points],False,255,max(2,round(width*sx)))
        elif g.type=='line' and width>0 and width<threshold and not text_hit and g.properties.get('fill') is None:
            a,b=coords;delta=b-a;length=float(np.linalg.norm(delta))
            if abs(delta[0])<.3 or abs(delta[1])<.3:
                role='dimension_or_extension_candidate'
                if length>8:
                    segments.append({'id':g.id,'a':a.tolist(),'b':b.tolist(),'length_pt':length,
                                     'orientation':'vertical' if abs(delta[0])<.3 else 'horizontal'})
            elif min(abs(delta[0]),abs(delta[1]))/max(abs(delta[0]),abs(delta[1]),1)>.5:
                role='hatching_or_oblique_candidate'
                hatching.append({'point':((a+b)/2).tolist(),'direction':(delta/length).tolist()})
            else:role='unknown'
        else:role='text_or_unknown'
        g.line_class=role
        g.properties['interpretations']=[role,'unknown'] if role!='text_or_unknown' else ['text','unknown']
        roles.append({'geometry_id':g.id,'role':role})
    method='vector_stroke_width_and_text_exclusion'
    if not widths:
        # Thickness filter on a copy, scaled to preserve heavy part boundaries.
        ink=(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)<150).astype(np.uint8)*255
        distance=cv2.distanceTransform(ink,cv2.DIST_L2,3)
        mask=(distance>=1.7).astype(np.uint8)*255
        mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
        for r in runs:
            # Page OCR also hallucinates text across hatch fields. Such a box
            # must not erase the very section we are trying to reconstruct.
            text=r.recognition.get('structured_text',r.raw_text).strip()
            if re.fullmatch(r'[Ø⌀ФMМR]?\s*\d[\d.,+−±() A-Za-zА-Яа-я×xх*-]{0,24}',text) and (r.confidence or 0)>.55:
                x0,y0,x1,y1=r.bbox
                short=min(x1-x0,y1-y0);long=max(x1-x0,y1-y0)
                oriented=(x1-x0>=y1-y0) == (abs(r.direction[0])>.8)
                if oriented and short<18 and long<130 and (x1-x0)*(y1-y0)<pw*ph*.002:
                    mask[max(0,round(y0*sy)-2):min(h,round(y1*sy)+2),max(0,round(x0*sx)-2):min(w,round(x1*sx)+2)]=0
        thin=cv2.HoughLinesP(ink,1,np.pi/180,45,minLineLength=40,maxLineGap=3)
        if thin is not None:
            for i,line in enumerate(thin[:,0]):
                a=np.array([line[0]/sx,line[1]/sy]);b=np.array([line[2]/sx,line[3]/sy]);delta=b-a
                if min(abs(delta))<.7:
                    segments.append({'id':f'CV-DL{i}','a':a.tolist(),'b':b.tolist(),'length_pt':float(np.linalg.norm(delta)),
                                     'orientation':'vertical' if abs(delta[0])<.7 else 'horizontal'})
                elif min(abs(delta))/max(abs(delta))>.5:
                    hatching.append({'point':((a+b)/2).tolist(),'direction':(delta/np.linalg.norm(delta)).tolist()})
        # Hough voting is dominated by hatching on scanned sections. Recover
        # complete orthogonal dimension strokes independently, including their
        # arrow-ended spans, from a directional opening of the untouched ink.
        for orientation,kernel in (('vertical',np.ones((35,1),np.uint8)),('horizontal',np.ones((1,35),np.uint8))):
            opened=cv2.morphologyEx(ink,cv2.MORPH_OPEN,kernel)
            for contour in cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
                x,y,cw,ch=cv2.boundingRect(contour)
                long,short=(ch,cw) if orientation=='vertical' else (cw,ch)
                if long<40 or short>max(8,long*.16):continue
                if orientation=='vertical':a=[(x+(cw-1)/2)/sx,y/sy];b=[a[0],(y+ch-1)/sy]
                else:a=[x/sx,(y+(ch-1)/2)/sy];b=[(x+cw-1)/sx,a[1]]
                segments.append({'id':f'CV-ORTHO{len(segments)+1}','a':a,'b':b,'length_pt':float(np.linalg.norm(np.array(b)-a)),
                                 'orientation':orientation,'method':'directional_opening_complete_stroke'})
        # Thin walls have hatch segments shorter than the old 40-pixel Hough
        # minimum. LSD plus parallel-neighbour evidence recovers these marks
        # without treating every oblique text/leader stroke as material.
        factor=min(1,2500/max(w,h));small=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None,fx=factor,fy=factor)
        detected=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(small)[0]
        hatch_candidates=[]
        if detected is not None:
            for x0,y0,x1,y1 in detected[:,0]:
                a=np.array([x0/(sx*factor),y0/(sy*factor)]);b=np.array([x1/(sx*factor),y1/(sy*factor)])
                delta=b-a;length=float(np.linalg.norm(delta))
                if not 4<length<80 or min(abs(delta))/max(abs(delta),default=1)<.82:continue
                direction=delta/length
                if direction[0]<0:direction=-direction
                hatch_candidates.append({'point':((a+b)/2).tolist(),'direction':direction.tolist(),'method':'lsd_parallel_short_strokes'})
        if hatch_candidates:
            points=np.array([c['point'] for c in hatch_candidates]);directions=np.array([c['direction'] for c in hatch_candidates])
            for i,candidate in enumerate(hatch_candidates):
                distance=np.linalg.norm(points-points[i],axis=1)
                neighbours=(distance>1.1)&(distance<14)&((directions@directions[i])>.995)
                if np.count_nonzero(neighbours)>=2:hatching.append(candidate)
        method='raster_thickness_and_text_exclusion'
    # Duplicate PDF paint runs are not independent material witnesses.
    unique_hatches={}
    for record in hatching:
        key=tuple(round(v,2) for v in record['point'])+tuple(round(abs(v),2) for v in record['direction'])
        unique_hatches.setdefault(key,record)
    hatching=list(unique_hatches.values())
    # Sheet border and title block are not part geometry; keep unmodified original.
    cv2.imencode('.png',mask)[1].tofile(str(directory/'full-physical-strokes.png'))
    mask[:round(h*.045)]=0;mask[round(h*.70):]=0
    mask[:,:round(w*.08)]=0;mask[:,round(w*.97):]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    def visible_hatch(record,polygon):
        p=np.array(record['point']);direction=np.array(record['direction'])
        if cv2.pointPolygonTest(polygon,tuple(p),False)<=0:return False
        observed=0
        for t in (-2,-1,0,1,2):
            q=p+t*direction;x,y=round(q[0]*sx),round(q[1]*sy)
            crop=rgb[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)]
            if crop.size and np.min(crop)<160:observed+=1
        return observed>=4
    for ci,c in enumerate(contours):
        area=cv2.contourArea(c);x,y,bw,bh=cv2.boundingRect(c)
        if bw<40 or bh<12 or area<180:continue
        # A thin open stroke is not a closed section.
        if area/(bw*bh)<.12:continue
        approx=cv2.approxPolyDP(c,max(1.0,.0007*cv2.arcLength(c,True)),True)[:,0]
        polygon=(approx/[sx,sy]).astype(np.float32)
        # PDF paths can extend beyond clipping boundaries. A hatch witness must
        # also be visible in the actual render, not merely exist in get_drawings().
        hatch_count=sum(visible_hatch(record,polygon) for record in hatching)
        parent=int(hierarchy[0][ci][3]);depth=0;ancestor=parent
        while ancestor>=0:
            depth+=1;ancestor=int(hierarchy[0][ancestor][3])
        candidates.append({'id':f'PROFILE{len(candidates)+1:03d}','bbox':[x/sx,y/sy,(x+bw-1)/sx,(y+bh-1)/sy],
            'polygon_pdf':polygon.tolist(),'area_pt2':area/(sx*sy),'method':method,'type':'material_section_candidate' if depth%2==1 and hatch_count>=2 else 'unknown',
            'contour_depth':depth,'hatch_line_count':int(hatch_count)})
    candidates.sort(key=lambda c:-c['area_pt2'])
    cv2.imencode('.png',mask)[1].tofile(str(directory/'physical-strokes.png'))
    # A rear flange rim is often drawn thin. Restricting circle witnesses to
    # heavy strokes removed the very pitch/end-view references we need.
    circle_ink=(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)<160).astype(np.uint8)*255
    circle_witnesses=detect_end_circles(circle_ink,pw,ph,runs)
    arrows=[]
    for path_id,points in filled_groups.items():
        points=np.unique(np.round(points,3),axis=0).astype(np.float32)
        if len(points)!=3:continue
        hull=cv2.convexHull(points)[:,0,:];area=cv2.contourArea(hull)
        if not 1<area<45:continue
        lengths=[float(np.linalg.norm(b-a)) for a,b in zip(hull,np.roll(hull,-1,axis=0))]
        if max(lengths)<min(lengths)*1.8:continue
        # The vertex opposite the shortest side is the observed arrow tip.
        tip=hull[(int(np.argmin(lengths))+2)%3].tolist()
        arrows.append({'id':f'ARROW{len(arrows)+1:03d}','tip_pdf':tip,'polygon_pdf':hull.tolist(),'path_index':path_id,
                       'method':'filled_slim_triangle_candidate'})
    graph=stroke_graph(geometry,pw,ph)
    physical_segments=[]
    if widths:
        for g in vector:
            if g.line_class=='physical_boundary_candidate' and g.type=='line' and g.properties.get('fill') is None:
                physical_segments.append({'id':g.id,'endpoints_pdf':[list(p) for p in g.coordinates],'source':'vector'})
    else:
        observed=cv2.HoughLinesP(mask,1,np.pi/180,45,minLineLength=round(12*sx),maxLineGap=round(.6*sx))
        if observed is not None:
            for index,(x0,y0,x1,y1) in enumerate(observed[:,0]):
                physical_segments.append({'id':f'BOUNDARY{index+1:04d}','endpoints_pdf':[[x0/sx,y0/sy],[x1/sx,y1/sy]],'source':'raster_thickness_mask'})
    for face in graph['faces']:
        polygon=np.array(face['polygon_pdf'],np.float32)
        face['hatch_line_count']=sum(visible_hatch(record,polygon) for record in hatching)
        if face['hatch_line_count']>=2:face['type']='material_section_candidate'
    return {'schema_version':'1.0.0','coordinate_unit':'pdf_point','method':method,'stroke_threshold_pt':threshold,
            'line_roles':roles,'dimension_line_candidates':segments,'profile_candidates':candidates[:50],
            'contour_graph':graph,
            'arrow_candidates':arrows,
            'physical_boundary_segments':physical_segments,'hatch_witnesses':hatching,
            'end_view_circle_witnesses':circle_witnesses,
            'warnings':['Кандидаты физических границ требуют проверки размерными связями; штриховка исключается по толщине, а не трактуется как ребро.']}


def associate_dimensions(dimensions,layout):
    for d in dimensions:
        d.associations=[]
        d.associated_geometry_id=None
        if d.type not in ('linear','diameter','thread'):continue
        cx,cy=(d.bbox[0]+d.bbox[2])/2,(d.bbox[1]+d.bbox[3])/2
        orientation='vertical' if abs(d.direction[1])>.8 else 'horizontal'
        matches=[]
        for line in layout['dimension_line_candidates']:
            if line['orientation']!=orientation:continue
            a,b=np.array(line['a']),np.array(line['b'])
            idx=0 if orientation=='horizontal' else 1;other=1-idx
            center=(cx,cy)
            perpendicular=abs(center[other]-a[other])
            along=center[idx]
            lo,hi=sorted([a[idx],b[idx]])
            allowance=max(20,(d.bbox[2]-d.bbox[0]) if idx==0 else (d.bbox[3]-d.bbox[1]))
            if not lo-allowance<=along<=hi+allowance or perpendicular>28:continue
            length=hi-lo
            if d.nominal_value<=0:continue
            scale=length/d.nominal_value
            if not .15<scale<35:continue
            # Baseline distance and centering are measured evidence; no semantic certainty implied.
            normal=(-d.direction[1],d.direction[0]);sign=1 if normal[other]>=0 else -1
            baseline=d.bbox[other+2] if sign>0 else d.bbox[other]
            signed_distance=sign*(a[other]-baseline)
            score=abs(signed_distance-2)+max(0,-signed_distance)*.5+abs(along-(lo+hi)/2)*.08
            matches.append((score,line,scale))
        matches.sort(key=lambda item:item[0])
        if matches:
            best=matches[0]
            d.associations=[{'geometry_id':line['id'],'method':'parallel_dimension_line_proximity',
                'score_distance_pt':round(score,4),'scale_pt_per_mm':scale,'endpoints_pdf':[line['a'],line['b']],
                'arrow_witness_ids':[arrow['id'] for arrow in layout.get('arrow_candidates',[])
                    if min(np.linalg.norm(np.array(arrow['tip_pdf'])-line['a']),np.linalg.norm(np.array(arrow['tip_pdf'])-line['b']))<3],
                'orientation':line['orientation'],'status':'candidate'} for score,line,scale in matches[:12]]
            if len(matches)==1 or matches[1][0]-best[0]>2:
                d.associated_geometry_id=best[1]['id']


def detect_end_circles(mask,pw,ph,runs):
    h,w=mask.shape
    circle_witnesses=[]
    if any(re.search(r'(?:[3-9]|\d{2,})\s*отв',r.raw_text,re.I) for r in runs):
        factor=min(1,1500/w);small=cv2.resize(mask,None,fx=factor,fy=factor)
        circles=cv2.HoughCircles(small,cv2.HOUGH_GRADIENT,1.2,10,param1=100,param2=22,minRadius=6,maxRadius=250)
        if circles is not None:
            dilated=cv2.dilate(small,np.ones((5,5),np.uint8));angles=np.linspace(0,2*np.pi,180,endpoint=False)
            for x,y,r in circles[0]:
                xx=np.rint(x+r*np.cos(angles)).astype(int);yy=np.rint(y+r*np.sin(angles)).astype(int)
                valid=(xx>=0)&(xx<small.shape[1])&(yy>=0)&(yy<small.shape[0])
                coverage=float(np.mean(valid&(dilated[np.clip(yy,0,small.shape[0]-1),np.clip(xx,0,small.shape[1]-1)]>150)))
                if coverage>.75:
                    circle_witnesses.append({'center_pdf':[float(x/small.shape[1]*pw),float(y/small.shape[0]*ph)],
                        'radius_pdf':float(r/small.shape[1]*pw),'observed_ring_fraction':coverage,
                        'method':'hough_with_rendered_ring_support'})
    return circle_witnesses
