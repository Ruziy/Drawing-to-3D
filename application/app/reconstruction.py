"""Reconstruct supported rotational sections from PDF evidence, never filenames."""
import math
import re
from pathlib import Path

import cv2
import numpy as np

from .constraints import solve_chain
from .schemas import CADIntent
from .storage import atomic_json


def canonical(points,axis):
    a=np.asarray(points,dtype=float)
    return a if axis=='horizontal' else a[:,::-1]


def clip_half(poly,axis):
    result=[]
    for a,b in zip(poly,np.vstack([poly[1:],poly[:1]])):
        inside_a=a[1]<=axis;inside_b=b[1]<=axis
        if inside_a:result.append(a.tolist())
        if inside_a!=inside_b:
            t=(axis-a[1])/(b[1]-a[1]);result.append((a+t*(b-a)).tolist())
    return np.array(result,dtype=float)


def bounds(poly):
    p=np.asarray(poly);return [float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]


def section_candidates(page):
    candidates=page['layout']['profile_candidates'];out=[]
    for axis in ('horizontal','vertical'):
        materials=[c for c in candidates if c['contour_depth']%2==1]
        for i,a in enumerate(materials):
            pa=canonical(a['polygon_pdf'],axis);ba=bounds(pa)
            for b in materials[i+1:]:
                if a['type']!='material_section_candidate' and b['type']!='material_section_candidate':continue
                pb=canonical(b['polygon_pdf'],axis);bb=bounds(pb)
                if max(abs(ba[0]-bb[0]),abs(ba[2]-bb[2]))>2.5:continue
                if abs(a['area_pt2']-b['area_pt2'])/max(a['area_pt2'],b['area_pt2'])>.12:continue
                if not (ba[3]<bb[1] or bb[3]<ba[1]):continue
                top=pa if ba[1]<bb[1] else pb
                top_record=a if ba[1]<bb[1] else b
                center=(ba[1]+ba[3]+bb[1]+bb[3])/4
                # A longitudinal section has an axial span, not an isolated circular end view.
                if ba[2]-ba[0]<15:continue
                out.append({'axis':axis,'axis_coordinate_pdf':center,'polygon_pdf_canonical':top,
                    'profile_id':top_record['id'],'proof':'mirrored_hatched_material_faces',
                    'full_bounds_canonical':[min(ba[0],bb[0]),min(ba[1],bb[1]),max(ba[2],bb[2]),max(ba[3],bb[3])]})
        for a in candidates:
            if a['contour_depth']!=0:continue
            pa=canonical(a['polygon_pdf'],axis);ba=bounds(pa)
            if (ba[2]-ba[0])<1.4*(ba[3]-ba[1]) or ba[2]-ba[0]<30:continue
            center=(ba[1]+ba[3])/2
            upper=clip_half(pa,center)
            if len(upper)<3:continue
            out.append({'axis':axis,'axis_coordinate_pdf':center,'polygon_pdf_canonical':upper,
                'profile_id':a['id'],'proof':'symmetric_external_silhouette_candidate',
                'full_bounds_canonical':ba})
    return out


def relevant_associations(page,section):
    result=[];axis=section['axis'];center=section['axis_coordinate_pdf'];box=section['full_bounds_canonical']
    for d in page['dimensions']:
        if d['nominal_value']<=0 or d['type'] not in ('linear','diameter','thread'):continue
        for association in d['associations']:
            endpoints=canonical(association['endpoints_pdf'],axis)
            axial=association['orientation']==axis
            if axial:
                if d['type']!='linear':continue
                lo,hi=sorted(endpoints[:,0])
                if lo<box[0]-3 or hi>box[2]+3:continue
                if min(abs(endpoints[0,1]-box[1]),abs(endpoints[0,1]-box[3]))>100:
                    # Long stacks of dimension chains can be far from the part.
                    # Require visible collinear extension strokes, not an
                    # arbitrary distance cutoff or a global sheet scale.
                    supported=0
                    for u,v in endpoints:
                        target=min((box[1],box[3]),key=lambda x:abs(x-v));left,right=sorted((target,v));intervals=[]
                        for line in page['layout'].get('dimension_line_candidates',[]):
                            if line['orientation']==axis:continue
                            line_ends=canonical([line['a'],line['b']],axis)
                            if max(abs(line_ends[:,0]-u))>1.2:continue
                            a,b=sorted(line_ends[:,1]);intervals.append((max(left,a),min(right,b)))
                        cursor=left;covered=0
                        for a,b in sorted(intervals):
                            if b<=max(a,cursor):continue
                            covered+=b-max(a,cursor);cursor=max(cursor,b)
                        if covered/max(right-left,1)>.80:supported+=1
                    if supported<2:continue
            else:
                if d['type'] not in ('diameter','thread'):continue
                if abs(np.mean(endpoints[:,1])-center)>2.0:continue
                lo,hi=sorted(endpoints[:,1])
                if lo<box[1]-3 or hi>box[3]+3:continue
            result.append({'dimension':d,'association':association,'axial':axial,'endpoints':endpoints})
    return result


def choose_scale(observations):
    best=None
    for seed in observations:
        scale=seed['association']['scale_pt_per_mm']
        group={}
        for o in observations:
            if abs(o['association']['scale_pt_per_mm']/scale-1)<.018:
                key=o['dimension']['id']
                if key not in group or o['association']['score_distance_pt']<group[key]['association']['score_distance_pt']:
                    group[key]=o
        values=list(group.values())
        axial=sum(o['axial'] for o in values);radial=sum(not o['axial'] for o in values)
        score=len(values)+min(axial,radial)*2
        if not axial or not radial:continue
        if best is None or score>best[0]:best=(score,float(np.median([o['association']['scale_pt_per_mm'] for o in values])),values)
    return best


def clean_polygon(profile):
    points=[]
    for p in profile:
        p=[round(max(0,float(p[0])),6),round(float(p[1]),6)]
        if not points or np.linalg.norm(np.array(p)-points[-1])>1e-5:points.append(p)
    if len(points)>2 and np.linalg.norm(np.array(points[0])-points[-1])<1e-5:points.pop()
    # Remove collinear intermediate points after snapping to explicit constraints.
    changed=True
    while changed and len(points)>3:
        changed=False
        for i in range(len(points)):
            a,b,c=np.array(points[i-1]),np.array(points[i]),np.array(points[(i+1)%len(points)])
            cross=(b-a)[0]*(c-b)[1]-(b-a)[1]*(c-b)[0]
            if abs(cross)<1e-7 and np.dot(b-a,c-b)>=0:
                points.pop(i);changed=True;break
    return points


def profile_intersections(points):
    conflicts=[];size=len(points)
    for i in range(size):
        a,b=np.array(points[i]),np.array(points[(i+1)%size]);u=b-a
        for j in range(i+2,size):
            if i==0 and j==size-1:continue
            c,d=np.array(points[j]),np.array(points[(j+1)%size]);v=d-c
            cross=u[0]*v[1]-u[1]*v[0]
            if abs(cross)<1e-10:continue
            offset=c-a;t=(offset[0]*v[1]-offset[1]*v[0])/cross;s=(offset[0]*u[1]-offset[1]*u[0])/cross
            if -1e-9<=t<=1+1e-9 and -1e-9<=s<=1+1e-9:conflicts.append({'segment_indices':[i,j],'point':(a+t*u).tolist()})
    return conflicts


def annotation_intrusions(poly,page,axis,center,source_sign=-1):
    """Reject a narrow deep contour excursion entered by an annotation leader.

    A black arrow touching a physical edge can look like a material pocket in
    a scanned white face. Do not rotate it into an annular groove. Another
    independently observed face/fragment may still reconstruct the part.
    """
    lines=[g for g in page.get('geometry',{}).get('lines',[]) if g.get('source')=='raster']
    if not lines:return []
    original=cv2.contourArea(np.asarray(poly,np.float32));size=len(poly)
    for i in range(size):
        rolled=np.roll(poly,-i,axis=0)
        for j in range(3,min(9,size-2)):
            a,b=rolled[0],rolled[j];neck=float(np.linalg.norm(b-a))
            if not .05<neck<2.5:continue
            distance=np.linalg.norm(rolled[1:j]-a,axis=1)
            if float(distance.max())<max(4,neck*3):continue
            replacement=np.vstack([rolled[:1],rolled[j:]])
            if cv2.contourArea(replacement.astype(np.float32))<=original+1:continue
            before=rolled[-1]
            for k in range(1,min(4,size-j)):
                if np.linalg.norm(a-before)>=8:break
                before=rolled[-1-k]
            after=rolled[j+1]
            for k in range(j+2,min(j+5,size)):
                if np.linalg.norm(after-b)>=8:break
                after=rolled[k]
            u=a-before;v=after-b
            if np.dot(u,v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-8)<.97:continue
            deepest=rolled[1:j][int(distance.argmax())]
            for line in lines:
                ends=canonical(line['coordinates'],axis)
                if source_sign==1:ends[:,1]=2*center-ends[:,1]
                direction=ends[1]-ends[0];length=float(np.linalg.norm(direction))
                if length<12:continue
                if abs(np.dot(direction,u))/max(length*np.linalg.norm(u),1e-8)>.90:continue
                t=float(np.dot(deepest-ends[0],direction)/(length*length))
                if not -.05<t<1.05:continue
                point=ends[0]+np.clip(t,0,1)*direction
                if np.linalg.norm(point-deepest)>2:continue
                polygon=np.asarray(poly,np.float32)
                if not any(cv2.pointPolygonTest(polygon,tuple(e),False)<0 for e in ends):continue
                return [{'leader_geometry_id':line['id'],'narrow_neck_pt':neck,
                         'excursion_points_pdf_canonical':rolled[:j+1].tolist(),
                         'reason':'annotation_arrow_intrudes_into_material_contour_not_a_confirmed_groove'}]
            # Even if the thin leader was fragmented by hatching and missed by
            # Hough, a deep cavity attached through a sub-stroke neck on an
            # otherwise straight physical edge is unresolved. It may be an
            # annotation, or a real small groove. Neither may be asserted from
            # this raster face alone; try another independently observed face.
            return [{'narrow_neck_pt':neck,'excursion_points_pdf_canonical':rolled[:j+1].tolist(),
                     'reason':'narrow_deep_pocket_on_straight_edge_requires_dimension_or_clean_second_section'}]
    return []


def off_axis_drill_pockets(profile):
    """A drill cone away from the main axis is not an annular turning groove."""
    result=[];size=len(profile)
    for i in range(size):
        a,b,tip,c,d=[np.array(profile[j%size],float) for j in range(i-2,i+3)]
        if abs(a[0]-b[0])>.08 or abs(c[0]-d[0])>.08 or abs(b[1]-c[1])>.12:continue
        radius=abs(c[0]-b[0])/2;center=(b[0]+c[0])/2;depth=abs(tip[1]-b[1])
        if radius<.3 or center<radius*2 or abs(tip[0]-center)>max(.1,radius*.08) or depth<.1:continue
        if min(abs(a[1]-b[1]),abs(c[1]-d[1]))<radius:continue
        if (tip[1]-b[1])*(a[1]-b[1])>=0 or (tip[1]-c[1])*(d[1]-c[1])>=0:continue
        included=math.degrees(2*math.atan2(radius,depth))
        if 108<included<130:
            result.append({'profile_tip_index':i,'axis_offset_mm_from_contour':float(center),
                           'drill_tip_included_angle_deg':included,'reason':'off_axis_axial_blind_drill_cannot_be_revolved_into_annular_cavity'})
    return result


def repeated_hole_evidence(page,section,scale):
    quantities=[int(m[1]) for run in page.get('text_runs',[]) for m in re.finditer(r'(\d+)\s*отв',run['raw_text'],re.I) if int(m[1])>=3]
    if not quantities:return []
    circles=page['layout'].get('end_view_circle_witnesses',[]);proofs=[]
    axis=section['axis'];box=section['full_bounds_canonical'];center=section['axis_coordinate_pdf']
    for parent in circles:
        c=canonical([parent['center_pdf']],axis)[0];r=parent['radius_pdf']
        if abs(c[1]-center)>4 or box[0]-10<c[0]<box[2]+10 or r<25:continue
        for d in page['dimensions']:
            if d['type']!='diameter':continue
            label=canonical([[(d['bbox'][0]+d['bbox'][2])/2,(d['bbox'][1]+d['bbox'][3])/2]],axis)[0]
            if abs(label[0]-c[0])>r*1.5 or abs(label[1]-c[1])>r*2:continue
            pitch=d['nominal_value']/2*scale
            if not .35*r<pitch<.92*r:continue
            locations=[]
            for child in sorted(circles,key=lambda item:-item['observed_ring_fraction']):
                p=np.array(child['center_pdf']);distance=np.linalg.norm(p-np.array(parent['center_pdf']))
                if child['radius_pdf']<r*.18 and abs(distance-pitch)<max(3,r*.025):
                    if not any(np.linalg.norm(p-np.array(x))<max(8,child['radius_pdf']) for x in locations):locations.append(p.tolist())
            if len(locations)>=3:
                residual=float(np.median([abs(np.linalg.norm(np.array(p)-parent['center_pdf'])-pitch) for p in locations]))
                proofs.append({'pitch_circle_diameter_mm':d['nominal_value'],'dimension_id':d['id'],
                    'observed_hole_centers_pdf':locations,'declared_quantities':quantities,
                    'circle_center_pdf':parent['center_pdf'],'residual_pt':residual})
    return sorted(proofs,key=lambda p:p['residual_pt'])


def suppress_repeated_hole_pockets(profile,patterns):
    """Return an explicit base envelope, not fictitious axisymmetric hole grooves."""
    result=list(profile);suppressed=[]
    for proof in patterns:
        pitch_radius=proof['pitch_circle_diameter_mm']/2
        found=True
        while found:
            found=False
            for i in range(len(result)-3):
                for j in range(i+3,len(result)):
                    a,b=result[i],result[j]
                    if b[0]<=a[0] or abs(a[1]-b[1])>.08:continue
                    if abs((a[0]+b[0])/2-pitch_radius)>.5:continue
                    inside=result[i+1:j]
                    if not all(p[1]>a[1]+.2 and a[0]-.8<=p[0]<=b[0]+.8 for p in inside):continue
                    if max(p[1] for p in inside)-a[1]<1:continue
                    suppressed.append({'type':'off_axis_repeated_holes_not_revolved','profile_points_omitted':inside,
                        'base_envelope_connection':[a,b],'pattern_evidence':proof,
                        'reason':'Локальные отверстия на окружности не являются сплошной кольцевой проточкой. Показана базовая геометрия без этих отверстий.'})
                    result=result[:i+1]+result[j:];found=True;break
                if found:break
    return result,suppressed


def interpret_rotational(page,sections=None,allow_base_envelope=False,verified_local_holes=()):
    slot_annotations=[r for r in page.get('text_runs',[]) if re.search(r'\b[1-9]\d*\s*паз',r['raw_text'],re.I)]
    if slot_annotations and not allow_base_envelope:
        return None,{'status':'insufficient_data','reason':'На чертеже заданы пазы. Их пространственная форма не подтверждена; вращение сечения могло бы создать ложную кольцевую проточку.'}
    choices=[]
    for section in section_candidates(page) if sections is None else sections:
        observations=relevant_associations(page,section)
        scaled=choose_scale(observations)
        if scaled and len(scaled[2])>=(3 if sections is None else 2):
            box=section['full_bounds_canonical']
            full_span=any(o['axial'] and abs(min(o['endpoints'][:,0])-box[0])<2.5 and
                          abs(max(o['endpoints'][:,0])-box[2])<2.5 for o in scaled[2])
            if not full_span:continue
            choices.append((scaled[0],(box[2]-box[0])*(box[3]-box[1]),section,scaled,observations))
    if not choices:return None,{'status':'insufficient_data','reason':'Стратегия вращения: нет размерной калибровки выбранного сечения и подтверждённых осевых границ.'}
    choices.sort(key=lambda x:(x[2]['proof']!='mirrored_hatched_material_faces',-x[0],-x[1]))
    _,_,section,(_,scale,observations),all_observations=choices[0]
    axis=section['axis'];center=section['axis_coordinate_pdf'];poly=section['polygon_pdf_canonical'];box=section['full_bounds_canonical']
    intrusions=annotation_intrusions(poly,page,axis,center,section.get('source_radial_sign',-1))
    if intrusions:return None,{'status':'conflict','reason':'Узкая глубокая выемка на прямой границе может быть стрелкой выноски. Её размерная принадлежность не подтверждена; проверяется другой профиль.',
                              'conflicts':intrusions}
    axial=[o for o in observations if o['axial']]
    diameters=[o for o in observations if not o['axial'] and o['dimension']['type']=='diameter']
    if not diameters:return None,{'status':'insufficient_data','reason':'Нет подтверждённого явного диаметра.'}
    # Ground the profile in a dimension line spanning the actual full contour.
    full=[o for o in axial if abs(min(o['endpoints'][:,0])-box[0])<2.5 and abs(max(o['endpoints'][:,0])-box[2])<2.5]
    if not full:return None,{'status':'insufficient_data','reason':'Не установлен явный размер полной длины профиля.'}
    length_observation=min(full,key=lambda o:o['association']['score_distance_pt'])
    origin=float(min(length_observation['endpoints'][:,0]));total=length_observation['dimension']['nominal_value']
    chain=[{'endpoints':o['endpoints'][:,0].tolist(),'value':o['dimension']['nominal_value'],
            'dimension_id':o['dimension']['id']} for o in axial]
    nodes,equations,conflicts=solve_chain(chain,origin)
    if conflicts:return None,{'status':'conflict','reason':'Противоречивая размерная цепь; построение остановлено.','conflicts':conflicts}
    if len(nodes)<2:return None,{'status':'insufficient_data','reason':'Размерная цепь не связана с началом профиля.'}
    grounded_ids={e['dimension_id'] for e in equations}
    # Only diameters whose endpoints are coaxial and match this view participate.
    radial=[{'pdf_distance':o['dimension']['nominal_value']*scale/2,'radius_mm':o['dimension']['nominal_value']/2,
             'dimension_id':o['dimension']['id'],'association':o['association']} for o in diameters]
    constraints=[];profile=[];estimated=[]
    for pi,(u,v) in enumerate(poly):
        raw_z=(u-origin)/scale;raw_r=(center-v)/scale
        if raw_r<-.15:continue
        n=min(nodes,key=lambda x:abs(x['pdf_position']-u))
        rd=min(radial,key=lambda x:abs(x['pdf_distance']-(center-v)))
        z=n['position_mm'] if abs(n['pdf_position']-u)<max(1.6,scale*.25) else raw_z
        if abs(center-v)<1.2:r=0.0
        else:r=rd['radius_mm'] if abs(rd['pdf_distance']-(center-v))<max(1.5,scale*.25) else raw_r
        if abs(z)<.25:z=0.0
        if abs(z-total)<.25:z=total
        profile.append([r,z])
        sources=[]
        if z==n['position_mm']:sources+=n['dimension_ids']
        if r==rd['radius_mm']:sources.append(rd['dimension_id'])
        record={'point_index':pi,'radius_mm':r,'z_mm':z,'source_dimension_ids':sources,
                'radius_status':'explicit_dimension' if r==rd['radius_mm'] else 'axis' if r==0 else 'scaled_contour_estimate',
                'axial_status':'dimension_chain' if z==n['position_mm'] else 'scaled_contour_estimate'}
        constraints.append(record)
        if 'scaled_contour_estimate' in record.values():estimated.append(record)
    profile=clean_polygon(profile)
    drill_pockets=off_axis_drill_pockets(profile)
    if drill_pockets:return None,{'status':'conflict','reason':'Сечение содержит внеосевое глухое сверление с коническим дном. Его вращение создало бы ложную кольцевую полость; нужны координаты и отдельная операция Hole/Cut.',
                                 'conflicts':drill_pockets}
    intersections=profile_intersections(profile)
    if intersections:return None,{'status':'conflict','reason':'После привязки размеров профиль самопересекается. Неоднозначная геометрия не строится.','conflicts':intersections}
    pattern_proofs=repeated_hole_evidence(page,section,scale)
    profile,suppressed=suppress_repeated_hole_pockets(profile,pattern_proofs)
    profile=clean_polygon(profile)
    if len(profile)<3:return None,{'status':'insufficient_data','reason':'Не получен замкнутый профиль.'}
    if section['proof']=='symmetric_external_silhouette_candidate':
        represented={round(p[0]*2,4) for p in profile}
        missing=[o['dimension']['id'] for o in diameters if round(o['dimension']['nominal_value'],4) not in represented
                 and o['dimension']['id'] not in verified_local_holes]
        if missing:return None,{'status':'insufficient_data','reason':'Наружный силуэт не объясняет внутренние диаметры; сплошное тело вместо полой детали не создаётся.',
                               'unresolved_dimension_ids':missing}
    max_r=max(p[0] for p in profile);min_r=min(p[0] for p in profile)
    source_poly=poly.copy()
    source_sign=section.get('source_radial_sign',-1)
    if source_sign==1:source_poly[:,1]=2*center-source_poly[:,1]
    part_type='bushing' if min_r>0 else 'stepped_shaft' if len({round(p[0],2) for p in profile if p[0]>0})>1 else 'axisymmetric'
    if total<max_r:part_type='disk_or_flange'
    unresolved=[{'type':'unverified_additional_features','reason':'Дополнительные виды, пазы, резьбы и локальные элементы требуют отдельного подтверждения; вращение восстанавливает только выбранное сечение.'}]
    if slot_annotations:
        unresolved.append({'type':'slots_not_cut','annotations':[r['raw_text'] for r in slot_annotations],
            'reason':'Показана только базовая оболочка комбинированной детали. Пазы не вырезаны; эта оболочка содержит лишний материал и не является готовой деталью.'})
    if section.get('fragment_dimension_id'):
        unresolved.append({'type':'bounded_fragment','dimension_id':section['fragment_dimension_id'],
            'reason':'Показана часть материала между подтверждёнными размерными плоскостями; плоскости отсечения не объявляются торцами всей детали.'})
    unresolved.extend(suppressed)
    disconnected=[o['dimension']['id'] for o in axial if o['dimension']['id'] not in grounded_ids]
    if disconnected:unresolved.append({'type':'disconnected_dimension_chain','dimension_ids':disconnected,
        'reason':'Масштаб согласован, но цепь не связана с началом детали; координаты этих размеров не используются как подтверждённые.'})
    if estimated:unresolved.append({'type':'contour_estimates','count':len(estimated),'reason':'Часть переходов измерена по контуру после калибровки; явный размер отсутствует.'})
    if any(d['type']=='thread' for d in page['dimensions']):unresolved.append({'type':'thread','reason':'Винтовая резьба не моделируется как гладкая поверхность с фиктивной точностью.'})
    unmatched=[{'dimension_id':d['id'],'raw_text':d['raw_text'],'nominal_value':d['nominal_value']} for d in page['dimensions']
               if d['id'] not in {o['dimension']['id'] for o in observations}]
    if unmatched:unresolved.append({'type':'unbound_annotations','annotations':unmatched,
                                    'reason':'Эти надписи не назначены поверхностям выбранного профиля.'})
    # Unsupported strongly related dimensions are contradictions, not numeric repairs.
    used={o['dimension']['id'] for o in observations}
    proportional_conflicts=[]
    for o in all_observations:
        d=o['dimension']
        if d['id'] in used or d['type']!='diameter':continue
        inferred=o['association']['scale_pt_per_mm']*d['nominal_value']/scale
        if abs(inferred-d['nominal_value'])/max(d['nominal_value'],1)>.12:
            proportional_conflicts.append({'dimension_id':d['id'],'raw_text':d['raw_text'],'recognized_nominal':d['nominal_value'],
                'alternative_from_geometry_mm':inferred,'action':'not_replaced','reason':'dimension_geometry_scale_disagreement'})
    if proportional_conflicts:
        return None,{'status':'conflict','reason':'Распознанный диаметр противоречит масштабу подтверждённых размеров.','conflicts':proportional_conflicts}
    expectations=[]
    for o in observations:
        d=o['dimension'];a=o['association'];ends=o['endpoints']
        if o['axial']:
            expectations.append({'type':'axial_distance','nominal_mm':d['nominal_value'],'dimension_id':d['id'],
                'positions_mm':[(float(u)-origin)/scale for u in ends[:,0]],'source_endpoints_pdf':a['endpoints_pdf'],
                'grounded':d['id'] in grounded_ids})
        elif d['type']=='diameter':
            radius=d['nominal_value']/2
            edges=[]
            for left,right in zip(profile,profile[1:]+profile[:1]):
                if abs(left[0]-radius)<1e-5 and abs(right[0]-radius)<1e-5 and abs(left[1]-right[1])>.3:
                    edges.append(sorted([left[1],right[1]]))
            expectations.append({'type':'cylindrical_surface','nominal_mm':d['nominal_value'],'dimension_id':d['id'],
                'axial_intervals_mm':edges,'status':'confirmed_profile_surface' if edges else 'unresolved_no_surface'})
    interpretation={'schema_version':'1.0.0','status':'partial_geometry','part_type':part_type,'page':page['page'],
        'axis':{'orientation':axis,'coordinate_pdf':center,'method':section['proof'],'source_radial_sign':source_sign},
        'scale_pt_per_mm':scale,'origin_pdf':origin,'total_length_mm':total,
        'length_scope':'selected_confirmed_section_not_necessarily_entire_part',
        'selected_profile_id':section['profile_id'],'profile_r_z_mm':profile,
        'source_section_polygon_pdf':source_poly.tolist(),'source_full_bounds_pdf_canonical':box,
        'dimension_bindings':[{'dimension_id':o['dimension']['id'],**o['association'],
            'status':'scale_and_topology_confirmed' if not o['axial'] or o['dimension']['id'] in grounded_ids else 'scale_only_disconnected_chain'} for o in observations],
        'constraints':{'nodes':nodes,'equations':equations,'point_sources':constraints,'conflicts':conflicts},
        'unresolved_features':unresolved,'validation_expectations':expectations,
        'repeated_hole_pattern_evidence':pattern_proofs,
        'warnings':['partial_geometry: подтверждено выбранное сечение вращения, полнота всей детали не доказана.',
                    'Координаты недоопределённых переходов измерены с изображения и не заявляются номинальными размерами.']}
    intent=CADIntent(provenance='automatic_pdf',reconstruction_status='partial_geometry',
        description='Автоматически восстановлен профиль вращения по замкнутому контуру и явным размерным связям PDF.',
        operations=[{'id':'reconstructed_section','type':'revolve','profile':profile}],result='reconstructed_section',
        evidence={'page':page['page'],'profile_id':section['profile_id'],'axis':interpretation['axis'],
                  'scale_pt_per_mm':scale,'dimension_bindings':interpretation['dimension_bindings'],
                  'source_section_polygon_pdf':source_poly.tolist(),'origin_pdf':origin},
        validation_expectations=expectations,unresolved_features=unresolved)
    return intent.model_dump(mode='json'),interpretation


def interpret_page(page):
    from .strategies import material_sections,prismatic,clarification
    attempts=[]
    intent,report=interpret_rotational(page)
    attempts.append({'strategy':'mirrored_rotational_section','status':report['status'],
                     'reason':report.get('reason'),'conflicts':report.get('conflicts',[])})
    initial=(intent,report) if intent else None
    hypotheses=material_sections(page)
    candidates=[]
    for section in hypotheses:
        candidate,diagnostic=interpret_rotational(page,[section],allow_base_envelope=True)
        attempts.append({'strategy':'dimension_datum_section_graph','profile_id':section['profile_id'],
                         'axis':section['axis'],'status':diagnostic['status'],'reason':diagnostic.get('reason'),
                         'conflicts':diagnostic.get('conflicts',[])})
        if candidate:
            score=len(diagnostic['dimension_bindings'])
            candidates.append((score,section['full_bounds_canonical'][2]-section['full_bounds_canonical'][0],candidate,diagnostic))
    if candidates:
        candidates.sort(key=lambda c:(-c[1],-c[0]));_,_,intent,report=candidates[0]
        if initial:
            old_intent,old_report=initial
            if report['total_length_mm']<=old_report['total_length_mm']*1.05 or abs(report['axis']['coordinate_pdf']-old_report['axis']['coordinate_pdf'])>2:
                intent,report=old_intent,old_report
        alternatives=[];signatures={str(intent['operations'])}
        for _,_,other,diagnostic in candidates:
            signature=str(other['operations'])
            if signature in signatures:continue
            signatures.add(signature);alternatives.append({'intent':other,'interpretation':dict(diagnostic)})
            if len(alternatives)>=5:break
        report['alternative_hypotheses']=alternatives
        report['strategy']='dimension_datum_rotational';report['strategy_attempts']=attempts
        return intent,report
    if initial:
        intent,report=initial;report['strategy']='rotational';report['strategy_attempts']=attempts
        return intent,report
    intent,report=prismatic(page)
    attempts.append({'strategy':'prismatic_multiview','status':report['status'],'reason':report.get('reason')})
    if intent:
        report['strategy_attempts']=attempts;return intent,report
    from .primitive_reconstruction import reconstruct_fragments
    intent,report=reconstruct_fragments(page)
    attempts.append({'strategy':'dimension_bounded_open_segments','status':report['status'],'reason':report.get('reason')})
    if intent:
        report['strategy_attempts']=attempts;return intent,report
    return None,clarification(page,attempts)


def reconstruct(analysis,output:Path):
    from .multiview import analyze_views
    from .feature_reconstruction import augment_features
    results=[];intent=None;chosen=None;alternatives=[]
    for page in analysis['pages']:
        page['layout']['multi_view']=analyze_views(page)
        page['layout']['axes']=page['layout']['multi_view']['axes']
        candidate,interpretation=interpret_page(page)
        candidate,interpretation=augment_features(page,candidate,interpretation)
        page_alternatives=interpretation.pop('alternative_hypotheses',[])
        interpretation['multi_view']=page['layout']['multi_view']
        results.append(interpretation)
        if candidate and intent is None:
            intent,chosen=candidate,interpretation;alternatives=page_alternatives
        elif candidate:
            alternatives.append({'intent':candidate,'interpretation':interpretation})
    report={'drawing_id':analysis['drawing_id'],'status':chosen['status'] if chosen else 'needs_review',
            'selected_page':chosen['page'] if chosen else None,'pages':results,
            'message':'Автоматически восстановлена частичная геометрия по PDF.' if chosen else 'Недостаточно подтверждённых данных для автоматического 3D-построения.'}
    if intent:
        intent['evidence']['source_sha256']=analysis['sha256']
        intent['evidence']['drawing_id']=analysis['drawing_id']
    for alternative in alternatives:
        alternative['intent']['evidence'].update(source_sha256=analysis['sha256'],drawing_id=analysis['drawing_id'])
    atomic_json(output/'cad-candidates.json',{'policy':'Only independently evidenced alternatives; tried only if CAD or source validation rejects the primary.',
                'alternatives':alternatives[:6]})
    atomic_json(output/'interpretation.json',report)
    atomic_json(output/'cad-intent.json',intent if intent else {'status':report['status'],'operations':[],
                'reason':report['message'],'provenance':'automatic_pdf','unresolved':results})
    return report,intent
