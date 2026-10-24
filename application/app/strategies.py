"""Evidence-gated alternative constructions. No corpus names or reference values."""
import math
import re
import numpy as np
import cv2

from .schemas import CADIntent


def material_sections(page):
    """A diameter line supplies a datum even when only one half is sectioned.

    A closed hatched face or a physical centreline-graph face must agree with
    an axial dimension and a coaxial diameter at the same local view scale.
    """
    from .reconstruction import canonical,bounds,clip_half,relevant_associations,choose_scale
    faces=page['layout']['profile_candidates']+page['layout'].get('contour_graph',{}).get('faces',[])
    out=[];seen=set()
    for axis in ('horizontal','vertical'):
        datums=[]
        for d in page['dimensions']:
            if d['type']!='diameter' or d['nominal_value']<=0:continue
            for a in d['associations']:
                if a['orientation']==axis:continue
                p=canonical(a['endpoints_pdf'],axis)
                center=float(np.mean(p[:,1]));scale=a['scale_pt_per_mm']
                if any(abs(center-c)<.5 and abs(scale/s-1)<.015 for c,s in datums):continue
                datums.append((center,scale))
        for face in faces:
            if face['type']!='material_section_candidate':continue
            if face.get('area_pt2',0)<80:continue
            original=canonical(face['polygon_pdf'],axis);bb=bounds(original)
            if bb[2]-bb[0]<15:continue
            for center,datum_scale in datums:
                if not bb[1]-2<=center<=bb[3]+max(bb[3]-bb[1],2)*3 and not bb[1]-max(bb[3]-bb[1],2)*3<=center<=bb[3]+2:continue
                poly=original.copy()
                source_sign=1 if np.mean(poly[:,1])>center else -1
                if source_sign==1:poly[:,1]=2*center-poly[:,1]
                poly=clip_half(poly,center)
                if len(poly)<3 or cv2.contourArea(poly.astype(np.float32))<40:continue
                b=bounds(poly)
                axial_support=sum(abs(q[0]-p[0]) for p,q in zip(poly,np.roll(poly,-1,axis=0))
                                  if abs(q[1]-p[1])<.25 and center-(p[1]+q[1])/2>1)
                if axial_support<(b[2]-b[0])*.35:continue
                # A closed circle/rectangle on an end view is not by itself a
                # longitudinal section. Require physical axial and radial spans.
                full=[b[0],b[1],b[2],2*center-b[1]]
                section={'axis':axis,'axis_coordinate_pdf':center,'polygon_pdf_canonical':poly,
                         'profile_id':face['id'],'proof':'dimension_datum_and_hatched_or_graph_face',
                         'full_bounds_canonical':full,'face_method':face.get('method','provided_contour'),'source_radial_sign':source_sign}
                observations=relevant_associations(page,section);scaled=choose_scale(observations)
                if not scaled or abs(scaled[1]/datum_scale-1)>.02:continue
                if page['layout'].get('method','').startswith('raster') and len(scaled[2])<3:continue
                radial=[o for o in scaled[2] if not o['axial'] and o['dimension']['type']=='diameter']
                if not radial:continue
                # A numerical diameter must also reach an observed surface.
                radii=center-poly[:,1]
                if max(radii)>max(o['dimension']['nominal_value'] for o in radial)*scaled[1]/2*1.7:continue
                if not any(np.min(abs(radii-o['dimension']['nominal_value']*scaled[1]/2))<max(1.7,scaled[1]*.25) for o in radial):continue
                branch_points=[]
                multi=page['layout'].get('multi_view',{})
                for branch in multi.get('orthogonal_axis_intersections',[]):
                    point=canonical([branch['point_pdf']],axis)[0]
                    frame=next((f for f in multi['frames'] if f['id']==branch['view_id']),None)
                    if not frame:continue
                    fb=canonical([[frame['bbox'][0],frame['bbox'][1]],[frame['bbox'][2],frame['bbox'][3]]],axis)
                    above=center-float(fb[:,1].min());below=float(fb[:,1].max())-center
                    if min(above,below)>2 and max(above,below)>min(above,below)*2 and abs(branch['scale_pt_per_mm']/scaled[1]-1)<.025:
                        branch_points.append(point[0])
                key=(axis,round(center,1),tuple(round(v,1) for v in b))
                if key in seen:continue
                seen.add(key)
                if not any(b[0]+2<u<b[2]-2 for u in branch_points):out.append(section)
                # A material face can extend beyond the portion bounded by an
                # explicit dimension. Cutting it at those planes yields a
                # confirmed fragment, without inventing the missing lengths.
                for observation in scaled[2]:
                    if not observation['axial']:continue
                    lo,hi=sorted(observation['endpoints'][:,0])
                    if any(lo+2<u<hi-2 for u in branch_points):continue
                    if lo<b[0]-2 or hi>b[2]+2 or hi-lo<(b[2]-b[0])*.15:continue
                    if abs(lo-b[0])<2 and abs(hi-b[2])<2:continue
                    clipped=clip_axial(poly,lo,hi)
                    if len(clipped)<3:continue
                    cb=bounds(clipped)
                    if max(abs(cb[0]-lo),abs(cb[2]-hi))>2:continue
                    fragment={**section,'polygon_pdf_canonical':clipped,'full_bounds_canonical':[lo,cb[1],hi,2*center-cb[1]],
                              'profile_id':face['id']+':'+observation['dimension']['id'],
                              'fragment_dimension_id':observation['dimension']['id']}
                    out.append(fragment)
    return out


def clip_axial(poly,lo,hi):
    result=np.asarray(poly,float)
    for limit,sign in ((lo,1),(hi,-1)):
        points=[]
        for a,b in zip(result,np.roll(result,-1,axis=0)):
            first=sign*(a[0]-limit)>=-1e-8;second=sign*(b[0]-limit)>=-1e-8
            if first:points.append(a.tolist())
            if first!=second:
                t=(limit-a[0])/(b[0]-a[0]);points.append((a+t*(b-a)).tolist())
        result=np.array(points,float)
        if len(result)<3:break
    return result


def matched_span(page,box,orientation):
    index=0 if orientation=='horizontal' else 1;other=1-index;result=[]
    for d in page['dimensions']:
        if d['type']!='linear' or d['nominal_value']<=0:continue
        for a in d['associations']:
            if a['orientation']!=orientation:continue
            ends=np.array(a['endpoints_pdf']);lo,hi=sorted(ends[:,index])
            if max(abs(lo-box[index]),abs(hi-box[index+2]))>2.3:continue
            distance=min(abs(np.mean(ends[:,other])-box[other]),abs(np.mean(ends[:,other])-box[other+2]))
            if distance>min(250,max(85,max(box[2]-box[0],box[3]-box[1])*.85)):continue
            result.append((distance+a['score_distance_pt'],d,a))
    return sorted(result,key=lambda x:x[0])


def prismatic(page):
    """Extrude an observed profile only with a dimensioned, aligned side view."""
    from .reconstruction import clean_polygon,profile_intersections
    faces=[c for c in page['layout'].get('contour_graph',{}).get('faces',[])+page['layout']['profile_candidates'] if c['area_pt2']>100]
    choices=[]
    for face in faces:
        b=face['bbox'];poly=np.array(face['polygon_pdf'],float)
        area=cv2.contourArea(poly.astype(np.float32));rect_area=(b[2]-b[0])*(b[3]-b[1])
        # A circle/end-view cannot establish a uniform extrusion; a separate
        # section/side view is always required. Reject circular rotational parts.
        perimeter=cv2.arcLength(poly.astype(np.float32),True)
        if 4*math.pi*area/max(perimeter*perimeter,1)>.87:continue
        rotational_span=False
        for d in page['dimensions']:
            if d['type']!='diameter':continue
            for a in d['associations']:
                index=0 if a['orientation']=='horizontal' else 1
                ends=np.array(a['endpoints_pdf'])[:,index]
                if max(abs(float(ends.min())-b[index]),abs(float(ends.max())-b[index+2]))<2.3:rotational_span=True
        if rotational_span:continue
        for orientation in ('horizontal','vertical'):
            for _,dim,association in matched_span(page,b,orientation):
                scale=association['scale_pt_per_mm'];index=0 if orientation=='horizontal' else 1
                for side in faces:
                    if side['id']==face['id']:continue
                    sb=side['bbox'];sp=np.array(side['polygon_pdf'],np.float32)
                    sarea=(sb[2]-sb[0])*(sb[3]-sb[1])
                    if cv2.contourArea(sp)/max(sarea,1)<.90:continue
                    # Orthographic side view shares actual sheet coordinates
                    # along the extrusion profile, not merely a similar size.
                    aligned=None
                    if max(abs(sb[1]-b[1]),abs(sb[3]-b[3]))<2.3 and (sb[0]>b[2]+10 or b[0]>sb[2]+10):aligned='horizontal'
                    if max(abs(sb[0]-b[0]),abs(sb[2]-b[2]))<2.3 and (sb[1]>b[3]+10 or b[1]>sb[3]+10):aligned='vertical'
                    if not aligned:continue
                    for _,thickness,ta in matched_span(page,sb,aligned):
                        if abs(ta['scale_pt_per_mm']/scale-1)>.025:continue
                        if thickness['nominal_value']>max(b[2]-b[0],b[3]-b[1])/scale*.6:continue
                        profile=(poly-np.array([b[0],b[1]]))/scale
                        # Pin the explicitly dimensioned envelope, keeping
                        # intermediate curve points as measured estimates.
                        nominal=dim['nominal_value'];span=b[index+2]-b[index]
                        profile[:,index]=(poly[:,index]-b[index])*nominal/span
                        other_index=1-index
                        additional=[record for record in matched_span(page,b,'vertical' if other_index else 'horizontal')
                                    if abs(record[2]['scale_pt_per_mm']/scale-1)<.025]
                        extra_binding=None
                        if additional:
                            _,extra_dimension,extra_association=additional[0]
                            other_span=b[other_index+2]-b[other_index]
                            profile[:,other_index]=(poly[:,other_index]-b[other_index])*extra_dimension['nominal_value']/other_span
                            extra_binding=(extra_dimension,extra_association)
                        profile=clean_polygon(profile.tolist())
                        if profile_intersections(profile):continue
                        unresolved=[{'type':'profile_curve_estimates','reason':'Контур измерен после калибровки; радиусы и локальные переходы требуют проверки.'},
                                    {'type':'uniform_extrusion_hypothesis','reason':'Постоянство сечения согласовано с боковой проекцией; скрытые локальные элементы не доказаны.'}]
                        if any(d['type'] in ('diameter','thread') for d in page['dimensions']):
                            unresolved.append({'type':'holes_not_cut','reason':'Распознанные отверстия/резьбы не вырезаны без подтверждённых координат и глубин; показано только базовое выдавливание.'})
                        bindings=[{'dimension_id':d['id'],**a,'status':'scale_and_topology_confirmed'} for d,a in ((dim,association),(thickness,ta))]
                        if extra_binding:
                            bindings.append({'dimension_id':extra_binding[0]['id'],**extra_binding[1],'status':'scale_and_topology_confirmed'})
                        origin=[b[0],b[1]]
                        evidence={'page':page['page'],'profile_id':face['id'],'strategy':'prismatic',
                                  'scale_pt_per_mm':scale,'source_profile_polygon_pdf':poly.tolist(),
                                  'profile_origin_pdf':origin,'profile_axis_scales_pt_per_mm':[(span/nominal if index==0 else scale),(span/nominal if index==1 else scale)],
                                  'side_view_id':side['id'],'dimension_bindings':bindings}
                        expectations=[{'type':'bounding_extent','axis':index,'nominal_mm':nominal,'dimension_id':dim['id']},
                                      {'type':'bounding_extent','axis':2,'nominal_mm':thickness['nominal_value'],'dimension_id':thickness['id']}]
                        if extra_binding:
                            evidence['profile_axis_scales_pt_per_mm'][other_index]=other_span/extra_binding[0]['nominal_value']
                            expectations.append({'type':'bounding_extent','axis':other_index,'nominal_mm':extra_binding[0]['nominal_value'],'dimension_id':extra_binding[0]['id']})
                        intent=CADIntent(provenance='automatic_pdf',reconstruction_status='partial_geometry',
                                         description='Профиль и согласованная боковая проекция: призматическое выдавливание.',
                                         operations=[{'id':'profile_pad','type':'extrude','profile':profile,'height':thickness['nominal_value']}],result='profile_pad',
                                         evidence=evidence,validation_expectations=expectations,unresolved_features=unresolved).model_dump(mode='json')
                        report={'page':page['page'],'status':'partial_geometry','strategy':'prismatic','part_type':'prismatic',
                                'dimension_bindings':bindings,'unresolved_features':unresolved,'profile_xy_mm':profile,
                                'scale_pt_per_mm':scale,'view_correspondence':{'profile_id':face['id'],'side_view_id':side['id'],'shared_sheet_axis':'y' if aligned=='horizontal' else 'x'},
                                'warnings':['Частичная геометрия: контур и толщина подтверждены двумя согласованными видами; полнота локальных элементов не доказана.']}
                        choices.append((area,intent,report))
    if choices:
        choices.sort(key=lambda c:-c[0]);return choices[0][1:]
    return None,{'status':'needs_review','strategy':'prismatic','reason':'Призматическая стратегия: требуется профиль с размерной калибровкой и согласованная проекция с явной толщиной.'}


def clarification(page,attempts):
    dimensions=page['dimensions'];layout=page['layout']
    missing=[]
    conflicts=[a for a in attempts if a.get('status')=='conflict' and a.get('conflicts')]
    for reason in dict.fromkeys(a['reason'] for a in conflicts if a.get('reason')):
        missing.append({'parameter':'ambiguous_profile_or_local_feature','question':reason,
                        'source':'rejected_geometry_hypotheses'})
    multi=layout.get('multi_view',{})
    if multi.get('orthogonal_axis_intersections'):
        missing.append({'parameter':'intersecting_channels_and_body_sections',
            'question':'Обнаружены пересекающиеся оси. Уточните наружные сечения ветвей, привязку их длин и глубины каналов; соединение тел пока не доказано.',
            'source':'multi_axis_constraints','axis_intersections':multi['orthogonal_axis_intersections']})
        attempts.append({'strategy':'multi_axis_constructive','status':'needs_review',
            'reason':'Проверены оси и локальные размерные системы; для объединения ветвей недостаточно подтверждённых наружных профилей и осевых привязок.'})
    if not any(d['type']=='diameter' for d in dimensions):
        missing.append({'parameter':'diameter_annotations','question':'Проверьте диаметры на основном виде: OCR не подтвердил знак Ø и числовой номинал.','source':'ocr'})
    if not any(d['type']=='linear' and d['associations'] for d in dimensions):
        missing.append({'parameter':'axial_length_or_thickness','question':'Укажите длину участка или толщину и его границы на основном виде.','source':'dimension_association'})
    if not layout.get('profile_candidates'):
        missing.append({'parameter':'profile_boundary','question':'Уточните границы основного вида: физические линии не образуют проверяемую область.','source':'contour_topology'})
    if not missing:
        missing.append({'parameter':'view_and_dimension_correspondence','question':'Уточните основной вид и принадлежность его осевых размеров; доступные гипотезы не согласованы.','source':'geometry_constraints'})
    return {'page':page['page'],'status':'needs_review','strategy_attempts':attempts,'clarifications':missing,
            'reason':' '.join(m['question'] for m in missing),'diagnostics':{
                'dimension_count':len(dimensions),'diameter_count':sum(d['type']=='diameter' for d in dimensions),
                'associated_dimension_count':sum(bool(d['associations']) for d in dimensions),
                'closed_profile_count':len(layout.get('profile_candidates',[])),
                'graph_open_vertex_count':len(layout.get('contour_graph',{}).get('open_vertices',[]))}}
