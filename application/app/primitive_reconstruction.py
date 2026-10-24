"""Dimension-bounded material fragments, without requiring any closed contour."""
import numpy as np
import cv2

from .constraints import solve_chain
from .schemas import CADIntent


def interval_coverage(intervals,lo,hi):
    cursor=lo;covered=0
    for a,b in sorted(intervals):
        a=max(lo,a,cursor);b=min(hi,b)
        if b>a:covered+=b-a;cursor=b
    return covered/max(hi-lo,1e-9)


def parallel_witnesses(page,axis,level,lo,hi,tolerance=.8):
    intervals=[];ids=[]
    for line in page['layout'].get('physical_boundary_segments',[]):
        ends=np.array(line['endpoints_pdf'],float)
        if axis=='vertical':ends=ends[:,::-1]
        if np.max(abs(ends[:,1]-level))>tolerance:continue
        a,b=sorted(ends[:,0]);intervals.append((a,b));ids.append(line['id'])
    return interval_coverage(intervals,lo,hi),ids


def reconstruct_fragments(page):
    """A pipe wall needs four measured parallel boundaries and a material proof.

    End closures are planes explicitly bounded by an axial dimension/chain,
    not attempts to connect arbitrary open endpoints. A solid cylinder is never
    inferred just because an inner bore dimension is missing.
    """
    from .reconstruction import canonical
    choices=[]
    for axis in ('horizontal','vertical'):
        radial=[];axial=[]
        for dimension in page['dimensions']:
            if dimension['nominal_value']<=0:continue
            for association in dimension['associations']:
                ends=canonical(association['endpoints_pdf'],axis)
                observation={'dimension':dimension,'association':association,'ends':ends}
                if dimension['type']=='diameter' and association['orientation']!=axis:radial.append(observation)
                elif dimension['type']=='linear' and association['orientation']==axis:axial.append(observation)
        for outer in radial:
            scale=outer['association']['scale_pt_per_mm'];center=float(np.mean(outer['ends'][:,1]))
            compatible=[o for o in radial if abs(o['association']['scale_pt_per_mm']/scale-1)<.018 and abs(np.mean(o['ends'][:,1])-center)<1]
            lengths=[o for o in axial if abs(o['association']['scale_pt_per_mm']/scale-1)<.018]
            if len({o['dimension']['id'] for o in compatible})<2 or not lengths:continue
            # Each interval is an explicit equation or an exactly solved chain.
            equations=[{'endpoints':o['ends'][:,0].tolist(),'value':o['dimension']['nominal_value'],'dimension_id':o['dimension']['id']} for o in lengths]
            origin=min(float(np.min(o['ends'][:,0])) for o in lengths)
            nodes,solved,conflicts=solve_chain(equations,origin)
            if conflicts or len(nodes)<2:continue
            nodes.sort(key=lambda n:n['pdf_position'])
            for ai,a in enumerate(nodes):
                for b in nodes[ai+1:]:
                    lo,hi=a['pdf_position'],b['pdf_position'];height=b['position_mm']-a['position_mm']
                    if height<1 or hi-lo<8:continue
                    # Four parallel ink strokes can also be a thread's major/
                    # minor diameters on a *solid* shaft. Require an observed
                    # unhatched central cavity before interpreting an inner
                    # surface as a bore.
                    cavity=False
                    for face in page['layout'].get('profile_candidates',[]):
                        if face['contour_depth']%2!=1 or face['hatch_line_count']!=0:continue
                        polygon=canonical(face['polygon_pdf'],axis).astype(np.float32)
                        if all(cv2.pointPolygonTest(polygon,(lo+(hi-lo)*t,center),False)>0 for t in (.25,.5,.75)):
                            cavity=True;break
                    if not cavity:continue
                    surfaces=[]
                    for observation in compatible:
                        radius=observation['dimension']['nominal_value']/2
                        first,one=parallel_witnesses(page,axis,center-radius*scale,lo,hi)
                        second,two=parallel_witnesses(page,axis,center+radius*scale,lo,hi)
                        if min(first,second)>=.94:surfaces.append((radius,observation,one+two))
                    # Only use adjacent measured surfaces; extra intermediate
                    # boundaries could denote a void or an unrelated feature.
                    surfaces.sort(key=lambda p:p[0])
                    for inner,outer_surface in zip(surfaces,surfaces[1:]):
                        inner_r,inner_o,inner_ids=inner;outer_r,outer_o,outer_ids=outer_surface
                        if outer_r-inner_r<.2:continue
                        witnesses=[]
                        for hatch in page['layout'].get('hatch_witnesses',[]):
                            point=canonical([hatch['point']],axis)[0];r=(center-point[1])/scale
                            if lo+.5<point[0]<hi-.5 and inner_r+.05<r<outer_r-.05:witnesses.append(hatch)
                        if len(witnesses)<3:continue
                        profile=[[inner_r,0],[outer_r,0],[outer_r,height],[inner_r,height]]
                        polygon=[[lo,center-inner_r*scale],[lo,center-outer_r*scale],
                                 [hi,center-outer_r*scale],[hi,center-inner_r*scale]]
                        ids={inner_o['dimension']['id'],outer_o['dimension']['id']}|set(a['dimension_ids'])|set(b['dimension_ids'])
                        bindings=[{'dimension_id':o['dimension']['id'],**o['association'],'status':'scale_and_topology_confirmed'}
                                  for o in compatible+lengths if o['dimension']['id'] in ids]
                        unresolved=[{'type':'bounded_material_fragment','reason':'Восстановлен только подтверждённый кольцевой участок между размерными плоскостями; остальные участки детали не построены.'}]
                        expectations=[{'type':'bounding_extent','axis':2,'nominal_mm':height,'dimension_ids':a['dimension_ids']+b['dimension_ids']}]
                        for radius,observation,_ in (inner,outer_surface):
                            expectations.append({'type':'cylindrical_surface','nominal_mm':radius*2,'dimension_id':observation['dimension']['id'],'axial_intervals_mm':[[0,height]]})
                        evidence={'page':page['page'],'strategy':'dimension_bounded_open_segments','axis':{'orientation':axis,'coordinate_pdf':center},
                                  'source_section_polygon_pdf':polygon,'scale_pt_per_mm':scale,'origin_pdf':lo,
                                  'dimension_bindings':bindings,'physical_boundary_ids':inner_ids+outer_ids,
                                  'material_hatch_witness_count':len(witnesses),'arbitrary_contour_closure':False}
                        intent=CADIntent(provenance='automatic_pdf',reconstruction_status='partial_geometry',
                            description='Подтверждённый кольцевой участок по четырём границам, штриховке и размерной цепи; исходный замкнутый контур не требуется.',
                            operations=[{'id':'confirmed_wall_fragment','type':'revolve','profile':profile}],result='confirmed_wall_fragment',
                            evidence=evidence,validation_expectations=expectations,unresolved_features=unresolved).model_dump(mode='json')
                        report={'status':'partial_geometry','page':page['page'],'strategy':'dimension_bounded_open_segments',
                                'part_type':'confirmed_tube_fragment','dimension_bindings':bindings,'axis':evidence['axis'],
                                'total_length_mm':height,'scale_pt_per_mm':scale,'profile_r_z_mm':profile,
                                'constraints':{'nodes':nodes,'equations':solved,'conflicts':[]},'unresolved_features':unresolved,
                                'selected_profile_id':'open_segment_graph','length_scope':'dimension_bounded_fragment'}
                        choices.append((height*(outer_r-inner_r),intent,report))
    if choices:
        choices.sort(key=lambda c:-c[0]);return choices[0][1:]
    return None,{'status':'needs_review','reason':'Сегментная стратегия: не подтверждены четыре границы стенки, материал и размерные плоскости одного участка.'}
