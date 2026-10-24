"""Constructive feature tree: a confirmed base followed by evidenced cuts."""
import math
import re
import numpy as np

from .schemas import CADIntent


def material_intervals(profile,radius):
    """Exact axial intervals at a radial coordinate in a polygonal base."""
    levels=[]
    for a,b in zip(profile,profile[1:]+profile[:1]):
        if (a[0]<=radius<b[0]) or (b[0]<=radius<a[0]):
            levels.append(a[1]+(radius-a[0])/(b[0]-a[0])*(b[1]-a[1]))
    levels=sorted(levels)
    return [(a,b) for a,b in zip(levels[::2],levels[1::2]) if b-a>.1]


def augment_features(page,intent,report):
    """Never guess tapped drill diameters, blind depths, or a flange side.

    An automatic pattern requires an explicit through-hole declaration,
    nominal diameter and PCD, a regular observed end-view pattern, and exactly
    one material interval. Ambiguous features remain detailed review items.
    """
    if not intent:return intent,report
    attempts=[];profile=report.get('profile_r_z_mm');scale=report.get('scale_pt_per_mm')
    proofs=report.get('repeated_hole_pattern_evidence',[])
    if not profile or not proofs:
        report.setdefault('strategy_attempts',[]).append({'strategy':'flange_and_combined_features','status':'needs_review',
            'reason':'Нет согласованных делительной окружности, диаметра отверстий и торцевого рисунка для автоматического вырезания.'})
        return intent,report
    circles=page['layout'].get('end_view_circle_witnesses',[])
    for proof in proofs:
        pitch=proof['pitch_circle_diameter_mm'];center=np.array(proof['circle_center_pdf'],float)
        for count in set(proof['declared_quantities']):
            if count<3:continue
            locations=np.array(proof['observed_hole_centers_pdf'],float)
            angles=np.arctan2(locations[:,1]-center[1],locations[:,0]-center[0])
            step=2*math.pi/count
            for seed in angles:
                expected=np.array([seed+k*step for k in range(count)])
                distances=abs(np.angle(np.exp(1j*(expected[:,None]-angles[None,:]))))
                matches=distances.argmin(axis=1)
                if len(set(matches.tolist()))!=count or distances.min(axis=1).max()>math.radians(2):continue
                for dimension in page['dimensions']:
                    if dimension['type']!='diameter' or dimension['nominal_value']>=pitch*.3:continue
                    # A count and the through declaration must belong to the
                    # same text neighbourhood as the diameter annotation.
                    box=dimension['bbox'];label=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
                    words=[r['raw_text'] for r in page.get('text_runs',[]) if
                           np.linalg.norm(np.array([(r['bbox'][0]+r['bbox'][2])/2,(r['bbox'][1]+r['bbox'][3])/2])-label)<65]
                    text=' '.join(words)
                    if not re.search(rf'\b{count}\s*от[вб]',text,re.I):continue
                    if not re.search(r'сквоз',text,re.I):
                        attempts.append({'strategy':'flange_hole_pattern','status':'needs_review','pitch_circle_diameter_mm':pitch,
                            'hole_dimension_id':dimension['id'],'reason':'Не подтверждены сквозность/глубина и сторона отверстий.'});continue
                    radius=dimension['nominal_value']/2
                    observed=[]
                    for location in locations[matches]:
                        candidates=[c for c in circles if np.linalg.norm(np.array(c['center_pdf'])-location)<3 and abs(c['radius_pdf']/scale-radius)<.6]
                        if not candidates:break
                        observed.append(candidates[0])
                    if len(observed)!=count:continue
                    intervals=material_intervals(profile,pitch/2)
                    if len(intervals)!=1:
                        attempts.append({'strategy':'flange_hole_pattern','status':'needs_review','pitch_circle_diameter_mm':pitch,
                            'reason':'Делительная окружность пересекает несколько осевых участков; принадлежность отверстий фланцу неоднозначна.'});continue
                    lo,hi=intervals[0];phase=float(seed);x=pitch/2*math.cos(phase);y=pitch/2*math.sin(phase)
                    tool_id=f'pattern_tool_{len(intent["operations"])}';cut_id=f'pattern_cut_{len(intent["operations"])}'
                    operations=intent['operations']+[
                        {'id':tool_id,'type':'cylinder','radius':radius,'height':hi-lo+.2,'origin':[x,y,lo-.1]},
                        {'id':cut_id,'type':'circular_pattern','base':intent['result'],'tool':tool_id,'count':count}]
                    expected_centers=[[pitch/2*math.cos(phase+k*step),pitch/2*math.sin(phase+k*step)] for k in range(count)]
                    checks=intent['validation_expectations']+[{'type':'hole_pattern','centers_xy_mm':expected_centers,
                            'radius_mm':radius,'z_mm':(lo+hi)/2,'diameter_dimension_id':dimension['id'],'pcd_dimension_id':proof['dimension_id']}]
                    intent={**intent,'operations':operations,'result':cut_id,'validation_expectations':checks}
                    intent=CADIntent.model_validate(intent).model_dump(mode='json')
                    evidence={'strategy':'flange_hole_pattern','status':'partial_geometry','count':count,'diameter_mm':radius*2,
                              'pitch_circle_diameter_mm':pitch,'axial_interval_mm':[lo,hi],
                              'phase_radians':phase,'phase_source':'observed_end_view_regular_pattern_not_claimed_nominal_angle',
                              'dimension_ids':[dimension['id'],proof['dimension_id']]}
                    intent['evidence'].setdefault('constructed_features',[]).append(evidence)
                    report.setdefault('constructed_features',[]).append(evidence)
                    report.setdefault('strategy_attempts',[]).extend(attempts+[evidence])
                    report['strategy']='combined_revolve_and_hole_pattern'
                    return intent,report
    report.setdefault('strategy_attempts',[]).extend(attempts or [{'strategy':'flange_hole_pattern','status':'needs_review',
        'reason':'Рисунок отверстий найден; диаметр, регулярность, сквозность или осевой участок не подтверждены совместно.'}])
    return intent,report
