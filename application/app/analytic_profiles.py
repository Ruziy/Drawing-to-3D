"""Fit real circular profile arcs, keeping baseline dimensional constraints."""
import copy
import math
import numpy as np
from .schemas import CADIntent


def fit_circle(points):
    points=np.array(points,float);matrix=np.c_[2*points,np.ones(len(points))];rhs=(points*points).sum(axis=1)
    solution,*_=np.linalg.lstsq(matrix,rhs,rcond=None);center=solution[:2]
    radius_squared=solution[2]+center@center
    if radius_squared<=0:return None
    radius=math.sqrt(radius_squared);residual=float(np.max(abs(np.linalg.norm(points-center,axis=1)-radius)))
    return center,radius,residual


def curved_segments(profile,nominals=(),source_tolerance=.05):
    p=np.array(profile,float)
    lengths=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
    start=(int(np.argmax(lengths))+1)%len(p);p=np.roll(p,-start,axis=0)
    angles=[]
    for previous,current,next_point in zip(np.roll(p,1,axis=0),p,np.roll(p,-1,axis=0)):
        a=current-previous;b=next_point-current
        angles.append(math.atan2(float(a[0]*b[1]-a[1]*b[0]),float(a@b)))
    arcs={};index=1
    while index<len(p)-1:
        angle=angles[index]
        if not math.radians(1)<abs(angle)<math.radians(40):index+=1;continue
        end=index
        while end+1<len(p)-1 and angles[end+1]*angle>0 and math.radians(1)<abs(angles[end+1])<math.radians(40):end+=1
        if end-index<2:index=end+1;continue
        lo=index-1;hi=end+1
        spacing=np.linalg.norm(np.diff(p[lo:hi+1],axis=0),axis=1);typical=float(np.median(spacing))
        if len(spacing)>3 and spacing[0]>typical*2.5:lo+=1
        if len(spacing)>3 and spacing[-1]>typical*2.5:hi-=1
        points=p[lo:hi+1];fitted=fit_circle(points) if len(points)>=4 else None
        if fitted:
            center,radius,residual=fitted
            if residual<=source_tolerance and radius>.1:
                raw_angles=np.unwrap(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))
                sweep=float(raw_angles[-1]-raw_angles[0])
                if math.radians(15)<abs(sweep)<math.radians(175):
                    nominal=min(nominals,key=lambda r:abs(r-radius)) if nominals else None
                    used_nominal=False
                    if nominal is not None and abs(nominal-radius)<max(.15,radius*.04):
                        if np.max(abs(np.linalg.norm(points-center,axis=1)-nominal))<=max(source_tolerance*2,.15):
                            radius=nominal;used_nominal=True
                    # Stroke-edge samples are moved only within the measured
                    # raster uncertainty to the fitted/constrained circle.
                    p[lo]=center+radius*(p[lo]-center)/np.linalg.norm(p[lo]-center)
                    p[hi]=center+radius*(p[hi]-center)/np.linalg.norm(p[hi]-center)
                    midpoint=center+radius*np.array([math.cos(raw_angles[0]+sweep/2),math.sin(raw_angles[0]+sweep/2)])
                    arcs[lo]={'end':hi,'points':[p[lo].tolist(),midpoint.tolist(),p[hi].tolist()],
                              'radius_mm':float(radius),'fit_residual_mm':residual,'explicit_radius_constraint':used_nominal}
        index=end+1
    segments=[];proof=[];index=0
    while index<len(p):
        if index in arcs:
            arc=arcs[index];segments.append({'kind':'arc','points':arc['points']});proof.append({k:v for k,v in arc.items() if k!='points'});index=arc['end']
        else:
            following=(index+1)%len(p)
            if np.linalg.norm(p[following]-p[index])>1e-8:segments.append({'kind':'line','points':[p[index].tolist(),p[following].tolist()]})
            index+=1
    return segments,proof


def constrain_arc_extents(segments,profile,expectations,tolerance):
    """Keep authoritative extents without turning circular arcs into ellipses."""
    p=np.array(profile,float);limits={e['axis']:(float(p[:,e['axis']].min()),float(p[:,e['axis']].min()+e['nominal_mm']))
        for e in expectations if e['type']=='bounding_extent' and e['axis'] in (0,1)}
    moved=[]
    for index,segment in enumerate(segments):
        if segment['kind']!='arc':continue
        points=np.array(segment['points']);center,radius,_=fit_circle(points)
        angles=np.unwrap(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]));lo,hi=min(angles[0],angles[-1]),max(angles[0],angles[-1])
        extrema=list(points)
        for angle in np.arange(-4*np.pi,4*np.pi,np.pi/2):
            if lo<=angle<=hi:extrema.append(center+radius*np.array([math.cos(angle),math.sin(angle)]))
        extrema=np.array(extrema);delta=np.zeros(2)
        for axis,(minimum,maximum) in limits.items():
            low=float(extrema[:,axis].min());high=float(extrema[:,axis].max())
            if low<minimum:delta[axis]=minimum-low
            if high>maximum:delta[axis]=maximum-high
        if max(abs(delta))>tolerance*2:raise ValueError('Arc fit contradicts an explicit profile extent')
        if np.any(delta):
            if segments[index-1]['kind']!='line' or segments[(index+1)%len(segments)]['kind']!='line':raise ValueError('Cannot safely constrain adjacent arcs')
            points+=delta;segment['points']=points.tolist()
            segments[index-1]['points'][-1]=points[0].tolist();segments[(index+1)%len(segments)]['points'][0]=points[-1].tolist()
            moved.append({'arc_index':index,'translation_mm':delta.tolist(),'reason':'explicit_envelope_constraint_within_original_stroke_uncertainty'})
    return moved


def analytic_candidate(source,page):
    intent=copy.deepcopy(source['intent']);evidence=intent['evidence']
    nominals=[d['nominal_value'] for d in page['dimensions'] if d['type']=='radius' and d['nominal_value']>0]
    all_proof=[]
    for operation in intent['operations']:
        if operation['type'] not in ('extrude','revolve') or operation.get('profile_segments'):continue
        scale=evidence.get('scale_pt_per_mm') or min(evidence.get('profile_axis_scales_pt_per_mm',[3,3]))
        segments,proof=curved_segments(operation['profile'],nominals,max(.035,1/(4.167*scale)))
        if not proof:continue
        if operation['type']=='extrude':
            shifts=constrain_arc_extents(segments,operation['profile'],intent['validation_expectations'],max(.035,1/(4.167*scale)))
            evidence.setdefault('analytic_extent_constraints',[]).extend(shifts)
        operation['profile_segments']=segments;all_proof+=proof
        for arc in proof:
            intent['validation_expectations'].append({'type':'analytic_profile_radius','nominal_mm':arc['radius_mm'],
                'required':True,'source':'explicit_OCR_radius_and_measured_curve' if arc['explicit_radius_constraint'] else 'measured_circle_fit_not_verified_design_nominal'})
    if not all_proof:return None
    evidence.update(analytic_profile=True,analytic_arc_evidence=all_proof,protected_baseline_operations=True,
                    implemented_analytic_arc_count=len(all_proof))
    intent['unresolved_features'].append({'type':'analytic_profile_scope',
        'reason':'Круговые участки построены аналитически. Радиусы без прочитанного номинала остаются оценками по контуру; полнота детали не доказана.'})
    intent=CADIntent.model_validate(intent).model_dump(mode='json')
    return {'intent':intent,'interpretation':copy.deepcopy(source.get('interpretation')),'origin':'hybrid_analytic_profile'}
