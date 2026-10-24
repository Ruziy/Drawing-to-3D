"""Compare actual model measurements, source coverage and uncertainty separately."""
import numpy as np


def protected_nominals(old_attempt,new_attempt):
    old=old_attempt.get('validation',{});new=new_attempt.get('validation',{})
    inventory=new.get('cylindrical_surface_inventory',[]);extents=new.get('geometry',{}).get('bounding_extents_mm',[])
    planes=new.get('topological_planes_by_axis_mm',{})
    results=[]
    for check in old.get('dimensions',[]):
        if check.get('status')!='passed':continue
        kind=check['type'];proof={'type':kind,'dimension_id':check.get('dimension_id'),'passed':False}
        nominal=check.get('nominal_mm')
        if kind=='bounding_extent' and extents:
            error=min(abs(value-nominal) for value in extents)
            proof.update(passed=error<.03,error_mm=error,scope='actual_final_bounding_extent_rotation_invariant')
        elif kind in ('cylindrical_surface','semantic_cylinder'):
            radius=nominal/2 if nominal is not None else check['radius_mm']
            previous=[s for s in old.get('cylindrical_surface_inventory',[]) if abs(s['radius_mm']-radius)<.015]
            roles={s['role'] for s in previous if s['role']!='unknown'}
            matching=[s for s in inventory if abs(s['radius_mm']-radius)<.015 and (not roles or s['role'] in roles)]
            if kind=='semantic_cylinder':
                length=check['length_mm']
                matching=[s for s in matching if abs(s['axial_span_mm']-length)<.03 or
                    (extents and abs(extents[int(np.argmax(np.abs(s['axis'])))]-length)<.03)]
            proof.update(passed=bool(matching),radius_mm=radius,material_roles=sorted(roles),scope='actual_cylindrical_surface_radius_and_material_side')
        elif kind=='axial_distance' and nominal is not None:
            errors=[]
            for values in planes.values():
                coordinates=np.array(values,float)
                if len(coordinates)>600:continue
                errors.append(float(np.min(abs(abs(coordinates[:,None]-coordinates)-nominal))))
            proof.update(passed=bool(errors) and min(errors)<.03,error_mm=min(errors) if errors else None,
                         scope='actual_topological_plane_distance_rotation_invariant')
        elif kind in ('coaxial_bore_depth','semantic_void'):
            radius=check['radius_mm'];depth=check.get('depth_mm')
            matching=[s for s in inventory if s['role']=='inner' and abs(s['radius_mm']-radius)<.015 and abs(s['axial_span_mm']-depth)<.03]
            proof.update(passed=bool(matching),radius_mm=radius,depth_mm=depth,scope='actual_internal_cylindrical_face_and_depth')
        elif kind in ('analytic_profile_radius','hole_pattern'):
            equivalents=[c for c in new.get('dimensions',[]) if c.get('status')=='passed' and c['type']==kind]
            if kind=='analytic_profile_radius':equivalents=[c for c in equivalents if abs(c['nominal_mm']-check['nominal_mm'])<.03]
            else:equivalents=[c for c in equivalents if c.get('centers_xy_mm')==check.get('centers_xy_mm') and abs(c['radius_mm']-check['radius_mm'])<.015]
            proof.update(passed=bool(equivalents),scope='native_feature_check')
        else:proof.update(scope='unsupported_protection_requires_exact_shared_dimension_id')
        results.append(proof)
    return results


def measured_quality(attempt,candidate):
    e=candidate['intent']['evidence'];v=attempt.get('validation',{})
    checks=v.get('dimensions',[])
    dims={c.get('dimension_id') for c in checks if c.get('dimension_id') and c.get('status')=='passed'}
    dims|={p['dimension_id'] for p in e.get('semantic_parameters',[]) if p.get('dimension_id') and p['status']=='verified_dimensions'}
    features=0
    for c in checks:
        if c.get('status')!='passed':continue
        if c['type'] in ('semantic_void','coaxial_bore_depth','analytic_profile_radius'):features+=1
        elif c['type']=='hole_pattern':features+=len(c.get('centers_xy_mm',[]))
    views=[c for c in v.get('view_comparisons',[]) if c.get('status')=='compared']
    iou=min((c['intersection_over_union'] for c in views),default=v.get('source_comparison',{}).get('intersection_over_union',0))
    return {'passed_dimension_ids':sorted(dims),'measured_features':features,'registered_views':len(views),
            'minimum_registered_iou':iou,'estimated_parameters':len(e.get('dimension_groups',{}).get('estimated_dimensions',[]))}


def compare_candidates(new_attempt,new,old_attempt,old):
    before=measured_quality(old_attempt,old);after=measured_quality(new_attempt,new)
    protected=protected_nominals(old_attempt,new_attempt)
    exact_ids=set(after['passed_dimension_ids']).issuperset(before['passed_dimension_ids'])
    for proof in protected:
        if proof.get('scope')=='unsupported_protection_requires_exact_shared_dimension_id' and proof.get('dimension_id') in after['passed_dimension_ids']:
            proof.update(passed=True,scope='exact_shared_passed_dimension_id_for_unhandled_feature')
    preserved=all(c['passed'] for c in protected) if protected else exact_ids
    level=new['intent'].get('reconstruction_status')
    minimum=.75 if level=='preliminary_geometry' and after['measured_features']>before['measured_features'] else .82
    more_evidence=(len(after['passed_dimension_ids'])>len(before['passed_dimension_ids']) or
                   after['measured_features']>before['measured_features'] or after['registered_views']>before['registered_views'])
    same_views=new['intent']['evidence'].get('page')==old['intent']['evidence'].get('page')
    projection_ok=after['minimum_registered_iou']>=minimum
    if not more_evidence:projection_ok&=after['minimum_registered_iou']>before['minimum_registered_iou']+.005
    uncertainty_ok=after['estimated_parameters']<=before['estimated_parameters'] or after['measured_features']>before['measured_features']
    passed=preserved and projection_ok and uncertainty_ok and same_views and (more_evidence or after['minimum_registered_iou']>before['minimum_registered_iou']+.005)
    return passed,{'selected':passed,'before':before,'after':after,'protected_nominal_checks':protected,
                   'same_source_page':same_views,'protected_evidence_preserved':preserved,'registered_projection_sufficient':bool(projection_ok),
                   'uncertainty_justified':uncertainty_ok,'policy':'geometry_invariants_then_preserved_nominals_then_measured_features_views_and_assumptions'}
