"""BRep measurements and registered section masks; no claimed full 3D accuracy."""
from pathlib import Path
import cv2
import numpy as np

from .reconstruction import clip_half
from .storage import atomic_json,read_json


def validate_model(model_dir:Path,intent):
    report=read_json(model_dir/'validation-brep.json')
    mesh=read_json(model_dir/'preview.json')
    vertices=np.array(mesh['vertices'],dtype=float);faces=np.array(mesh['triangles'],dtype=int)
    report['mesh']={'finite_vertices':bool(np.isfinite(vertices).all()),
        'valid_indices':bool(faces.min()>=0 and faces.max()<len(vertices)),'triangle_count':len(faces)}
    projections=[]
    for name,(a,b) in {'xy':(0,1),'xz':(0,2),'yz':(1,2)}.items():
        points=vertices[:,[a,b]];lo=points.min(axis=0);hi=points.max(axis=0)
        scale=900/max(float((hi-lo).max()),.01)
        transformed=(points-lo)*scale+20
        width,height=np.ceil((hi-lo)*scale+40).astype(int)
        image=np.zeros((max(height,1),max(width,1)),np.uint8)
        for tri in faces:cv2.fillConvexPoly(image,np.round(transformed[tri]).astype(np.int32),255)
        cv2.imencode('.png',255-image)[1].tofile(str(model_dir/f'projection-{name}.png'))
        projections.append({'plane':name,'file':f'projection-{name}.png','status':'saved_not_registered_to_a_source_view'})
    report['orthographic_projections']=projections
    evidence=intent.get('evidence',{})
    if evidence.get('independent_semantic_graph'):
        from .semantic_validation import compare
        report.update(compare(model_dir,mesh,intent))
    elif evidence.get('source_section_polygon_pdf'):
        source=np.array(evidence['source_section_polygon_pdf'],float)
        scale=evidence['scale_pt_per_mm'];origin=evidence['origin_pdf'];axis=evidence['axis']['coordinate_pdf']
        source_uv=source if evidence['axis']['orientation']=='horizontal' else source[:,::-1]
        # Source is already canonical (axial, transverse), irrespective of sheet rotation.
        source_uv=source
        wires=[]
        for wire in mesh.get('section_xz_wires',[]):
            points=np.array(wire,float)
            if len(points)<3:continue
            # Intersect true CAD section with positive radial half-space.
            canonical=np.c_[points[:,1],-points[:,0]]
            clipped=clip_half(canonical,0)
            if len(clipped)>2:
                sign=evidence['axis'].get('source_radial_sign',-1)
                wires.append(np.c_[origin+clipped[:,0]*scale,axis-clipped[:,1]*scale*sign])
        allpoints=np.vstack([source_uv]+wires) if wires else source_uv
        lo=allpoints.min(axis=0)-3;hi=allpoints.max(axis=0)+3
        factor=min(6,1300/max(hi-lo))
        width,height=np.ceil((hi-lo)*factor+2).astype(int)
        src=np.zeros((height,width),np.uint8);cad=src.copy()
        def polygon(p):return np.round((p-lo)*factor).astype(np.int32)
        cv2.fillPoly(src,[polygon(source_uv)],255)
        for void in evidence.get('source_section_void_polygons_pdf',[]):
            cv2.fillPoly(src,[polygon(np.array(void,float))],0)
        for wire in wires:cv2.fillPoly(cad,[polygon(wire)],255)
        intersection=int(np.count_nonzero((src>0)&(cad>0)));union=int(np.count_nonzero((src>0)|(cad>0)))
        iou=intersection/union if union else 0
        kernel=np.ones((3,3),np.uint8)
        source_edge=src-cv2.erode(src,kernel);model_edge=cad-cv2.erode(cad,kernel)
        to_model=cv2.distanceTransform(255-model_edge,cv2.DIST_L2,3)[source_edge>0]
        to_source=cv2.distanceTransform(255-source_edge,cv2.DIST_L2,3)[model_edge>0]
        distances=np.r_[to_model,to_source]/(factor*scale)
        overlay=np.full((height,width,3),255,np.uint8)
        overlay[src>0]=(100,160,255);overlay[cad>0]=(120,220,120);overlay[(src>0)&(cad>0)]=(90,170,90)
        cv2.imencode('.png',overlay)[1].tofile(str(model_dir/'section-comparison.png'))
        report['source_comparison']={'method':'registered_actual_brep_longitudinal_section_vs_hatched_source_face',
            'intersection_over_union':iou,'status':'consistent_partial_section' if iou>=.85 else 'mismatch',
            'mean_boundary_distance_mm':float(np.mean(distances)) if distances.size else None,
            'p95_boundary_distance_mm':float(np.percentile(distances,95)) if distances.size else None,
            'file':'section-comparison.png','source_coordinate_unit':'pdf_point','text_dimensions_hatching_excluded':True,
            'scope':'selected_section_only_not_complete_part_accuracy','explicit_dimensions_optimized':False}
    elif evidence.get('source_profile_polygon_pdf'):
        source=np.array(evidence['source_profile_polygon_pdf'],float)
        scale=np.array(evidence['profile_axis_scales_pt_per_mm'],float)
        origin=np.array(evidence['profile_origin_pdf'],float)
        projected=vertices[:,:2]*scale+origin
        lo=np.minimum(source.min(axis=0),projected.min(axis=0))-3
        hi=np.maximum(source.max(axis=0),projected.max(axis=0))+3
        factor=min(6,1300/max(hi-lo));width,height=np.ceil((hi-lo)*factor+2).astype(int)
        src=np.zeros((height,width),np.uint8);cad=src.copy()
        cv2.fillPoly(src,[np.round((source-lo)*factor).astype(np.int32)],255)
        points=np.round((projected-lo)*factor).astype(np.int32)
        for tri in faces:cv2.fillConvexPoly(cad,points[tri],255)
        intersection=int(np.count_nonzero((src>0)&(cad>0)));union=int(np.count_nonzero((src>0)|(cad>0)))
        iou=intersection/union if union else 0
        overlay=np.full((height,width,3),255,np.uint8)
        overlay[src>0]=(100,160,255);overlay[cad>0]=(120,220,120);overlay[(src>0)&(cad>0)]=(90,170,90)
        cv2.imencode('.png',overlay)[1].tofile(str(model_dir/'section-comparison.png'))
        report['source_comparison']={'method':'actual_brep_mesh_xy_projection_vs_dimensioned_source_profile',
            'intersection_over_union':iou,'status':'consistent_partial_section' if iou>=.85 else 'mismatch',
            'file':'section-comparison.png','scope':'selected_extrusion_profile_only_not_complete_part_accuracy',
            'explicit_dimensions_optimized':False}
    else:report['source_comparison']={'status':'not_registered' if intent.get('provenance')=='automatic_pdf' else 'not_applicable_manual_intent'}
    report['overall_status']='failed' if not report['dimension_checks_passed'] or report['source_comparison'].get('status')=='mismatch' else intent.get('reconstruction_status','manual')
    if evidence.get('independent_semantic_graph') and report['overall_status']!='failed' and report.get('quality_level')=='C':
        report['overall_status']='preliminary_geometry'
    if evidence.get('independent_semantic_graph') and report.get('nonblocking_dimension_conflicts') and report['overall_status']!='failed':
        report['quality_level']='C';report['overall_status']='preliminary_geometry'
    atomic_json(model_dir/'validation.json',report)
    return report
