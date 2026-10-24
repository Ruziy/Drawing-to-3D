"""Registered orthographic comparisons and explicit reconstruction uncertainty."""
from pathlib import Path
import cv2
import numpy as np
from .evidence_registry import pdf_box


def metric_registration_check(evidence):
    scales=evidence['registration']['scales_pt_per_mm'];ratio=max(scales)/min(scales)
    bounded_break=any(p.get('role')=='length' and p.get('status') in ('verified_dimensions','inferred_dimensions') and
        p.get('reason')=='dimension_bounded_shortened_source_view' for p in evidence.get('semantic_parameters',[]))
    piecewise=bool(evidence['registration'].get('shortening'))
    return {'passed':bool(ratio<=1.15 or bounded_break or piecewise),'axis_scale_ratio':ratio,
        'dimension_bounded_graphical_break':bounded_break,'piecewise_break_registration':piecewise,
        'reason':'A source view cannot be stretched independently on two axes to conceal unrelated dimension roles'}


def target_silhouette(image,box,page,maximum=800):
    pixels=np.array([image.shape[1]/page['width'],image.shape[0]/page['height']])
    low=np.floor(np.array(box[:2])*pixels).astype(int);high=np.ceil(np.array(box[2:])*pixels).astype(int)
    low=np.maximum(low,0);high=np.minimum(high,[image.shape[1]-1,image.shape[0]-1])
    crop=image[low[1]:high[1]+1,low[0]:high[0]+1]
    if not crop.size:raise ValueError('Empty comparison view')
    ink=(crop<180).astype(np.uint8)*255
    contours,_=cv2.findContours(cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    large=[c for c in contours if cv2.contourArea(c)>max(50,ink.size*.025)]
    if not large:raise ValueError('No substantial source outline for registered comparison')
    points=np.vstack(large)
    # This is an OUTER silhouette check. Cavities are measured separately in BRep.
    # A hull can overfill concavities, which is recorded and cannot prove level A.
    silhouette=np.zeros_like(ink);cv2.fillConvexPoly(silhouette,cv2.convexHull(points),255)
    factor=min(1,maximum/max(silhouette.shape))
    if factor<1:silhouette=cv2.resize(silhouette,None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST)
    return silhouette,low,pixels,factor


def compare(model_dir,mesh,intent):
    evidence=intent['evidence'];page_number=evidence['page'];source=Path(evidence['source_output_directory'])
    image=cv2.imdecode(np.fromfile(source/f'page-{page_number}'/'original.png',np.uint8),0)
    registered=evidence['registration'];objects=evidence['registered_objects'];primary=next(o for o in objects if o['id']==evidence['profile_id'])
    views=evidence['semantic_views'];selected=next(v for v in views if v['id']==primary['view_id'])
    # Page dimensions are retained explicitly so saved images are sufficient evidence.
    page=evidence['page_size_pdf'];vertices=np.array(mesh['vertices'],float);triangles=np.array(mesh['triangles'],int)
    original=np.array(registered['origin_pdf']);scale=np.array(registered['scales_pt_per_mm'])
    plane=registered['plane'];uv=vertices[:,[0,1]] if plane=='xy' else vertices[:,[0,2]]*np.array([1,-1])
    if registered.get('shortening'):
        short=registered['shortening'];axis=short['axis'];cut=short['cut_mm'];extra=short['extra_mm']
        uv[:,axis]=np.where(uv[:,axis]>cut+extra,uv[:,axis]-extra,np.minimum(uv[:,axis],cut))
    if registered.get('revolve_axis')=='horizontal':uv=vertices[:,[2,0]]*np.array([1,-1])
    elif registered.get('revolve_axis')=='vertical':uv=vertices[:,[0,2]]
    projected=uv*scale+original
    comparisons=[]
    def registered_mask(view,points,label,box_override=None):
        box=box_override if box_override is not None else pdf_box(view['bbox'],page)
        target,low,pixels,factor=target_silhouette(image,box,page)
        transformed=(points*pixels-low)*factor
        # Never clip model material outside the selected source frame. Extend
        # the comparison canvas with empty source pixels so excess is penalized.
        shift=np.minimum(np.floor(transformed.min(axis=0)-2).astype(int),[0,0])
        maximum=np.maximum(np.ceil(transformed.max(axis=0)+3).astype(int),[target.shape[1],target.shape[0]])
        dimensions=maximum-shift
        if max(dimensions)>6000:raise ValueError('Projection extends implausibly far beyond source view')
        expanded=np.zeros((dimensions[1],dimensions[0]),np.uint8)
        expanded[-shift[1]:-shift[1]+target.shape[0],-shift[0]:-shift[0]+target.shape[1]]=target
        target=expanded;transformed-=shift
        model=np.zeros_like(target)
        for tri in triangles:cv2.fillConvexPoly(model,np.round(transformed[tri]).astype(np.int32),255)
        intersection=int(np.count_nonzero((target>0)&(model>0)));union=int(np.count_nonzero((target>0)|(model>0)))
        iou=intersection/union if union else 0
        extra=int(np.count_nonzero((target==0)&(model>0)))/max(int(np.count_nonzero(model)),1)
        missing=int(np.count_nonzero((target>0)&(model==0)))/max(int(np.count_nonzero(target)),1)
        overlay=np.full((*target.shape,3),255,np.uint8);overlay[target>0]=(100,160,255);overlay[model>0]=(120,220,120);overlay[(target>0)&(model>0)]=(90,170,90)
        filename=f'semantic-view-{len(comparisons)+1}.png'
        cv2.imencode('.png',overlay)[1].tofile(str(Path(model_dir)/filename))
        return {'view_id':view['id'],'role':view['role'],'status':'compared','intersection_over_union':iou,
                'extra_projected_material_fraction':extra,'missing_projected_material_fraction':missing,
                'file':filename,'registration':label,'source_bbox_pdf':list(box),
                'scope':'outer_convex_silhouette_only_cavities_and_concavities_not_validated_by_this_metric'}
    main=registered_mask(selected,projected,'primary_semantic_frame_affine_mm_to_pdf')
    comparisons.append(main)
    # Independent end views can expose a badly oriented or missing body. Their
    # own frames/scales are used; no sheet-global scale is propagated.
    for view in views:
        if view['id']==selected['id']:continue
        if view['role']!='end':
            comparisons.append({'view_id':view['id'],'role':view['role'],'status':'not_registered','reason':'No proven 3D-to-view correspondence'});continue
        try:
            box=pdf_box(view['bbox'],page);plane_axes=[1,2] if selected['axis']=='horizontal' else [0,1]
            points=vertices[:,plane_axes];extent=points.max(axis=0)-points.min(axis=0)
            if min(extent)<1e-6:raise ValueError('Degenerate orthographic extent')
            sx,sy=(box[2:]-box[:2])/extent
            # Fitting anisotropic end frames would conceal a non-circular model.
            if abs(sx/sy-1)>.08:raise ValueError('End-view aspect ratio does not agree with model')
            points=(points-points.min(axis=0))*min(sx,sy)+box[:2]
            comparisons.append(registered_mask(view,points,'independent_end_frame_uniform_scale'))
        except ValueError as exc:comparisons.append({'view_id':view['id'],'role':view['role'],'status':'not_registered','reason':str(exc)})
    # Preview level is a distinct output. Poor physical agreement still rejects
    # it; an approximate label is not a license for arbitrary solids.
    invariant_ok=mesh['valid'] and mesh['volume_mm3']>0
    registered_values=[c['intersection_over_union'] for c in comparisons if c['status']=='compared']
    primary_iou=main['intersection_over_union'];level=evidence.get('quality_level','C')
    metric=metric_registration_check(evidence)
    if primary_iou<.45 or not invariant_ok or not metric['passed']:status='mismatch'
    elif primary_iou<.82 or (len(registered_values)>1 and min(registered_values)<.7):status='consistent_preliminary_views';level='C'
    else:status='consistent_partial_views' if level=='B' else 'consistent_preliminary_views'
    # Always expose how many declared views could not be independently registered.
    return {'source_comparison':{'method':'registered_semantic_orthographic_outer_silhouettes',
                'status':status,'intersection_over_union':primary_iou,'file':main['file'],
                'scope':'registered_outer_views_not_complete_part_accuracy','full_reconstruction_verified':False},
            'view_comparisons':comparisons,'quality_level':level,'metric_registration_check':metric,
            'completeness':{'declared_views':len(views),'registered_views':len(registered_values),
                'unregistered_views':len(views)-len(registered_values),'expected_features':evidence['expected_features'],
                'implemented_features':evidence['implemented_features'],'full_part_verified':False}}
