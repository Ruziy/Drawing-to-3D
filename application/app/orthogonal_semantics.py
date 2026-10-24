"""Conservative multi-axis preview from measured channel walls in an L-shaped view.

Activated by a vision-localized view, never a part name or corpus filename.
Outer envelopes are approximate and remain level C. Both channel axes and
diameters require actual parallel strokes and existing diameter annotations.
"""
from pathlib import Path
import cv2
import numpy as np
from .calibration import robust_view_scale
from .evidence_registry import pdf_box
from .schemas import CADIntent


def orthogonal_candidate(page,document,output,semantics,diagnostic=None):
    diagnostic={} if diagnostic is None else diagnostic
    diagnostic['stage']='view_localization'
    main=next((v for v in semantics.views if v.role=='main'),None)
    if main is None:return None
    box=pdf_box(main.bbox,page);size=box[2:]-box[:2]
    calibration=robust_view_scale(page,box);scale=calibration['scale_pt_per_mm']
    diagnostic.update(source_bbox=box.tolist(),scale=scale,scale_inliers=calibration['inlier_dimension_count'])
    if not scale or calibration['inlier_dimension_count']<2:return None
    diameters=[d for d in page['dimensions'] if d['type']=='diameter' and 0<d['nominal_value']<min(size)/scale*.8]
    if not diameters:return None
    image=cv2.imdecode(np.fromfile(Path(output)/f"page-{page['page']}"/'original.png',np.uint8),0)
    pixels=np.array([image.shape[1]/page['width'],image.shape[0]/page['height']])
    p0=np.floor(box[:2]*pixels).astype(int);p1=np.ceil(box[2:]*pixels).astype(int)
    ink=(image[p0[1]:p1[1],p0[0]:p1[0]]<175).astype(np.uint8)*255
    lines=cv2.HoughLinesP(ink,1,np.pi/720,threshold=45,minLineLength=max(35,int(min(ink.shape)*.19)),maxLineGap=8)
    if lines is None:return None
    segments={0:[],1:[]}
    for line in lines[:,0]:
        p=(line.reshape(2,2)+p0)/pixels;delta=abs(p[1]-p[0]);axis=int(delta[1]>delta[0])
        if delta[1-axis]>max(.8,delta[axis]*.015):continue
        lo,hi=sorted(p[:,axis]);cross=float(p[:,1-axis].mean())
        if any(abs(cross-c)<.7 and abs(lo-a)<2 and abs(hi-b)<2 for c,a,b in segments[axis]):continue
        segments[axis].append((cross,lo,hi))
    pairs={0:[],1:[]}
    diagnostic['stage']='channel_wall_pairs'
    for axis in (0,1):
        for left in segments[axis]:
            for right in segments[axis]:
                if right[0]<=left[0]:continue
                diameter=(right[0]-left[0])/scale
                matches=[d for d in diameters if abs(d['nominal_value']-diameter)<max(.15,diameter*.035)]
                if not matches:continue
                lo=max(left[1],right[1]);hi=min(left[2],right[2])
                if hi-lo<min(size)*.24:continue
                center=(left[0]+right[0])/2
                # Channel interior must be predominantly unhatched, with two
                # original walls. This rejects many pairs of hatch/extension lines.
                if axis==0:b=[lo,center-(right[0]-left[0])*.45,hi,center+(right[0]-left[0])*.45]
                else:b=[center-(right[0]-left[0])*.45,lo,center+(right[0]-left[0])*.45,hi]
                q0=np.floor(np.array(b[:2])*pixels).astype(int);q1=np.ceil(np.array(b[2:])*pixels).astype(int)
                interior=image[q0[1]:q1[1],q0[0]:q1[0]]
                if not interior.size:continue
                occupancy=(interior<175).mean(axis=0 if axis==0 else 1)
                if np.median(occupancy)>.035:continue
                d=min(matches,key=lambda d:abs(d['nominal_value']-diameter))
                pairs[axis].append({'coordinate':center,'lo':lo,'hi':hi,'dimension':d,'measured_diameter_mm':diameter})
    diagnostic['pairs']={str(k):[{'coordinate':p['coordinate'],'lo':p['lo'],'hi':p['hi'],'dimension_id':p['dimension']['id'],
        'diameter':p['dimension']['nominal_value']} for p in records] for k,records in pairs.items()}
    combinations=[]
    for horizontal in pairs[0]:
        for vertical in pairs[1]:
            x=vertical['coordinate'];y=horizontal['coordinate']
            margin_x=vertical['dimension']['nominal_value']*scale/2+2
            margin_y=horizontal['dimension']['nominal_value']*scale/2+2
            if not horizontal['lo']-margin_x<=x<=horizontal['hi']+margin_x or not vertical['lo']-margin_y<=y<=vertical['hi']+margin_y:continue
            xr=(x-box[0])/size[0];yr=(y-box[1])/size[1]
            if .35<xr<.65 or .35<yr<.65:continue
            combinations.append((horizontal['hi']-horizontal['lo']+vertical['hi']-vertical['lo'],horizontal,vertical))
    diagnostic['stage']='channel_junction';diagnostic['junction_candidates']=len(combinations)
    if not combinations:return None
    _,h,v=max(combinations,key=lambda c:c[0]);x=v['coordinate'];y=h['coordinate']
    # The actual outer view frame and measured axis locations bound the preview.
    horizontal_radius=min(y-box[1],box[3]-y)/scale
    vertical_radius=min(x-box[0],box[2]-x)/scale
    if horizontal_radius<=h['dimension']['nominal_value']/2 or vertical_radius<=v['dimension']['nominal_value']/2:return None
    diagnostic['stage']='compiled_envelope'
    width=size[0]/scale;height=size[1]/scale;cx=(x-box[0])/scale
    bottom=-(box[3]-y)/scale;top=(y-box[1])/scale
    # Round-body depth is an envelope estimate, not an exact depth dimension.
    depth=2*min(horizontal_radius,vertical_radius)
    h_origin=[0,-depth/2,-horizontal_radius];h_height=2*horizontal_radius
    v_origin=[cx-vertical_radius,-depth/2,bottom]
    ops=[{'id':'horizontal_envelope','type':'box','length':width,'width':depth,'height':h_height,'origin':h_origin},
         {'id':'vertical_envelope','type':'box','length':2*vertical_radius,'width':depth,'height':height,'origin':v_origin},
         {'id':'joined_envelope','type':'fuse','base':'horizontal_envelope','tool':'vertical_envelope'}]
    left=x>np.mean(box[[0,2]]);down=y<np.mean(box[[1,3]])
    start_h=[0 if left else width,0,0];dir_h=[1 if left else -1,0,0];len_h=cx if left else width-cx
    start_v=[cx,0,bottom if down else top];dir_v=[0,0,1 if down else -1];len_v=-bottom if down else top
    dh=h['dimension']['nominal_value'];dv=v['dimension']['nominal_value']
    ops+=[{'id':'horizontal_channel','type':'hole','base':'joined_envelope','radius':dh/2,'height':len_h+.02,'origin':start_h,'direction':dir_h},
          {'id':'vertical_channel','type':'hole','base':'horizontal_channel','radius':dv/2,'height':len_v+.02,'origin':start_v,'direction':dir_v}]
    expected=[{'type':'semantic_void','origin':start_h,'direction':dir_h,'radius_mm':dh/2,'depth_mm':len_h,
               'surrounding_radius_mm':min(horizontal_radius*.8,dh/2+.7),'object_id':'L_CHANNEL_H'},
              {'type':'semantic_void','origin':start_v,'direction':dir_v,'radius_mm':dv/2,'depth_mm':len_v,
               'surrounding_radius_mm':min(vertical_radius*.8,dv/2+.7),'object_id':'L_CHANNEL_V'}]
    params=[{'object_id':'orthogonal_envelope','role':'depth','value_mm':depth,'status':'estimated_dimensions',
             'reason':'outer_depth_envelope_from_visible_arm_widths_not_a_verified_out_of_plane_dimension','dimension_id':None},
            {'object_id':'horizontal_channel','role':'diameter','value_mm':dh,'status':'verified_dimensions','dimension_id':h['dimension']['id'],'reason':'observed_parallel_unhatched_channel_walls'},
            {'object_id':'vertical_channel','role':'diameter','value_mm':dv,'status':'verified_dimensions','dimension_id':v['dimension']['id'],'reason':'observed_parallel_unhatched_channel_walls'}]
    evidence={'page':page['page'],'page_size_pdf':{'width':page['width'],'height':page['height']},'drawing_id':document['drawing_id'],
        'source_sha256':document['sha256'],'source_output_directory':str(Path(output).resolve()),'llm_assisted':True,'independent_semantic_graph':True,
        'profile_id':'LLM_ORTHOGONAL_ENVELOPE','primary_bbox_pdf':box.tolist(),
        'registered_objects':[{'id':'LLM_ORTHOGONAL_ENVELOPE','local_id':'orthogonal_envelope','view_id':main.id,'bbox_pdf':box.tolist(),
                               'origin':'vision_localization_plus_measured_orthogonal_channel_pairs','proof':{'method':'two_measured_intersecting_cavity_axes'}}],
        'semantic_views':[view.model_dump(mode='json') for view in semantics.views],
        'registration':{'plane':'xz','origin_pdf':[box[0],y],'scales_pt_per_mm':[scale,scale]},
        'semantic_parameters':params,'dimension_groups':{s:[p for p in params if p['status']==s] for s in ('verified_dimensions','inferred_dimensions','estimated_dimensions','unknown_dimensions')},
        'expected_features':semantics.expected_features,'implemented_features':[{'id':'h_channel','kind':'hole','status':'implemented'},
            {'id':'v_channel','kind':'hole','status':'implemented'},{'id':'body_union','kind':'fuse','status':'implemented'}],
        'quality_level':'C','full_reconstruction_verified':False,'local_calibration':calibration,
        'channel_witnesses':{'horizontal':h,'vertical':v},'source_dimension_ids':[i['dimension_id'] for i in calibration['inliers']]}
    unresolved=[{'type':'estimated_outer_envelope','reason':'Угловая 3D-оболочка из ограничивающих призм. Наружные круглые ступени, резьбы и поперечная толщина не подтверждены; уровень C.'}]
    intent=CADIntent.model_validate({'provenance':'automatic_pdf','reconstruction_status':'preliminary_geometry','evidence':evidence,
        'operations':ops,'result':'vertical_channel','validation_expectations':expected,'unresolved_features':unresolved,
        'description':'Предварительная угловая оболочка с двумя измеренными пересекающимися каналами.'}).model_dump(mode='json')
    return {'intent':intent,'origin':'llm_region_orthogonal_channels','interpretation':{'page':page['page'],
        'status':'preliminary_geometry','strategy':'measured_orthogonal_channels_with_estimated_outer_envelope','dimension_bindings':[],
        'unresolved_features':unresolved}}
