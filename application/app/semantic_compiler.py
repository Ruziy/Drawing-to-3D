"""Independent semantic graph -> bounded parametric CAD operations.

No filename branches, reference fixtures, executable model text or free-form CAD
JSON. New sheet objects are registered against pixels before compilation.
"""
import math
from pathlib import Path
import numpy as np
from .evidence_registry import EvidenceRegistry,pdf_box
from .local_dimension_ocr import reread_dimension
from .schemas import CADIntent,Operation
from .storage import atomic_json


class SemanticCompiler:
    def __init__(self,page,document,output,target,semantics,graph):
        self.page=page;self.document=document;self.output=output;self.target=Path(target)
        self.views={v.id:v for v in semantics.views};self.graph={o.id:o for o in graph.objects}
        self.registry=EvidenceRegistry(page,output);self.parameters=[];self.conflicts=[];self.rejections=[]
        self.dimension_map={d['id']:(page,d) for d in page['dimensions']}
        for other in document['pages']:
            for d in other['dimensions']:self.dimension_map[f"P{other['page']}:{d['id']}"]=(other,d)
        self.operations=[];self.results={};self.expectations=[];self.features=[];self.registrations=[]
        self.transform=None;self.unknown=[];self.ocr_cache={}
        self.family_resolutions=[];self.force_preliminary=False

    def geometric_family(self,obj):
        if obj.kind not in ('cylinder','tube'):return obj
        cited=[self.dimension_map[p.dimension_id][1] for p in obj.parameters if p.dimension_id in self.dimension_map]
        if any(d['type'] in ('diameter','thread') for d in cited):return obj
        for view in self.views.values():
            box=pdf_box(view.bbox,self.page);size=box[2:]-box[:2]
            if abs(size[0]/size[1]-1)>.15:continue
            for circle in self.page.get('geometry',{}).get('circles',[]):
                center=np.array(circle.get('properties',{}).get('center',[1e8,1e8]))
                radius=circle.get('properties',{}).get('radius_pt',0)
                if np.linalg.norm(center-(box[:2]+box[2:])/2)<min(size)*.12 and radius>min(size)*.35:return obj
        # A rectangular view with only linear dimensions cannot justify a tube.
        # Retain the LLM's source localization but lower it to a visible sheet
        # profile. Its out-of-plane interpretation remains explicitly level C.
        source={p.role:p for p in obj.parameters}
        if 'diameter' not in source or 'length' not in source:raise ValueError('No rotational evidence or usable prismatic dimensions')
        parameters=[source['diameter'].model_copy(update={'role':'width'}),source['length']]
        if 'inner_diameter' in source:parameters.append(source['inner_diameter'].model_copy(update={'role':'thickness'}))
        else:
            from .semantic_schema import SemanticParameter
            width=source['diameter'].value or (self.dimension_map[source['diameter'].dimension_id][1]['nominal_value'] if source['diameter'].dimension_id in self.dimension_map else 1)
            parameters.append(SemanticParameter(role='thickness',value=width*.1,evidence='visual_estimate'))
        self.family_resolutions.append({'object_id':obj.id,'proposed_kind':obj.kind,'compiled_kind':'extrusion',
            'reason':'No diameter datum or circular end-view evidence; use observed profile, with uncertain out-of-plane thickness'})
        self.force_preliminary=True
        return obj.model_copy(update={'kind':'extrusion','axis':'normal','parameters':parameters})

    def parameter(self,obj,record,role,default=None):
        proposal=next((p for p in obj.parameters if p.role==role),None)
        rebound=None
        if proposal and role in ('length','width') and obj.kind in ('tube','cylinder','box','extrusion') and not (obj.kind in ('tube','cylinder') and obj.axis=='normal'):
            axis=(0 if obj.axis=='horizontal' else 1) if obj.kind in ('tube','cylinder') else (0 if role=='width' else 1)
            orientation='horizontal' if axis==0 else 'vertical';b=record['bbox_pdf'];choices=[];current_matches=False
            current=self.dimension_map.get(proposal.dimension_id,(None,{}))[1]
            for d in self.page['dimensions']:
                if d['type']!='linear':continue
                for a in d.get('associations',[]):
                    if a['orientation']!=orientation:continue
                    points=np.array(a['endpoints_pdf']);lo,hi=sorted(points[:,axis])
                    error=max(abs(lo-b[axis]),abs(hi-b[axis+2]))
                    if error<3:
                        choices.append((error+.03*a.get('score_distance_pt',0),d))
                        if d['id']==current.get('id'):current_matches=True
            if choices and not current_matches:
                d=min(choices,key=lambda v:v[0])[1]
                rebound=proposal.dimension_id
                proposal=proposal.model_copy(update={'dimension_id':d['id'],'value':None})
        if proposal is None:
            if default is None:raise ValueError(f'{obj.id}: missing {role}')
            if callable(default):default=default()
            value,status,source=default
            entry={'object_id':obj.id,'role':role,'value_mm':value,'status':status,'reason':source,'dimension_id':None}
        elif proposal.dimension_id:
            if proposal.dimension_id not in self.dimension_map:raise ValueError('Unknown existing dimension ID '+proposal.dimension_id)
            source_page,d=self.dimension_map[proposal.dimension_id]
            if d['type'] in ('roughness','angle','tolerance') and role!='angle':raise ValueError('Annotation is not a physical length')
            if d['nominal_value']<=0:raise ValueError('Nonpositive physical parameter')
            value=d['nominal_value'];status='inferred_dimensions'
            source='existing_ocr_nominal_semantic_role_requires_geometric_confirmation'
            inliers=record['calibration']['inliers']
            if any(i['dimension_id']==d['id'] for i in inliers) and source_page['page']==self.page['page']:
                status='verified_dimensions';source='local_robust_scale_inlier_and_declared_feature_role'
            if proposal.value is not None and abs(proposal.value-value)>1e-6:
                key=(source_page['page'],d['id'])
                if key not in self.ocr_cache:
                    self.ocr_cache[key]=reread_dimension(source_page,self.output,d['bbox'],self.target/'ocr-recheck'/f"p{key[0]}-{key[1]}")
                reread=self.ocr_cache[key]
                accepted=any(abs(v-proposal.value)<1e-6 for v in reread['consensus_values'])
                self.conflicts.append({'dimension_id':proposal.dimension_id,'original_value':value,'proposed_value':proposal.value,
                    'action':'independent_crop_recheck_accepted_original_record_preserved' if accepted else 'quarantined_original_preserved',
                    'recheck':reread})
                if accepted:value=proposal.value;status='inferred_dimensions';source='independent_local_ocr_recheck'
                else:status='estimated_dimensions';source='unresolved_ocr_and_vision_conflict_original_nominal_used'
            entry={'object_id':obj.id,'role':role,'value_mm':value,'status':status,'reason':source,
                   'dimension_id':proposal.dimension_id,'original_raw_text':d['raw_text'],'bbox_pdf':d['bbox'],
                   'source_page':source_page['page']}
        else:
            if proposal.value is None:
                if default is None:raise ValueError('Parameter has neither source dimension nor estimate')
                if callable(default):default=default()
                value,status,source=default
            else:value=proposal.value;status='estimated_dimensions';source=proposal.evidence
            entry={'object_id':obj.id,'role':role,'value_mm':value,'status':status,'reason':source,'dimension_id':None,
                   'source_bbox_normalized':proposal.source_bbox,'proposed_raw_text':proposal.raw_text}
            if proposal.source_bbox and len(proposal.source_bbox)==4 and proposal.evidence=='image_dimension':
                if not all(0<=v<=1 for v in proposal.source_bbox):raise ValueError('Reading region outside page')
                box=pdf_box(proposal.source_bbox,self.page)
                reread=reread_dimension(self.page,self.output,box,self.target/'ocr-recheck'/f'{obj.id}-{role}')
                entry['independent_recheck']=reread
                if any(abs(v-value)<1e-6 for v in reread['consensus_values']):
                    entry.update(status='inferred_dimensions',reason='independently_reread_new_dimension')
        if not math.isfinite(value) or not 0<value<=10000:raise ValueError('Parameter outside geometric limits')
        if rebound:entry.update(requested_dimension_id=rebound,role_binding_method='full_observed_axial_dimension_boundaries')
        entry['unit']='count' if role=='count' else 'deg' if role=='angle' else 'mm'
        if entry.get('dimension_id'):
            d=self.dimension_map[entry['dimension_id']][1];b=record['bbox_pdf'];consensus=record['calibration']['scale_pt_per_mm']
            entry.update(status='estimated_dimensions',reason='OCR_nominal_known_but_feature_role_not_independently_confirmed')
            axis=None
            if role=='width':axis=0
            elif role=='length' and not (obj.kind in ('tube','cylinder') and obj.axis=='normal'):
                axis=0 if obj.axis=='horizontal' and obj.kind in ('tube','cylinder') else 1
            elif role=='diameter' and obj.kind in ('tube','cylinder'):axis=1 if obj.axis=='horizontal' else 0
            elif role=='depth' and obj.axis!='normal':axis=0 if obj.axis=='horizontal' else 1
            if axis is not None:
                orientation='horizontal' if axis==0 else 'vertical'
                full=any(a['orientation']==orientation and max(abs(min(np.array(a['endpoints_pdf'])[:,axis])-b[axis]),
                    abs(max(np.array(a['endpoints_pdf'])[:,axis])-b[axis+2]))<3 for a in d.get('associations',[]))
                numerical=consensus and abs((b[axis+2]-b[axis])/consensus/value-1)<.04
                if numerical or (full and self.views[obj.view_id].shortened and role=='length'):
                    entry.update(status='verified_dimensions' if full and numerical else 'inferred_dimensions',
                                 reason='original_boundary_extent_and_local_scale' if numerical else 'dimension_bounded_shortened_source_view')
            if role=='radius' and d['type']=='radius':entry.update(status='inferred_dimensions',reason='radius_annotation_requires_native_surface_verification')
        if (role=='thickness' or (role=='length' and obj.kind in ('tube','cylinder') and obj.axis=='normal')) and entry.get('source_page',self.page['page'])==self.page['page']:
            d=self.dimension_map.get(entry.get('dimension_id'),(None,{}))[1]
            import re
            stock_label=bool(re.search(r'(?:S|δ|толщ)\s*\d',d.get('raw_text',''),re.I))
            if not stock_label:
                entry.update(status='estimated_dimensions',reason='unverified_out_of_plane_role_the_OCR_number_is_not_a_proven_thickness')
                main_box=record['bbox_pdf']
                for face in self.page.get('layout',{}).get('profile_candidates',[]):
                    side=face['bbox'];axis=None
                    if max(abs(side[0]-main_box[0]),abs(side[2]-main_box[2]))<3 and (side[3]<main_box[1]-8 or side[1]>main_box[3]+8):axis=1
                    if max(abs(side[1]-main_box[1]),abs(side[3]-main_box[3]))<3 and (side[2]<main_box[0]-8 or side[0]>main_box[2]+8):axis=0
                    if axis is None:continue
                    orientation='horizontal' if axis==0 else 'vertical'
                    for association in d.get('associations',[]):
                        ends=np.array(association['endpoints_pdf'])[:,axis]
                        if association['orientation']==orientation and max(abs(min(ends)-side[axis]),abs(max(ends)-side[axis+2]))<3:
                            entry.update(status='verified_dimensions',reason='dimensioned_orthographic_side_frame_sharing_actual_profile_extrema',
                                         supporting_view_profile_id=face['id']);break
            if role=='thickness' and entry['status']=='estimated_dimensions':
                in_plane=[p['value_mm'] for p in self.parameters if p['object_id']==obj.id and p['role'] in ('width','length')]
                if in_plane and value>min(in_plane)*2:
                    # An unrelated OCR number cannot create a kilometres-thick
                    # plate. Retain the proposed binding as rejected evidence;
                    # the explicitly assumed preview thickness is independent.
                    entry.update(rejected_out_of_plane_nominal_mm=value,requested_dimension_id=entry.get('dimension_id'),
                        dimension_id=None,status='estimated_dimensions',
                        reason='unconfirmed_thickness_outside_visible_proportions_preview_assumption_10_percent_minimum_span')
                    value=min(in_plane)*.10;entry['value_mm']=value
        self.parameters.append(entry);return float(value)

    def inferred_span(self,record,axis):
        scale=record['calibration']['scale_pt_per_mm']
        if scale is None:raise ValueError('No physical scale anchor for inferred extent')
        b=record['bbox_pdf'];return ((b[axis+2]-b[axis])/scale,'inferred_dimensions','original_boundary_extent_at_local_consensus_scale')

    def world(self,point):
        t=self.transform;p=np.array(point,float)
        q=(p-t['origin_pdf'])/t['scales']
        return [float(q[0]),float(q[1]),0.] if t['plane']=='xy' else [float(q[0]),0.,float(-q[1])]

    def hole_frame(self,obj,record,diameter,depth,require_opening=True):
        b=np.array(record['bbox_pdf']);center=np.array(obj.center)*[self.page['width'],self.page['height']] if obj.center else (b[:2]+b[2:])/2
        primary=self.registry.records[next(iter(self.results))]
        if obj.view_id==primary['view_id']:
            if obj.axis=='horizontal' and obj.center is None:
                high=abs(b[2]-primary['bbox_pdf'][2])<abs(b[0]-primary['bbox_pdf'][0])
                center=np.array([b[2] if high else b[0],(b[1]+b[3])/2])
            elif obj.axis=='vertical' and obj.center is None:
                high=abs(b[1]-primary['bbox_pdf'][1])<abs(b[3]-primary['bbox_pdf'][3])
                center=np.array([(b[0]+b[2])/2,b[1] if high else b[3]])
            origin=self.world(center)
            direction={'horizontal':[1,0,0],'vertical':[0,0,1],'normal':[0,0,1] if self.transform['plane']=='xy' else [0,1,0]}[obj.axis]
            if obj.axis=='normal' and self.transform['plane']=='xz':origin[1]=-depth/2
            if obj.axis=='normal' and self.transform['plane']=='xy':
                thickness=next((p['value_mm'] for p in self.parameters if p['role']=='thickness' and p['object_id']==primary['local_id']),None)
                if thickness is None:thickness=next((p['value_mm'] for p in self.parameters if p['role']=='length' and p['object_id']==primary['local_id']),depth)
                origin[2]=max(0,thickness-depth)
            if obj.axis in ('horizontal','vertical') and obj.center is None and high:direction=[-v for v in direction]
            scale=record['calibration']['scale_pt_per_mm'] or float(np.mean(self.transform['scales']))
        else:
            view=self.views[obj.view_id];viewbox=pdf_box(view.bbox,self.page);size=viewbox[2:]-viewbox[:2]
            if abs(size[0]/size[1]-1)>.2:raise ValueError('No independently registered circular end frame for this hole')
            outer=next((p['value_mm'] for p in self.parameters if p['role']=='diameter'),None)
            length=next((p['value_mm'] for p in self.parameters if p['role']=='length'),None)
            if not outer or not length:raise ValueError('End frame needs a dimensioned primary axis')
            scale=min(size)/outer;radial=(center-(viewbox[:2]+viewbox[2:])/2)/scale
            main=self.views[primary['view_id']];mainbox=pdf_box(main.bbox,self.page)
            if main.axis=='horizontal':
                high=viewbox[0]>mainbox[2];origin=[length if high else 0,float(radial[0]),float(-radial[1])];direction=[-1 if high else 1,0,0]
            else:
                high=viewbox[3]<mainbox[1];origin=[float(radial[0]),float(radial[1]),length if high else 0];direction=[0,0,-1 if high else 1]
        # A diameter label alone cannot turn an arbitrary point into a hole.
        circles=[]
        for circle in self.page.get('geometry',{}).get('circles',[]):
            prop=circle.get('properties',{});point=np.array(prop.get('center',[1e8,1e8]));radius=prop.get('radius_pt',0)
            if abs(2*radius/scale-diameter)>max(.3,diameter*.08):continue
            if np.linalg.norm(point-center)>max(3,radius*.4):continue
            circles.append(circle['id'])
        walls=self.registry.inner_wall_spans(record,obj.axis) if obj.axis!='normal' else []
        paired=any(abs(w/scale-diameter)<max(.3,diameter*.05) for w in walls)
        if obj.axis!='normal':
            radial=1 if obj.axis=='horizontal' else 0
            span=b[radial+2]-b[radial]
            paired|=min(record['proof'].get('side_support',[0]))>=.6 and abs(span/scale-diameter)<max(.3,diameter*.05)
        if require_opening and not circles and not paired:raise ValueError('Hole has no matching observed circular opening or pair of cavity walls')
        record['proof'].update(hole_witness_circle_ids=circles,paired_cavity_walls_confirmed=bool(paired))
        return origin,direction

    def initialize_transform(self,obj,record,values):
        if self.transform:return
        b=np.array(record['bbox_pdf']);size=b[2:]-b[:2]
        if obj.kind in ('tube','cylinder') and obj.axis=='normal':
            sx=size[0]/values['diameter'];sy=size[1]/values['diameter'];origin=(b[:2]+b[2:])/2;plane='xy'
        elif obj.kind in ('box','extrusion','pocket'):
            sx=size[0]/values['width'];sy=size[1]/values['length'];origin=b[:2];plane='xy'
        else:
            axis=0 if obj.axis=='horizontal' else 1;radial=1-axis
            scale=np.empty(2);scale[axis]=size[axis]/values['length'];scale[radial]=size[radial]/values['diameter']
            sx,sy=scale;origin=np.array([b[0],(b[1]+b[3])/2]) if axis==0 else np.array([(b[0]+b[2])/2,b[3]])
            plane='xz'
        consensus=record['calibration']['scale_pt_per_mm']
        if consensus and record['calibration']['inlier_dimension_count']>=2:
            if self.views[obj.view_id].shortened and obj.axis!='normal':
                if obj.axis=='horizontal':sy=consensus
                elif obj.axis=='vertical':sx=consensus
                else:sx=sy=consensus
            else:sx=sy=consensus
        if min(sx,sy)<=0:raise ValueError('Invalid local mapping')
        self.transform={'plane':plane,'origin_pdf':origin,'scales':np.array([sx,sy])}

    def operation(self,kind,**kwargs):
        name=f"sem_{len(self.operations)+1:03d}"
        operation=Operation.model_validate({'id':name,'type':kind,**kwargs}).model_dump(mode='json')
        self.operations.append(operation);return name

    def compile_object(self,obj,record,base=None):
        b=np.array(record['bbox_pdf']);mid=(b[:2]+b[2:])/2
        direction={'horizontal':[1,0,0],'vertical':[0,0,1],'normal':[0,0,1]}[obj.axis]
        values={};kind=obj.kind
        if kind in ('cylinder','tube'):
            length=self.parameter(obj,record,'length',lambda:self.inferred_span(record,0 if obj.axis=='horizontal' else 1))
            diameter=self.parameter(obj,record,'diameter',lambda:self.inferred_span(record,1 if obj.axis=='horizontal' else 0))
            consensus=record['calibration']['scale_pt_per_mm'];axis=0 if obj.axis=='horizontal' else 1
            if consensus and record['calibration']['inlier_dimension_count']>=2:
                observed_diameter=(b[3-axis]-b[1-axis])/consensus
                if abs(observed_diameter/diameter-1)>.04:
                    matches=[d for d in self.page['dimensions'] if d['type']=='diameter' and abs(d['nominal_value']/observed_diameter-1)<.025]
                    if matches and len({round(d['nominal_value'],6) for d in matches})==1:
                        d=matches[0];entry=self.parameters[-1]
                        entry.update(requested_dimension_id=entry.get('dimension_id'),dimension_id=d['id'],value_mm=d['nominal_value'],
                                     original_raw_text=d['raw_text'],bbox_pdf=d['bbox'],status='verified_dimensions',reason='outer_diameter_role_resolved_by_observed_surface_span_and_local_scale')
                        diameter=d['nominal_value']
                    else:
                        self.parameters[-1].update(status='estimated_dimensions',reason='diameter_nominal_disagrees_with_original_radial_extent')
                visible_length=(b[axis+2]-b[axis])/consensus
                if obj.axis!='normal' and self.views[obj.view_id].shortened and length<visible_length*.85:raise ValueError('Declared shortened length is smaller than visible physical span; wrong dimension role')
            values.update(length=length,diameter=diameter);self.initialize_transform(obj,record,values)
            origin=self.world(mid if obj.axis=='normal' else [b[0],mid[1]] if obj.axis=='horizontal' else [mid[0],b[3]])
            result=self.operation('cylinder',radius=diameter/2,height=length,origin=origin,direction=direction)
            self.expectations.append({'type':'semantic_cylinder','radius_mm':diameter/2,'origin':origin,'direction':direction,
                                      'object_id':record['id'],'role':'outer_surface','length_mm':length})
            if kind=='tube':
                inner=self.parameter(obj,record,'inner_diameter')
                if consensus:
                    spans=self.registry.inner_wall_spans(record,obj.axis)
                    matches=[d for d in self.page['dimensions'] if d['type']=='diameter' and d['nominal_value']<diameter and
                             any(abs(span/consensus/d['nominal_value']-1)<.025 for span in spans)]
                    if matches and len({round(d['nominal_value'],6) for d in matches})==1:
                        d=matches[0];entry=self.parameters[-1];entry.update(requested_dimension_id=entry.get('dimension_id'),
                            dimension_id=d['id'],value_mm=d['nominal_value'],status='verified_dimensions',
                            original_raw_text=d['raw_text'],bbox_pdf=d['bbox'],
                            reason='inner_diameter_role_resolved_by_two_observed_coaxial_walls')
                        inner=d['nominal_value']
                if inner>=diameter:raise ValueError('Inner diameter must be smaller than outer diameter')
                proof=self.registry.through_wall_evidence(record,obj.axis,inner*(consensus or float(self.transform['scales'][1-axis])))
                record['proof']['through_bore']=proof
                if not proof['confirmed']:
                    self.force_preliminary=True
                    self.rejections.append({'object_id':obj.id,'reason':'Through-bore depth is not established by the original walls; retained outer envelope only'})
                    self.features.append({'id':record['id'],'kind':'outer_envelope','source_object':obj.id,'status':'implemented'})
                    return result
                entry=(np.array(origin)-np.array(direction)*.01).tolist()
                result=self.operation('hole',base=result,radius=inner/2,height=length+.02,origin=entry,direction=direction)
                self.expectations.append({'type':'semantic_void','origin':origin,'direction':direction,'radius_mm':inner/2,
                    'depth_mm':length,'surrounding_radius_mm':(inner+diameter)/4,'object_id':record['id']})
                self.features.append({'id':record['id']+':bore','kind':'through_bore','source_object':obj.id,'status':'implemented'})
        elif kind in ('box','extrusion'):
            if self.transform and self.transform['plane']!='xy':raise ValueError('Extrusion needs an established matching profile plane')
            width=self.parameter(obj,record,'width',lambda:self.inferred_span(record,0))
            length=self.parameter(obj,record,'length',lambda:self.inferred_span(record,1))
            thickness=self.parameter(obj,record,'thickness',(min(width,length)*.10,'estimated_dimensions','unknown_thickness_visual_preview_assumption_10_percent_minimum_span'))
            values.update(width=width,length=length);self.initialize_transform(obj,record,values)
            if kind=='box':
                result=self.operation('box',length=width,width=length,height=thickness,origin=self.world(b[:2]))
            else:
                points=np.array(record['polygon_pdf']);profile=[self.world(p)[:2] for p in points]
                if self.views[obj.view_id].shortened:
                    consensus=record['calibration']['scale_pt_per_mm']
                    if consensus:
                        observed=(b[3]-b[1])/consensus;extra=length-observed
                        if extra>0:
                            cut=(b[3]-b[1])/consensus/2
                            profile=[[(p[0]-b[0])/consensus,(p[1]-b[1])/consensus+(extra if (p[1]-b[1])/consensus>cut else 0)] for p in points]
                            self.transform.update(scales=np.array([consensus,consensus]),
                                shortening={'axis':1,'cut_mm':cut,'extra_mm':extra})
                from .polygon_repair import clean_pixel_polygon
                profile,repair=clean_pixel_polygon(profile)
                profile=np.array(profile)
                record['proof']['tiny_pixel_loop_repair']=repair
                if repair['discarded_loop_area']>0:self.force_preliminary=True
                constraints=[]
                for index,nominal in ((0,width),(1,length)):
                    lo=float(profile[:,index].min());span=float(np.ptp(profile[:,index]))
                    if span and abs(span-nominal)<=max(.03,nominal*.015):
                        profile[:,index]=(profile[:,index]-lo)*nominal/span
                        constraints.append({'axis':index,'nominal_mm':nominal,'raw_extent_mm':span,'method':'explicit_extent_constraint_within_original_stroke_envelope'})
                record['proof']['dimensional_constraints_applied']=constraints
                result=self.operation('extrude',profile=profile.tolist(),height=thickness)
            self.expectations.append({'type':'bounding_extent','axis':2,'nominal_mm':thickness,'object_id':record['id']})
            if not base:
                for axis,role,value in ((0,'width',width),(1,'length',length)):
                    parameter=next(p for p in reversed(self.parameters) if p['object_id']==obj.id and p['role']==role)
                    self.expectations.append({'type':'bounding_extent','axis':axis,'nominal_mm':value,'object_id':record['id'],
                        'dimension_id':parameter.get('dimension_id'),'required':parameter['status']=='verified_dimensions',
                        'confidence_class':parameter['status']})
        elif kind=='revolution':
            length=self.parameter(obj,record,'length',lambda:self.inferred_span(record,0 if obj.axis=='horizontal' else 1))
            diameter=self.parameter(obj,record,'diameter',lambda:self.inferred_span(record,1 if obj.axis=='horizontal' else 0))
            values.update(length=length,diameter=diameter);self.initialize_transform(obj,record,values)
            axis=0 if obj.axis=='horizontal' else 1;radial=1-axis
            center=(b[radial]+b[radial+2])/2
            polygon=np.array(record['polygon_pdf']);side=np.median(polygon[:,radial])-center
            # A longitudinal material face must stay on one side of a declared
            # axis; never rotate individual off-axis holes into annular cavities.
            if polygon[:,radial].min()<center-1 and polygon[:,radial].max()>center+1:
                from .reconstruction import clip_half
                canonical=polygon if axis==0 else polygon[:,::-1]
                polygon=clip_half(canonical,center)
                if axis==1:polygon=polygon[:,::-1]
            profile=[[abs(float(p[radial]-center))/self.transform['scales'][radial],
                      float(p[axis]-b[axis])/self.transform['scales'][axis]] for p in polygon]
            result=self.operation('revolve',profile=profile)
            # Axis mapping of a revolve is independent from primitive placement.
            self.transform={'plane':'xz','origin_pdf':np.array([b[0],center]) if axis==0 else np.array([center,b[1]]),
                            'scales':self.transform['scales'],'revolve_axis':obj.axis}
        elif kind in ('hole','pattern'):
            if not base:raise ValueError('Subtractive feature requires a previously built base')
            diameter=self.parameter(obj,record,'diameter')
            default_depth=None
            if self.transform:
                span=(b[2:]-b[:2])/self.transform['scales']
                default_depth=(float(max(span)),'estimated_dimensions','depth_from_visible_feature_span')
            depth=self.parameter(obj,record,'depth',default_depth)
            origin,direction=self.hole_frame(obj,record,diameter,depth,require_opening=kind!='pattern')
            if kind=='hole':result=self.operation('hole',base=base,radius=diameter/2,height=depth,origin=origin,direction=direction)
            else:
                raw_count=self.parameter(obj,record,'count');count=int(raw_count);pitch=self.parameter(obj,record,'pitch_diameter')
                if count!=raw_count:raise ValueError('Pattern count must be an integer')
                if not 2<=count<=36:raise ValueError('Pattern count out of range')
                center_pdf=np.array(obj.center)*[self.page['width'],self.page['height']] if obj.center else mid
                scale=record['calibration']['scale_pt_per_mm'] or float(np.mean(self.transform['scales']))
                observed=[]
                for circle in self.page.get('geometry',{}).get('circles',[]):
                    props=circle.get('properties',{});point=np.array(props.get('center',[1e8,1e8]));radius=props.get('radius_pt',0)
                    if abs(2*radius/scale-diameter)>max(.15,diameter*.05):continue
                    if abs(np.linalg.norm(point-center_pdf)/scale-pitch/2)>max(.25,pitch*.025):continue
                    if any(np.linalg.norm(point-p)<max(.7,radius*.1) for p in observed):continue
                    observed.append(point)
                if len(observed)!=count:raise ValueError('Pattern count/diameter/pitch does not match observed distinct openings')
                angles=np.sort([np.arctan2(*(p-center_pdf)[::-1]) for p in observed]);gaps=np.diff(np.r_[angles,angles[0]+2*np.pi])
                if np.max(abs(gaps-2*np.pi/count))>.08:raise ValueError('Observed openings are not an equiangular circular pattern')
                phase=float(np.angle(np.mean(np.exp(1j*count*angles)))/count)
                center_origin=origin.copy()
                if abs(direction[2])>.9:origin[0]+=pitch/2*np.cos(phase);origin[1]+=pitch/2*np.sin(phase)
                elif abs(direction[0])>.9:origin[1]+=pitch/2*np.cos(phase);origin[2]-=pitch/2*np.sin(phase)
                else:raise ValueError('Unregistered pattern orientation')
                tool=self.operation('cylinder',radius=diameter/2,height=depth,origin=origin,direction=direction)
                result=self.operation('circular_pattern',base=base,tool=tool,count=count,origin=center_origin,direction=direction)
                record['proof'].update(observed_opening_count=len(observed),observed_openings_pdf=[p.tolist() for p in observed],phase_radians=phase)
                if abs(direction[2])>.9:
                    centers=[self.world(p)[:2] for p in observed]
                    self.expectations.append({'type':'hole_pattern','z_mm':origin[2]+depth/2,'centers_xy_mm':centers,
                        'radius_mm':diameter/2,'required':False,'object_id':record['id']})
        elif kind=='pocket':
            if not base or self.transform['plane']!='xy':raise ValueError('Pocket requires an established profile plane and base')
            depth=self.parameter(obj,record,'depth')
            profile=[self.world(p)[:2] for p in record['polygon_pdf']]
            tool=self.operation('extrude',profile=profile,height=depth)
            result=self.operation('cut',base=base,tool=tool)
        elif kind in ('chamfer','fillet'):
            if not base:raise ValueError('Edge feature requires a base')
            radius=self.parameter(obj,record,'radius')
            point=self.world(mid)
            selector={'point':point,'tolerance_mm':max(radius*2,.2),'kind':'nearest'}
            result=self.operation(kind,base=base,edge_selector=selector,**({'radius':radius} if kind=='fillet' else {'distance':radius}))
        else:raise ValueError('No compiler for object kind')
        self.features.append({'id':record['id'],'kind':kind,'source_object':obj.id,'status':'implemented'})
        return result

    def compile(self,plan,semantics):
        for step in plan.steps:
            if step.object_id not in self.graph:raise ValueError('Undeclared new object '+step.object_id)
            obj=self.graph[step.object_id]
            if obj.view_id not in self.views:raise ValueError('Unknown declared source view')
            before=len(self.operations);old_parameters=len(self.parameters)
            old_features=len(self.features);old_expectations=len(self.expectations);old_transform=self.transform
            try:
                if step.base_id and step.base_id not in self.results:raise ValueError('Earlier base object was rejected; dependent feature remains unresolved')
                obj=self.geometric_family(obj)
                try:record=self.registry.register(obj,self.views[obj.view_id])
                except ValueError:
                    if obj.kind!='extrusion' or not any(r['object_id']==obj.id for r in self.family_resolutions):raise
                    obj=obj.model_copy(update={'kind':'box'})
                    record=self.registry.register(obj,self.views[obj.view_id])
                    self.family_resolutions.append({'object_id':obj.id,'compiled_kind':'box','reason':'Visible bounding-envelope preview; a closed detailed profile is not established'})
                base=self.results.get(step.base_id)
                if obj.kind in ('hole','pocket','pattern') and step.action=='add':raise ValueError('Subtractive geometry cannot use add action')
                if step.action=='subtract' and obj.kind not in ('hole','pocket','pattern'):
                    tool=self.compile_object(obj,record)
                    if not base:raise ValueError('Subtraction has no base')
                    result=self.operation('cut',base=base,tool=tool)
                else:
                    result=self.compile_object(obj,record,base)
                    if step.action=='add' and base:result=self.operation('fuse',base=base,tool=result)
                self.results[obj.id]=result
            except (ValueError,KeyError,TypeError) as exc:
                self.operations=self.operations[:before];self.parameters=self.parameters[:old_parameters]
                self.features=self.features[:old_features];self.expectations=self.expectations[:old_expectations];self.transform=old_transform
                self.rejections.append({'object_id':step.object_id,'reason':str(exc),'stage':'semantic_compilation'})
        if not self.results:raise ValueError('No executable grounded geometric object: '+str(self.rejections))
        chosen=self.results.get(plan.result_id) or next(reversed(self.results.values()))
        # Only ancestors of the declared result exist in the delivered model.
        # Disconnected construction branches cannot earn feature/dimension credit.
        by_id={op['id']:op for op in self.operations};live={chosen};pending=[chosen]
        while pending:
            op=by_id[pending.pop()]
            for ref in (op.get('base'),op.get('tool')):
                if ref and ref not in live:live.add(ref);pending.append(ref)
        live_objects={local for local,result in self.results.items() if result in live}
        for local in self.results:
            if local not in live_objects:self.rejections.append({'object_id':local,'reason':'Not a dependency of the declared final result; no feature coverage credit'})
        self.operations=[op for op in self.operations if op['id'] in live]
        self.parameters=[p for p in self.parameters if p['object_id'] in live_objects]
        self.features=[f for f in self.features if f['source_object'] in live_objects]
        live_registry_ids={self.registry.records[local]['id'] for local in live_objects}
        self.expectations=[e for e in self.expectations if not e.get('object_id') or e['object_id'] in live_registry_ids]
        estimates=[p for p in self.parameters if p['status']=='estimated_dimensions']
        missing=semantics.missing_information+plan.missing_features+[r['reason'] for r in self.rejections]
        # Coverage and projection validation can downgrade B to C, never turn a
        # single matching contour into a complete A reconstruction.
        status='preliminary_geometry' if estimates or self.force_preliminary else 'partial_geometry'
        t=self.transform
        transform={'plane':t['plane'],'origin_pdf':t['origin_pdf'].tolist(),'scales_pt_per_mm':t['scales'].tolist(),
                   'revolve_axis':t.get('revolve_axis'),'shortening':t.get('shortening')}
        primary=self.registry.records[next(local for local in self.results if local in live_objects)]
        evidence={'page':self.page['page'],'page_size_pdf':{'width':self.page['width'],'height':self.page['height']},'source_sha256':self.document['sha256'],'drawing_id':self.document['drawing_id'],
            'llm_assisted':True,'independent_semantic_graph':True,'source_output_directory':str(Path(self.output).resolve()),
            'registered_objects':list(self.registry.records.values()),'semantic_views':[v.model_dump(mode='json') for v in semantics.views],
            'semantic_parameters':self.parameters,'dimension_groups':{s:[p for p in self.parameters if p['status']==s]
                for s in ('verified_dimensions','inferred_dimensions','estimated_dimensions','unknown_dimensions')},
            'expected_features':semantics.expected_features,'implemented_features':self.features,
            'registration':transform,'primary_bbox_pdf':primary['bbox_pdf'],'profile_id':primary['id'],
            'compiler_rejections':self.rejections,'ocr_conflicts':self.conflicts,
            'quality_level':'C' if status=='preliminary_geometry' else 'B','family_resolutions':self.family_resolutions,'full_reconstruction_verified':False}
        unresolved=[{'type':'semantic_unresolved','reason':s} for s in missing]
        unresolved+=[{'type':'estimated_dimension','reason':f"{p['object_id']}.{p['role']} = {p['value_mm']:.6g} мм: {p['reason']}"} for p in estimates]
        unresolved.append({'type':'whole_part_not_verified','reason':'Полнота всех видов и конструктивных элементов не доказана; модель не имеет уровня A.'})
        intent=CADIntent.model_validate({'provenance':'automatic_pdf','reconstruction_status':status,'evidence':evidence,
            'validation_expectations':self.expectations,'unresolved_features':unresolved,
            'description':'Независимая семантическая реконструкция по изображению и размерным свидетельствам.',
            'operations':self.operations,'result':chosen}).model_dump(mode='json')
        report={'page':self.page['page'],'status':status,'part_type':semantics.part_description,'strategy':'independent_semantic_constructive_graph',
                'dimension_bindings':[],'unresolved_features':unresolved,'registered_objects':list(self.registry.records.values()),
                'dimension_groups':evidence['dimension_groups'],'expected_features':semantics.expected_features}
        atomic_json(self.target/'registry.json',list(self.registry.records.values()))
        atomic_json(self.target/'compiled-intent.json',intent)
        return {'intent':intent,'interpretation':report,'origin':'llm_independent_semantic_graph'}


def compile_semantic(semantics,graph,plan,page,document,output,target):
    compiler=SemanticCompiler(page,document,output,target,semantics,graph)
    return compiler.compile(plan,semantics)
