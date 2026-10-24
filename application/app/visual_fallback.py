"""Image-derived meshes, explicitly D; no STEP substitution, filename rules or GT."""
import math
from pathlib import Path
import struct
import cv2
import numpy as np
from .storage import atomic_json


def alternate_vector_mask(page,folder):
    """Infer body stroke styles independently of a heavier sheet border."""
    original=cv2.imdecode(np.fromfile(Path(folder)/'original.png',np.uint8),0)
    if original is None:return None
    height,width=original.shape;sx=width/page['width'];sy=height/page['height'];records=[];vertical={}
    for g in page.get('geometry',{}).get('lines',[]):
        stroke=g.get('properties',{}).get('stroke_width_pt') or 0
        if g.get('source')!='vector' or not stroke:continue
        points=np.array(g['coordinates'],float)
        if len(points)!=2:continue
        a,b=points
        if abs(a[0]-b[0])<.3 and (a[0]<page['width']*.16 or a[0]>page['width']*.94):
            vertical.setdefault(round(a[0]),[]).extend(points[:,1].tolist())
        records.append((stroke,points))
    if not records:return None
    frame={x for x,ys in vertical.items() if max(ys)-min(ys)>page['height']*.55}
    eligible=[]
    for stroke,points in records:
        a,b=points;center=points.mean(axis=0)
        if abs(a[0]-b[0])<.3 and round(a[0]) in frame:continue
        if np.any(points[:,0]<page['width']*.04) or np.any(points[:,0]>page['width']*.96):continue
        if np.any(points[:,1]<page['height']*.04) or np.any(points[:,1]>page['height']*.94):continue
        if any(r['bbox'][0]<=center[0]<=r['bbox'][2] and r['bbox'][1]<=center[1]<=r['bbox'][3] for r in page['layout'].get('non_geometry_regions',[])):continue
        if np.linalg.norm(b-a)<.5:continue
        eligible.append((stroke,points))
    if not eligible:return None
    threshold=max(.28,float(np.percentile([s for s,_ in eligible],80))*.75)
    mask=np.zeros_like(original)
    for stroke,points in eligible:
        if stroke<threshold:continue
        samples=points[0]+(points[1]-points[0])*np.linspace(0,1,9)[:,None]
        pixels=np.round(samples*[sx,sy]).astype(int);pixels[:,0]=np.clip(pixels[:,0],0,width-1);pixels[:,1]=np.clip(pixels[:,1],0,height-1)
        if np.mean(original[pixels[:,1],pixels[:,0]]<200)<.45:continue
        cv2.polylines(mask,[np.round(points*[sx,sy]).astype(np.int32)],False,255,max(2,round(stroke*sx)))
    cv2.imencode('.png',mask)[1].tofile(str(Path(folder)/'visual-vector-strokes.png'))
    return mask


def alternate_vector_regions(page,folder):
    mask=alternate_vector_mask(page,folder)
    if mask is None:return []
    closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    records=[]
    for contour in cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        area=cv2.contourArea(contour);x,y,w,h=cv2.boundingRect(contour)
        if area<mask.size*.0008 or area>mask.size*.3 or min(w,h)<mask.shape[1]*.015:continue
        if area/(w*h)<.1:continue
        records.append((area,[x/mask.shape[1],y/mask.shape[0],(x+w-1)/mask.shape[1],(y+h-1)/mask.shape[0]]))
    return [b for _,b in sorted(records,reverse=True)[:4]]


def triangulate(points):
    """Ear clipping of an observed simple outline; reject degenerate polygons."""
    p=np.asarray(points,float)
    if cv2.contourArea(p.astype(np.float32),oriented=True)<0:p=p[::-1]
    indices=list(range(len(p)));faces=[];limit=len(p)**2
    def cross(a,b,c):
        u,v=b-a,c-a
        return u[0]*v[1]-u[1]*v[0]
    while len(indices)>3 and limit:
        limit-=1;found=False
        for j,b in enumerate(indices):
            a,c=indices[j-1],indices[(j+1)%len(indices)]
            if cross(p[a],p[b],p[c])<=1e-9:continue
            others=[i for i in indices if i not in (a,b,c)]
            if any(cross(p[a],p[b],p[i])>=-1e-9 and cross(p[b],p[c],p[i])>=-1e-9 and cross(p[c],p[a],p[i])>=-1e-9 for i in others):continue
            faces.append([a,b,c]);indices.pop(j);found=True;break
        if not found:raise ValueError('Observed silhouette is not a simple triangulable polygon')
    if len(indices)!=3:raise ValueError('Triangulation failed')
    return p,faces+[indices]


def prism(points,depth):
    p,f=triangulate(points);n=len(p)
    vertices=np.r_[np.c_[p,np.zeros(n)],np.c_[p,np.full(n,depth)]]
    faces=[[c,b,a] for a,b,c in f]+[[a+n,b+n,c+n] for a,b,c in f]
    for i in range(n):
        j=(i+1)%n;faces.extend([[i,j,j+n],[i,j+n,i+n]])
    return vertices,np.array(faces,int)


def revolution(profile,segments=64):
    # Profile is ordered along axis, radius >= 0. Rings + end caps, no made-up bore.
    profile=np.asarray(profile,float);vertices=[];faces=[]
    for radius,z in profile:
        vertices.extend([[radius*math.cos(t),radius*math.sin(t),z] for t in np.arange(segments)*2*math.pi/segments])
    for i in range(len(profile)-1):
        for j in range(segments):
            a=i*segments+j;b=i*segments+(j+1)%segments;c=b+segments;d=a+segments
            faces.extend([[a,b,c],[a,c,d]])
    first=len(vertices);vertices.append([0,0,profile[0,1]])
    last=len(vertices);vertices.append([0,0,profile[-1,1]])
    offset=(len(profile)-1)*segments
    for j in range(segments):
        k=(j+1)%segments;faces.extend([[first,k,j],[last,offset+j,offset+k]])
    return np.array(vertices),np.array(faces,int)


def save_mesh(vertices,faces,target,description,evidence):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
    if not len(faces) or not np.isfinite(vertices).all() or min(np.ptp(vertices,axis=0))<=0:raise ValueError('Empty/flat visual model')
    triangles=vertices[faces];volume=float(abs(np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()/6))
    preview={'vertices':vertices.tolist(),'triangles':faces.tolist(),'edges':[],'quality_level':'D',
             'reconstruction_status':'visual_preview','provenance':'automatic_pdf','description':description,
             'volume_mm3':volume,'valid':True,'solid_count':None,'parametric_cad':False,
             'available_formats':['stl'],'unresolved_features':[{'reason':'Приблизительная форма по изображению; размеры и внутренняя геометрия не подтверждены.'}],
             'evidence':evidence,'bounding_box':{'min':vertices.min(axis=0).tolist(),'max':vertices.max(axis=0).tolist()}}
    atomic_json(target/'preview.json',preview)
    with (target/'model.stl').open('wb') as stream:
        stream.write(b'Image-derived D preview; not verified CAD'.ljust(80,b' '));stream.write(struct.pack('<I',len(faces)))
        for tri in triangles:
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=max(np.linalg.norm(normal),1e-12)
            stream.write(struct.pack('<12fH',*normal,*tri.reshape(-1),0))
    atomic_json(target/'validation.json',{'quality_level':'D','overall_status':'visual_preview','mesh_finite':True,
        'source_evidence':evidence,'scope':'Image-derived approximate shape; not verified dimensions or BRep'})
    return preview


class Visual3DReconstructionFallback:
    def __init__(self,document,output):self.document=document;self.output=Path(output)

    def outline(self,body):
        page=next(p for p in self.document['pages'] if p['page']==body.page)
        folder=self.output/f'page-{body.page}'
        if body._observed_profile_id:
            source=next(p for p in page['layout']['profile_candidates'] if p['id']==body._observed_profile_id)
            points=np.array(source['polygon_pdf'],np.float32)
            box=np.array(source['bbox'],float);span=box[2:]-box[:2]
            # Material contours can touch at a stroke-width junction. Repair
            # only the D preview on a bounded raster copy, preserving concavity.
            scale=650/max(span);local=np.round((points-box[:2])*scale+4).astype(np.int32)
            mask=np.zeros(tuple((np.ceil(span[::-1]*scale)+9).astype(int)),np.uint8)
            cv2.fillPoly(mask,[local],255);mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
            contour=max(cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea)
            points=(cv2.approxPolyDP(contour,max(1,cv2.arcLength(contour,True)*.002),True)[:,0]-4)/scale+box[:2]
            return (points-box[:2])/span,{'page':body.page,'bbox_normalized':(box/([page['width'],page['height']]*2)).tolist(),
                'source_profile_id':source['id'],'method':'observed_non_axisymmetric_material_face_angular_visual_extrusion',
                'hatch_witnesses':source.get('hatch_line_count',0),'all_unconfirmed_parameters':'estimated','family':body.family}
        path=folder/'full-physical-strokes.png'
        if not path.is_file():path=folder/'physical-strokes.png'
        mask=cv2.imdecode(np.fromfile(path,np.uint8),0)
        if mask is None:raise ValueError('No observed physical stroke mask')
        alternative=alternate_vector_mask(page,folder)
        if alternative is not None:mask|=alternative
        x,y,u,v=np.round(np.array(body.bbox)*([mask.shape[1],mask.shape[0]]*2)).astype(int)
        x=max(0,x);y=max(0,y);u=min(mask.shape[1],u);v=min(mask.shape[0],v)
        crop=mask[y:v,x:u].copy()
        original=cv2.imdecode(np.fromfile(folder/'original.png',np.uint8),0)
        if original is not None and original.shape==mask.shape:
            crop|=(original[y:v,x:u]<155).astype(np.uint8)*255
        if not crop.size or np.count_nonzero(crop)<50:raise ValueError('Proposed view has insufficient physical strokes')
        # Bounded gap closure for visual approximation only; keep exact source intact.
        small=cv2.resize(crop,(max(2,round(crop.shape[1]*min(1,900/max(crop.shape)))),max(2,round(crop.shape[0]*min(1,900/max(crop.shape))))))
        closed=cv2.morphologyEx(small,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        # Reconnect collinear observed outer sides across graphical breaks. This
        # belongs only to approximate reconstruction and is recorded explicitly.
        bridged=0
        for transposed in (False,True):
            view=closed.T if transposed else closed
            for column in range(view.shape[1]):
                ys=np.where(view[:,column]>0)[0]
                if len(ys)<view.shape[0]*.35:continue
                for a,b in zip(ys,ys[1:]):
                    if 3<b-a<view.shape[0]*.3:
                        view[a:b+1,column]=255;bridged+=1
        contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        choices=[c for c in contours if cv2.contourArea(c)>small.size*.025]
        if not choices:
            regions=alternate_vector_regions(page,folder)
            if not regions:raise ValueError('No significant source-supported silhouette')
            nearest=min(regions,key=lambda b:np.linalg.norm((np.array(b[:2])+b[2:])/2-(np.array(body.bbox[:2])+body.bbox[2:])/2))
            if np.max(abs(np.array(nearest)-body.bbox))<.002:raise ValueError('No closed observed silhouette after alternate localization')
            traced,evidence=self.outline(body.model_copy(update={'bbox':tuple(nearest)}))
            evidence['semantic_crop_relocalized']=list(body.bbox)
            return traced,evidence
        contour=max(choices,key=cv2.contourArea)
        epsilon=max(1,cv2.arcLength(contour,True)*.0025)
        points=cv2.approxPolyDP(contour,epsilon,True)[:,0].astype(float)
        if len(points)<3:raise ValueError('Insufficient outline vertices')
        # A long line alone is not a volume. Require substantial two-dimensional coverage.
        if cv2.contourArea(points.astype(np.float32))<small.size*.02:raise ValueError('Stroke-only region has no coherent silhouette')
        points/=np.array([small.shape[1],small.shape[0]])
        evidence={'page':body.page,'bbox_normalized':list(body.bbox),'source_mask':str(path),
                  'source_outline_normalized_to_crop':points.tolist(),'method':'observed_physical_strokes_bounded_visual_gap_closure',
                  'outline_area_fraction':float(cv2.contourArea(points.astype(np.float32))),
                  'collinear_visual_gaps_bridged':bridged,
                  'all_unconfirmed_parameters':'estimated','family':body.family}
        return points,evidence

    def construct(self,body):
        points,evidence=self.outline(body)
        lo=points.min(axis=0);span=np.ptp(points,axis=0);points=(points-lo)/span
        if body.family=='rotational':
            axial=0 if body.axis=='horizontal' else 1
            p=points[:,[axial,1-axial]]
            # Sample the observed envelope, retaining shoulders/tapers. Internal voids remain unknown.
            contour=np.round(p*[700,400]).astype(np.int32);mask=np.zeros((403,703),np.uint8);cv2.fillPoly(mask,[contour],255)
            profile=[]
            for z in np.linspace(0,700,65).astype(int):
                ys=np.where(mask[:,max(0,z-1):min(703,z+2)].any(axis=1))[0]
                if not len(ys):continue
                radius=max(abs(ys-200))/200*body.width.value/2
                profile.append([max(radius,body.width.value*.002),z/700*body.length.value])
            if len(profile)<4:raise ValueError('No axial radial envelope')
            profile=np.array(profile);profile[:,0]*=(body.width.value/2)/profile[:,0].max()
            vertices,faces=revolution(profile)
            evidence['radial_profile_mm']=profile.tolist()
            operation={'id':'visual_body','type':'revolve','profile':[[0,float(profile[0,1])]]+profile.tolist()+[[0,float(profile[-1,1])]]}
        else:
            xy=points*[body.width.value,body.length.value]
            vertices,faces=prism(xy,body.depth.value)
            operation={'id':'visual_body','type':'extrude','profile':xy.tolist(),'height':body.depth.value}
        evidence['parameters']={k:getattr(body,k).model_dump() for k in ('length','width','depth')}
        return vertices,faces,operation,evidence

    def build(self,plan,target):
        vertices=[];faces=[];evidence=[];failures=[];offset=0
        for body in plan.bodies:
            try:
                v,f,_,e=self.construct(body)
                direction=np.array(body.direction,float);direction/=np.linalg.norm(direction)
                cross=np.cross([0,0,1],direction);cos=float(direction[2]);sine=float(np.linalg.norm(cross))
                if sine>1e-9:
                    axis=cross/sine;k=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                    rotation=np.eye(3)+sine*k+(1-cos)*(k@k)
                else:rotation=np.diag([1,-1,-1]) if cos<0 else np.eye(3)
                v=v@rotation.T+body.origin
                vertices.extend(v.tolist());faces.extend((f+offset).tolist());offset+=len(v)
                e.update(origin_mm=list(body.origin),direction=list(body.direction));evidence.append(e)
                # Only explicit components become an assembly; ordinary views
                # remain alternative observations of the same physical object.
                if plan.composition=='single':break
            except Exception as exc:failures.append({'page':body.page,'reason':str(exc)})
        if not len(vertices):raise ValueError('No meaningful source-supported visual form: '+str(failures))
        return save_mesh(vertices,faces,target,plan.part_description,{'views':evidence,'composition':plan.composition,
            'components_are_approximate_not_boolean_union':len(evidence)>1,'rejected_views':failures,'uncertainties':plan.uncertainties})
