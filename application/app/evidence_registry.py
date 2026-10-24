"""Register newly proposed sheet geometry against original image pixels.

LLM local names are never interpreted as existing CV IDs. Registry IDs are
assigned here; normalized page coordinates are converted exactly once.
"""
import math
from pathlib import Path
import cv2
import numpy as np
from .calibration import robust_view_scale


def pdf_box(box,page):return np.array(box,float)*[page['width'],page['height'],page['width'],page['height']]


class EvidenceRegistry:
    def __init__(self,page,output):
        self.page=page;self.records={}
        image=cv2.imdecode(np.fromfile(Path(output)/f"page-{page['page']}"/'original.png',np.uint8),0)
        if image is None:raise ValueError('Original raster missing')
        self.image=image;self.pixels=np.array([image.shape[1]/page['width'],image.shape[0]/page['height']])
        directory=Path(output)/f"page-{page['page']}"
        physical=directory/'full-physical-strokes.png'
        if not physical.is_file():physical=directory/'physical-strokes.png'
        self.physical=cv2.imdecode(np.fromfile(physical,np.uint8),0) if physical.is_file() else None
        # Two thresholds preserve fine strokes and suppress light raster noise.
        self.ink=(image<185).astype(np.uint8)*255
        self.distance=cv2.distanceTransform(255-self.ink,cv2.DIST_L2,3)

    def support(self,points,tolerance_pt=1.2):
        points=np.asarray(points,float);indices=np.round(points*self.pixels).astype(int)
        valid=(indices[:,0]>=0)&(indices[:,0]<self.image.shape[1])&(indices[:,1]>=0)&(indices[:,1]<self.image.shape[0])
        distances=np.full(len(indices),1e6,float)
        distances[valid]=self.distance[indices[valid,1],indices[valid,0]]/float(np.mean(self.pixels))
        return float(np.mean(distances<=tolerance_pt)),float(np.percentile(distances,95))

    def boundary_support(self,points):
        samples=[]
        for a,b in zip(points,np.roll(points,-1,axis=0)):
            a,b=np.asarray(a),np.asarray(b);n=max(8,min(1000,int(np.linalg.norm(b-a)*3)))
            samples.extend(a+(b-a)*np.linspace(0,1,n)[:,None])
        return self.support(samples)

    def snap_box(self,box):
        """Snap proposed sides to long original strokes, never to OCR text bounds."""
        box=np.array(box,float);pad=max(3,min(9,max(box[2:]-box[:2])*.025))
        lower=np.maximum(box[:2]-pad,0);upper=np.minimum(box[2:]+pad,[self.page['width'],self.page['height']])
        p0=np.floor(lower*self.pixels).astype(int);p1=np.ceil(upper*self.pixels).astype(int)
        crop=self.ink[p0[1]:p1[1],p0[0]:p1[0]]
        lines=cv2.HoughLinesP(crop,1,np.pi/720,threshold=30,minLineLength=max(20,int(min(crop.shape)*.24)),maxLineGap=5)
        if lines is None:return box
        records=[]
        for line in lines[:,0]:
            points=(line.reshape(2,2)+p0)/self.pixels;delta=abs(points[1]-points[0])
            axis=0 if delta[0]>delta[1] else 1
            if delta[1-axis]>max(.8,delta[axis]*.015):continue
            records.append((axis,points,float(delta[axis])))
        result=box.copy()
        for index in range(4):
            normal=index%2;axis=1-normal
            span=box[axis+2]-box[axis]
            candidates=[]
            for orientation,p,length in records:
                if orientation!=axis or length<span*.35:continue
                coordinate=float(p[:,normal].mean());delta=abs(coordinate-box[index])
                if delta>pad:continue
                overlap=max(0,min(p[:,axis].max(),box[axis+2])-max(p[:,axis].min(),box[axis]))
                if overlap<span*.3:continue
                candidates.append((delta-.002*length,coordinate))
            if candidates:result[index]=min(candidates)[1]
        return result if all(result[2:]>result[:2]+2) else box

    def profile(self,obj,box,shortened=False):
        proposed=np.array(obj.profile_points,float)*[self.page['width'],self.page['height']] if obj.profile_points else None
        if proposed is not None:
            support,p95=self.boundary_support(proposed)
            if support>=.80 and p95<=3:return proposed,{'method':'llm_traced_original_strokes','edge_support':support,'p95_distance_pt':p95}
        p0=np.floor(box[:2]*self.pixels).astype(int);p1=np.ceil(box[2:]*self.pixels).astype(int)
        mask=self.physical if self.physical is not None else self.ink
        crop=mask[p0[1]:p1[1]+1,p0[0]:p1[0]+1].copy()
        if not crop.size:raise ValueError('Empty proposed profile region')
        # Interior faces can close despite drawing strokes intersecting outside the view.
        closed=cv2.morphologyEx(crop,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        break_bridge=False
        if shortened:
            # Only reconnect opposite, observed COLLINEAR outer sides across a
            # graphical break. Interior material is not inferred by this mask.
            for x in (0,closed.shape[1]-1):
                strip=closed[:,max(0,x-4):min(closed.shape[1],x+5)]
                observed=np.where(np.any(strip>0,axis=1))[0]
                for a,b in zip(observed,observed[1:]):
                    if 3<b-a<closed.shape[0]*.25:
                        cv2.line(closed,(x,int(a)),(x,int(b)),255,3);break_bridge=True
        contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        choices=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            if area<max(100,crop.size*.035):continue
            x,y,w,h=cv2.boundingRect(contour)
            if w<crop.shape[1]*.6 or h<crop.shape[0]*.6:continue
            filled=np.zeros_like(closed);cv2.drawContours(filled,[contour],-1,255,cv2.FILLED)
            filled=cv2.morphologyEx(filled,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
            cleaned=cv2.findContours(filled,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
            if not cleaned:continue
            contour=max(cleaned,key=cv2.contourArea)
            points=cv2.approxPolyDP(contour,max(1.2,cv2.arcLength(contour,True)*.0012),True)[:,0]
            points=(points+p0)/self.pixels
            support,p95=self.boundary_support(points)
            if support<.75:continue
            if len(points)>160:continue
            choices.append((area,points,{'method':'new_local_raster_closed_face','edge_support':support,'p95_distance_pt':p95,
                                         'collinear_graphical_break_bridged':break_bridge}))
        if not choices:raise ValueError('No closed original-stroke profile inside proposed region')
        _,points,diagnostic=max(choices,key=lambda v:v[0]);return points,diagnostic

    def register(self,obj,view):
        proposed=pdf_box(obj.bbox,self.page);viewbox=pdf_box(view.bbox,self.page)
        from .non_geometry import overlap_fraction
        if any(overlap_fraction(proposed,r['bbox'])>.15 for r in self.page.get('layout',{}).get('non_geometry_regions',[])):
            raise ValueError('Geometry overlaps a title/revision table, not a part view')
        if proposed[1]>self.page['height']*.84:raise ValueError('Geometry proposal lies in title block')
        if min(proposed[2:]-proposed[:2])<1:raise ValueError('Geometry proposal too small')
        intersection=np.maximum(0,np.minimum(proposed[2:],viewbox[2:])-np.maximum(proposed[:2],viewbox[:2])).prod()
        if intersection/max((proposed[2:]-proposed[:2]).prod(),1)<.70:raise ValueError('Object not inside its source view')
        box=self.snap_box(proposed);x0,y0,x1,y1=box
        if obj.kind in ('extrusion','revolution','pocket'):
            polygon,proof=self.profile(obj,box,view.shortened)
        else:
            polygon=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
            # Long opposite sides establish the primitive footprint; a round end
            # view can independently establish radius later in multiview validation.
            axis=0 if obj.axis=='horizontal' else 1
            samples=[]
            for side in (0,1):
                along=np.linspace(box[axis]+.1*(box[axis+2]-box[axis]),box[axis+2]-.1*(box[axis+2]-box[axis]),120)
                cross=np.full_like(along,box[1-axis+2*side])
                samples.append(np.c_[along,cross] if axis==0 else np.c_[cross,along])
            supports=[self.support(s,1.7)[0] for s in samples]
            circular=False
            if obj.kind in ('tube','cylinder') and obj.axis=='normal':
                if abs((x1-x0)/(y1-y0)-1)>.10:raise ValueError('Normal cylinder requires a circular source end view')
                angles=np.linspace(0,2*np.pi,400)
                points=np.c_[(x0+x1)/2+(x1-x0)/2*np.cos(angles),(y0+y1)/2+(y1-y0)/2*np.sin(angles)]
                circular=self.support(points,1.7)[0]>=.65
                if not circular:raise ValueError('Normal cylinder has no observed circular outline')
            if min(supports)<.25 and not circular and obj.kind not in ('hole','pattern','fillet','chamfer'):
                raise ValueError('Primitive footprint lacks two observed boundary sides')
            proof={'method':'new_semantic_primitive_observed_parallel_sides','side_support':supports,
                   'edge_support':min(supports),'shortened_source_view':view.shortened,'circular_end_view_confirmed':circular}
        assigned=f"LLM_P{self.page['page']}_{len(self.records)+1:03d}"
        record={'id':assigned,'local_id':obj.id,'page':self.page['page'],'view_id':obj.view_id,
                'proposed_bbox_pdf':proposed.tolist(),'bbox_pdf':box.tolist(),'polygon_pdf':polygon.tolist(),
                'origin':'local_multimodal_semantic_graph','proof':proof,
                'calibration':robust_view_scale(self.page,viewbox),'model_confidence_used_as_geometry_proof':False}
        self.records[obj.id]=record
        return record

    def inner_wall_spans(self,record,axis):
        b=np.array(record['bbox_pdf']);p0=np.floor(b[:2]*self.pixels).astype(int);p1=np.ceil(b[2:]*self.pixels).astype(int)
        crop=self.ink[p0[1]:p1[1]+1,p0[0]:p1[0]+1]
        longitudinal=0 if axis=='horizontal' else 1;radial=1-longitudinal
        length=(b[longitudinal+2]-b[longitudinal])*.2*self.pixels[longitudinal]
        lines=cv2.HoughLinesP(crop,1,np.pi/720,threshold=35,minLineLength=max(25,int(length)),maxLineGap=5)
        if lines is None:return []
        positions=[];center=(b[radial]+b[radial+2])/2;width=b[radial+2]-b[radial]
        for line in lines[:,0]:
            points=(line.reshape(2,2)+p0)/self.pixels;delta=abs(points[1]-points[0])
            if delta[radial]>.8 or delta[longitudinal]<(b[longitudinal+2]-b[longitudinal])*.19:continue
            position=float(points[:,radial].mean())
            if abs(position-center)<width*.15 or min(position-b[radial],b[radial+2]-position)<width*.04:continue
            if not any(abs(position-p)<.8 for p in positions):positions.append(position)
        return [right-left for left in positions for right in positions if left<center<right and abs((left+right)/2-center)<1.5]

    def through_wall_evidence(self,record,axis,diameter_pt):
        """A local pair of lines proves a cavity, not a through bore.

        Require the same two walls to reach both observed end faces. A declared
        graphical break permits missing middle ink, never a missing end.
        """
        if axis=='normal':return {'confirmed':False,'reason':'End opening does not establish axial depth'}
        b=np.array(record['bbox_pdf']);longitudinal=0 if axis=='horizontal' else 1;radial=1-longitudinal
        p0=np.floor(b[:2]*self.pixels).astype(int);p1=np.ceil(b[2:]*self.pixels).astype(int)
        crop=self.ink[p0[1]:p1[1]+1,p0[0]:p1[0]+1]
        span=b[longitudinal+2]-b[longitudinal];center=(b[radial]+b[radial+2])/2
        lines=cv2.HoughLinesP(crop,1,np.pi/720,threshold=25,
            minLineLength=max(15,int(span*self.pixels[longitudinal]*.12)),maxLineGap=7)
        intervals=[[],[]]
        for line in ([] if lines is None else lines[:,0]):
            points=(line.reshape(2,2)+p0)/self.pixels;delta=abs(points[1]-points[0])
            if delta[radial]>1.2 or delta[longitudinal]<span*.1:continue
            position=float(points[:,radial].mean())
            for side in (0,1):
                expected=center+(-.5 if side==0 else .5)*diameter_pt
                if abs(position-expected)<max(1.2,diameter_pt*.025):
                    intervals[side].append(sorted(points[:,longitudinal].tolist()))
        proof=[]
        for segments in intervals:
            merged=[]
            for lo,hi in sorted(segments):
                if merged and lo<=merged[-1][1]+1:merged[-1][1]=max(hi,merged[-1][1])
                else:merged.append([lo,hi])
            coverage=sum(hi-lo for lo,hi in merged)/span
            ends=bool(merged and merged[0][0]-b[longitudinal]<max(2,span*.08) and
                b[longitudinal+2]-merged[-1][1]<max(2,span*.08))
            proof.append({'coverage_fraction':coverage,'both_end_faces_reached':ends,'intervals_pdf':merged})
        minimum=.50 if record['proof'].get('shortened_source_view') else .78
        return {'confirmed':all(p['both_end_faces_reached'] and p['coverage_fraction']>=minimum for p in proof),
                'walls':proof,'minimum_coverage':minimum,'reason':'Observed coaxial walls must reach both end faces'}
