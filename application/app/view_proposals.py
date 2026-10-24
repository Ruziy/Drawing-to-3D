"""Multiscale localization proposals. Joining ROIs never closes CAD material."""
import cv2
import numpy as np


def propose_views(page,output):
    folder=output/f"page-{page['page']}"
    path=folder/'full-physical-strokes.png'
    full=path.is_file()
    if not full:path=folder/'physical-strokes.png'
    mask=cv2.imdecode(np.fromfile(path,np.uint8),0) if path.is_file() else None
    if mask is None:return []
    if not full:
        original=cv2.imdecode(np.fromfile(folder/'original.png',np.uint8),0)
        ink=(original<150).astype(np.uint8)*255
        heavy=cv2.dilate((cv2.distanceTransform(ink,cv2.DIST_L2,3)>=1.7).astype(np.uint8)*255,np.ones((3,3),np.uint8))
        mask[round(mask.shape[0]*.7):]=heavy[round(mask.shape[0]*.7):]
    sx,sy=mask.shape[1]/page['width'],mask.shape[0]/page['height']
    scale=1.5
    small=cv2.resize(mask,(round(page['width']*scale),round(page['height']*scale)),interpolation=cv2.INTER_AREA)
    small=(small>100).astype(np.uint8)*255
    # Remove small symbol strokes before connected-component localization.
    long=cv2.morphologyEx(small,cv2.MORPH_OPEN,np.ones((1,24),np.uint8))
    long|=cv2.morphologyEx(small,cv2.MORPH_OPEN,np.ones((24,1),np.uint8))
    long[:round(page['height']*.065*scale)]=0;long[round(page['height']*.92*scale):]=0
    for region in page.get('views',[]):
        if region.get('type')!='title_block':continue
        x0,y0,x1,y1=[round(v*scale) for v in region['bbox']]
        long[y0:y1,x0:x1]=0
    for region in page['layout'].get('non_geometry_regions',[]):
        x0,y0,x1,y1=[round(v*scale) for v in region['bbox']]
        long[y0:y1,x0:x1]=0
    long[:,:round(page['width']*.13*scale)]=0;long[:,round(page['width']*.93*scale):]=0
    joined=cv2.dilate(long,np.ones((9,9),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(joined)
    boxes=[]
    for label in range(1,count):
        x,y,w,h,area=stats[label]
        if area<150 or max(w,h)<35*scale:continue
        # Recover exact stroke bounds before dilation.
        yy,xx=np.where((labels==label)&(long>0))
        if not len(xx):continue
        boxes.append(np.array([xx.min(),yy.min(),xx.max(),yy.max()],float)/scale)
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(boxes):
            for j in range(i+1,len(boxes)):
                b=boxes[j];sa=a[2:]-a[:2];sb=b[2:]-b[:2]
                xalign=max(abs(a[0]-b[0]),abs(a[2]-b[2]))<max(5,max(sa[0],sb[0])*.05)
                yalign=max(abs(a[1]-b[1]),abs(a[3]-b[3]))<max(5,max(sa[1],sb[1])*.05)
                xgap=max(0,max(a[0],b[0])-min(a[2],b[2]));ygap=max(0,max(a[1],b[1])-min(a[3],b[3]))
                side_x=min(abs(a[0]-b[0]),abs(a[2]-b[2]))<2.5 and min(sa[1],sb[1])>25 and min(sa[0],sb[0])<max(sa[0],sb[0])*.6
                side_y=min(abs(a[1]-b[1]),abs(a[3]-b[3]))<2.5 and min(sa[0],sb[0])>25 and min(sa[1],sb[1])<max(sa[1],sb[1])*.6
                if (xalign and ygap<min(35,max(sa[1],sb[1])*.4)) or (yalign and xgap<min(35,max(sa[0],sb[0])*.4)) or (side_x and ygap<22) or (side_y and xgap<22):
                    boxes[i]=np.r_[np.minimum(a[:2],b[:2]),np.maximum(a[2:],b[2:])];boxes.pop(j);changed=True;break
            if changed:break
    for face in page['layout'].get('profile_candidates',[])[:12]:
        if face.get('contour_depth')!=0 or face.get('area_pt2',0)<500:continue
        b=np.array(face['bbox'],float)
        if b[3]>page['height']*.92:continue
        if any(np.max(abs(b-a))<5 for a in boxes):continue
        boxes.append(b)
    from .non_geometry import overlap_fraction
    boxes=[b for b in boxes if not any(overlap_fraction(b,r['bbox'])>.1 for r in page['layout'].get('non_geometry_regions',[]))]
    boxes=sorted(boxes,key=lambda b:-(b[2]-b[0])*(b[3]-b[1]))[:8]
    return [{'candidate':f'ROI{i+1}','bbox':[round(float(v/s),5) for v,s in zip(b,[page['width'],page['height']]*2)],
             'method':'long_stroke_clusters_with_collinear_break_roi_linking','not_a_confirmed_material_boundary':True}
            for i,b in enumerate(boxes) if min(b[2:]-b[:2])>8]
