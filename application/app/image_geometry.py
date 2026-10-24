"""Observable 2D primitives; none are automatically treated as part boundaries."""
import math

import cv2
import numpy as np

from .schemas import GeometryElement, Region


def bounds(points):
    xs, ys = zip(*points)
    return (min(xs), min(ys), max(xs), max(ys))


def circular_fit(points):
    a = np.asarray(points, dtype=float)
    if len(a) < 5:
        return None
    center, *_ = np.linalg.lstsq(np.c_[2*a[:,0], 2*a[:,1], np.ones(len(a))], (a*a).sum(axis=1), rcond=None)
    c = center[:2]
    radii = np.linalg.norm(a-c, axis=1)
    radius = float(radii.mean())
    err = float(np.max(np.abs(radii-radius)))
    if radius < .5 or err > max(.035, radius*.0015):
        return None
    angles = np.unwrap(np.arctan2(a[:,1]-c[1], a[:,0]-c[0]))
    sweep = float(angles[-1]-angles[0])
    if abs(sweep) < math.radians(20):
        return None
    return c.tolist(), radius, err, float(angles[0]), sweep


def vector_geometry(page):
    geometry = {k:[] for k in ("lines", "circles", "arcs", "polylines", "contours", "intersections")}
    counter = 0

    def add(kind, points, method, props):
        nonlocal counter
        counter += 1
        item = GeometryElement(id=f"V{counter:06d}",type=kind,coordinates=points,bbox=bounds(points),
            source="vector",method=method,properties=props)
        geometry[{"line":"lines","circle":"circles","arc":"arcs","polyline":"polylines"}[kind]].append(item)
        return item

    for pi, path in enumerate(page.get_drawings()):
        common = {"path_index":pi,"stroke_width_pt":path.get("width"),"color":path.get("color"),
                  "fill":path.get("fill"),"dashes":path.get("dashes"),"closed":path.get("closePath"),
                  "pdf_path_type":path.get("type"),"semantic_role":"unknown"}
        samples_all = []
        curves = []
        for cmd in path["items"]:
            if cmd[0] == "l":
                add("line",[tuple(cmd[1]),tuple(cmd[2])],"pymupdf_path_line",common)
            elif cmd[0] == "re":
                rect = cmd[1]
                points = [tuple(rect.tl),tuple(rect.tr),tuple(rect.br),tuple(rect.bl),tuple(rect.tl)]
                add("polyline",points,"pymupdf_path_rectangle",common)
            elif cmd[0] == "qu":
                q = cmd[1]
                add("polyline",[tuple(q.ul),tuple(q.ur),tuple(q.lr),tuple(q.ll),tuple(q.ul)],"pymupdf_path_quad",common)
            elif cmd[0] == "c":
                controls = np.array([tuple(p) for p in cmd[1:]], dtype=float)
                t = np.linspace(0,1,17)[:,None]
                pts = (1-t)**3*controls[0]+3*(1-t)**2*t*controls[1]+3*(1-t)*t**2*controls[2]+t**3*controls[3]
                samples_all.extend(pts.tolist())
                curves.append((pts, controls))
        entire = circular_fit(samples_all) if len(curves) >= 2 and len(curves) == len(path["items"]) else None
        if entire and np.linalg.norm(np.asarray(samples_all[0])-samples_all[-1]) < .05 and abs(entire[4]) > 6.1:
            c,r,e,start,sweep = entire
            obj = add("circle",[(c[0]-r,c[1]-r),(c[0]+r,c[1]+r)],"bezier_circle_fit",{
                **common,"center":c,"radius_pt":r,"max_radial_error_pt":e,
                "bezier_control_points":[x[1].tolist() for x in curves]})
        else:
            for pts, controls in curves:
                fit = circular_fit(pts)
                props = {**common,"bezier_control_points":controls.tolist()}
                if fit:
                    c,r,e,start,sweep = fit
                    props.update(center=c,radius_pt=r,max_radial_error_pt=e,start_radians=start,sweep_radians=sweep)
                    add("arc",pts.tolist(),"bezier_circular_arc_fit",props)
                else:
                    add("polyline",pts.tolist(),"sampled_cubic_bezier",props)
    return geometry


def raster_geometry(rgb, pdf_width, pdf_height):
    # Original 300 DPI image is never modified. Detection uses <=150 DPI equivalent.
    h0,w0 = rgb.shape[:2]
    scale = min(1, 2000/max(w0,h0))
    work = cv2.resize(rgb, (round(w0*scale),round(h0*scale)), interpolation=cv2.INTER_AREA)
    h,w = work.shape[:2]
    sx,sy = pdf_width/w,pdf_height/h
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    count,labels,stats,_ = cv2.connectedComponentsWithStats(binary,8)
    keep = np.zeros(count,dtype=np.uint8)
    keep[1:] = (stats[1:,cv2.CC_STAT_AREA] >= 3)*255
    clean = keep[labels]
    geometry = {k:[] for k in ("lines","circles","arcs","polylines","contours","intersections")}
    counter = 0

    def point(p):
        return (float(p[0]*sx),float(p[1]*sy))

    def add(kind,points,method,props=None):
        nonlocal counter
        counter += 1
        obj = GeometryElement(id=f"R{counter:06d}",type=kind,coordinates=points,bbox=bounds(points),source="raster",method=method,properties=props or {})
        geometry[{"line":"lines","circle":"circles","arc":"arcs","contour":"contours","intersection":"intersections"}[kind]].append(obj)
        return obj

    lines = cv2.HoughLinesP(clean,1,np.pi/180,threshold=40,minLineLength=25,maxLineGap=4)
    line_count = len(lines) if lines is not None else 0
    if lines is not None:
        # Longest first, deterministic. Multiple raster edges can represent one stroke.
        lines = sorted(lines[:,0].tolist(),key=lambda p: -((p[2]-p[0])**2+(p[3]-p[1])**2))
        for x1,y1,x2,y2 in lines[:1500]:
            add("line",[point((x1,y1)),point((x2,y2))],"opencv_hough_lines_p",{"pixel_coordinates":[x1,y1,x2,y2],"semantic_role":"unknown"})
    contours,_ = cv2.findContours(clean,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    eligible = [c for c in contours if cv2.arcLength(c,True) >= 16 and cv2.contourArea(c) >= 4]
    eligible.sort(key=lambda c:-cv2.contourArea(c))
    for c in eligible[:1200]:
        perimeter = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,.7,True)[:,0,:]
        area = cv2.contourArea(c)
        add("contour",[point(p) for p in approx],"opencv_find_contours",{"area_pt2":area*sx*sy,"closed":True,"semantic_role":"unknown"})
        # Full circle candidates, supported by the complete contour rather than Hough guesses.
        if len(c) >= 10 and 4*np.pi*area/(perimeter*perimeter) > .80:
            (cx,cy),r = cv2.minEnclosingCircle(c)
            radial = np.linalg.norm(c[:,0,:]-[cx,cy],axis=1)
            if r >= 4 and np.std(radial)/r < .08:
                add("circle",[point((cx-r,cy-r)),point((cx+r,cy+r))],"closed_contour_circularity",{
                    "center":point((cx,cy)),"radius_pt":r*(sx+sy)/2,"circularity":4*np.pi*area/(perimeter*perimeter),"status":"candidate"})
        # Partial arcs: fit ellipse, require near-circular axes and a large angular span.
        if len(c) >= 12 and area/(perimeter*perimeter) < .015:
            try:
                (cx,cy),(a,b),angle = cv2.fitEllipse(c)
                if min(a,b) > 12 and max(a,b)/min(a,b) < 1.08:
                    radii = np.linalg.norm(c[:,0,:]-[cx,cy],axis=1)
                    r = (a+b)/4
                    angles = np.arctan2(c[:,0,1]-cy,c[:,0,0]-cx)
                    sorted_angles = np.sort(angles)
                    gaps = np.diff(np.r_[sorted_angles,sorted_angles[0]+2*np.pi])
                    sweep = float(2*np.pi-gaps.max())
                    if np.max(np.abs(radii-r)) < max(1.5,r*.04) and .5 < sweep < 5.8:
                        add("arc",[point(p) for p in approx],"opencv_fit_ellipse_near_circle",{
                            "center":point((cx,cy)),"radius_pt":r*(sx+sy)/2,"sweep_radians":sweep,"status":"candidate"})
            except cv2.error:
                pass

    # Segment intersections, capped to longest lines. Each retains source IDs.
    seen = set()
    for i,la in enumerate(geometry["lines"][:300]):
        p,q = np.asarray(la.coordinates)
        a = q-p
        for lb in geometry["lines"][:300][i+1:]:
            r,s = np.asarray(lb.coordinates)
            b = s-r
            cross = a[0]*b[1]-a[1]*b[0]
            if abs(cross) < 1e-7:
                continue
            v = r-p
            t = (v[0]*b[1]-v[1]*b[0])/cross
            u = (v[0]*a[1]-v[1]*a[0])/cross
            if 0 <= t <= 1 and 0 <= u <= 1:
                xy = p+t*a
                key = tuple(np.round(xy,0))
                if key not in seen:
                    seen.add(key)
                    add("intersection",[tuple(xy)],"finite_segment_intersection",{"line_ids":[la.id,lb.id]})
    diagnostics = {"detection_image_px":[w,h],"pixel_to_pdf_scale":[sx,sy],"noise_min_area_px":3,
                   "line_candidates_before_cap":line_count,"contours_before_cap":len(eligible),
                   "line_cap":1500,"contour_cap":1200,"intersection_line_cap":300,
                   "truncated":line_count>1500 or len(eligible)>1200}
    return geometry,clean,diagnostics


def find_regions(binary, runs, width, height):
    regions = []
    anchors = [r for r in runs if r.raw_text.strip() in ("Масса","Масштаб","Разраб.","Н.контр.") and r.bbox[1] > height*.55]
    if len(anchors) >= 2:
        top = min(r.bbox[1] for r in anchors)-8
        left = min(r.bbox[0] for r in anchors)-10
        regions.append(Region(id="ROI001",type="title_block",bbox=(max(0,left),top,width,height),
                              method="bottom_title_label_cluster",evidence=[r.id for r in anchors]))
    h,w = binary.shape
    mask = binary.copy()
    # Remove sheet border and title block on a detection copy only.
    mask[:round(h*.035),:] = 0
    mask[round(h*.97):,:] = 0
    mask[:,:round(w*.065)] = 0
    mask[:,round(w*.975):] = 0
    for r in regions:
        x0,y0,x1,y1 = r.bbox
        mask[int(y0/height*h):int(y1/height*h),int(x0/width*w):int(x1/width*w)] = 0
    small = cv2.resize(mask,(max(1,w//3),max(1,h//3)),interpolation=cv2.INTER_AREA)
    # Coarse grouping only, never used for primitives or text.
    grouped = cv2.dilate(small,np.ones((9,9),np.uint8))
    contours,_ = cv2.findContours(grouped,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted([b for b in boxes if b[2]*b[3] > small.size*.004],key=lambda b:-b[2]*b[3])[:16]
    for x,y,bw,bh in boxes:
        box = (x/small.shape[1]*width,y/small.shape[0]*height,
               min(width,(x+bw)/small.shape[1]*width),min(height,(y+bh)/small.shape[0]*height))
        evidence = [r.id for r in runs if box[0]<=r.bbox[0]<=box[2] and box[1]<=r.bbox[1]<=box[3]
                    and ("-" in r.raw_text or ":" in r.raw_text)]
        regions.append(Region(id=f"ROI{len(regions)+1:03d}",type="unknown",bbox=box,
            method="coarse_connected_ink_groups",evidence=evidence))
    return regions
