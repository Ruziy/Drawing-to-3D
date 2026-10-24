"""Endpoint graph of physical strokes. Only sub-stroke gaps may be snapped.

This graph is diagnostic and supplies closed centreline faces independently of
the raster outline of a thick stroke. It never closes arbitrary open chains.
"""
import math
import cv2
import numpy as np


def stroke_graph(geometry, page_width, page_height, tolerance=.32):
    segments = []
    for group in geometry.values():
        for g in group:
            if g.source != 'vector' or g.line_class != 'physical_boundary_candidate':
                continue
            if g.properties.get('fill') is not None:
                continue
            if g.type == 'circle':
                center = np.array(g.properties['center']); radius = g.properties['radius_pt']
                angles = np.linspace(0, 2 * math.pi, 65)
                points = center + np.c_[np.cos(angles), np.sin(angles)] * radius
            else:
                points = np.array(g.coordinates)
            for a, b in zip(points, points[1:]):
                if np.linalg.norm(b-a) < .025: continue
                midpoint = (a+b)/2
                if not (.08*page_width < midpoint[0] < .97*page_width and .045*page_height < midpoint[1] < .70*page_height): continue
                segments.append((a, b, g.id))
    # Spatial hash avoids quadratic endpoint matching on outlined fonts.
    nodes=[]; buckets={}; edges=[]; adjacency={}; snaps=[]; seen=set()
    def vertex(p):
        key=tuple(np.floor(p/tolerance).astype(int))
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for index in buckets.get((key[0]+dx,key[1]+dy),[]):
                    gap=float(np.linalg.norm(np.array(nodes[index])-p))
                    if gap <= tolerance:
                        if gap>.025: snaps.append({'node':index,'gap_pt':gap,'method':'sub_stroke_endpoint_coincidence'})
                        return index
        index=len(nodes); nodes.append(p.tolist()); buckets.setdefault(key,[]).append(index); return index
    for a,b,source in segments:
        u,v=vertex(a),vertex(b)
        if u==v or tuple(sorted((u,v))) in seen: continue
        seen.add(tuple(sorted((u,v)))); edges.append({'vertices':[u,v],'geometry_id':source,'role':'physical_boundary'})
        adjacency.setdefault(u,set()).add(v);adjacency.setdefault(v,set()).add(u)
    ordered={u:sorted(vs,key=lambda v:math.atan2(nodes[v][1]-nodes[u][1],nodes[v][0]-nodes[u][0])) for u,vs in adjacency.items()}
    used=set();faces=[]
    for edge in edges:
        for start in (tuple(edge['vertices']),tuple(reversed(edge['vertices']))):
            if start in used:continue
            walk=[];current=start;local=set()
            while current not in local and len(walk)<=len(edges)*2:
                local.add(current);used.add(current);a,b=current;walk.append(a)
                neighbors=ordered[b];current=(b,neighbors[(neighbors.index(a)-1)%len(neighbors)])
            if current!=start or len(walk)<3 or len(set(walk))!=len(walk):continue
            polygon=np.array([nodes[i] for i in walk],np.float32)
            area=float(cv2.contourArea(polygon,oriented=True))
            if area<40:continue
            polygon=cv2.approxPolyDP(polygon,.10,True)[:,0,:]
            if len(polygon)>200:continue
            faces.append({'id':f'GRAPH{len(faces)+1:03d}','polygon_pdf':polygon.tolist(),'area_pt2':area,
                          'bbox':[*polygon.min(axis=0).tolist(),*polygon.max(axis=0).tolist()],
                          'type':'graph_face','contour_depth':1,'hatch_line_count':0,
                          'method':'physical_stroke_endpoint_graph','source_vertex_indices':walk})
    return {'coordinate_unit':'pdf_point','nodes':nodes,'edges':edges,'endpoint_snaps':snaps,
            'open_vertices':[i for i,vs in adjacency.items() if len(vs)==1],
            'faces':faces,'snap_tolerance_pt':tolerance,'arbitrary_gap_closure':False}


def mirror_error(first, second, center):
    """Symmetric sampled boundary distance in points, not area similarity."""
    def samples(p):
        return np.vstack([a+(b-a)*t for a,b in zip(p,np.roll(p,-1,axis=0)) for t in np.linspace(0,1,5)])
    a=samples(np.array(first,float));b=samples(np.array(second,float));a[:,1]=2*center-a[:,1]
    distances=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
    return max(float(np.percentile(distances.min(axis=0),90)),float(np.percentile(distances.min(axis=1),90)))
