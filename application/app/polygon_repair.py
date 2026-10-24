"""Remove only tiny zero-width raster loops; large topology changes are rejected."""
import numpy as np


def area(points):return abs(sum(np.cross(a,b) for a,b in zip(points,np.roll(points,-1,axis=0)))/2)


def clean_pixel_polygon(points,maximum_loss_fraction=.015):
    points=np.array(points,float);original=area(points);discarded=0.
    def choose(a,b):
        nonlocal discarded
        candidates=sorted((p for p in (a,b) if len(p)>=3),key=area,reverse=True)
        if not candidates:raise ValueError('Raster boundary collapsed')
        loss=area(candidates[-1]) if len(candidates)>1 else 0
        if discarded+loss>max(original,1)*maximum_loss_fraction:raise ValueError('Boundary repair would change substantial material topology')
        discarded+=loss;return candidates[0]
    for _ in range(100):
        changed=False
        for i in range(len(points)):
            for j in range(i+2,len(points)):
                if i==0 and j==len(points)-1:continue
                if np.linalg.norm(points[i]-points[j])<1e-7:
                    points=choose(points[i:j],np.vstack([points[j:],points[:i]]));changed=True;break
                a=points[i];b=points[(i+1)%len(points)];c=points[j];d=points[(j+1)%len(points)]
                u=b-a;v=d-c;den=float(np.cross(u,v))
                if abs(den)<1e-10:continue
                t=float(np.cross(c-a,v)/den);s=float(np.cross(c-a,u)/den)
                if 1e-8<t<1-1e-8 and 1e-8<s<1-1e-8:
                    intersection=a+t*u
                    points=choose(np.vstack([intersection,points[i+1:j+1]]),np.vstack([intersection,points[j+1:],points[:i+1]]))
                    changed=True;break
            if changed:break
        if not changed:return points,{'discarded_loop_area':discarded,'maximum_discard_fraction':maximum_loss_fraction}
    raise ValueError('Too many raster-boundary intersections')
