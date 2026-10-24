"""Deterministic RANSAC-style local scale consensus, one vote per dimension."""
import math
import numpy as np


def robust_view_scale(page,bbox,relative_tolerance=.035):
    observations=[];x0,y0,x1,y1=bbox
    for d in page['dimensions']:
        if d['type'] not in ('linear','diameter','radius') or d['nominal_value']<=0:continue
        for a in d.get('associations',[]):
            points=np.array(a['endpoints_pdf'],float);axis=0 if a['orientation']=='horizontal' else 1
            low,high=(x0,x1) if axis==0 else (y0,y1)
            other=(y0,y1) if axis==0 else (x0,x1)
            ends=sorted(points[:,axis]);position=float(points[:,1-axis].mean())
            if ends[0]<low-8 or ends[1]>high+8 or position<other[0]-90 or position>other[1]+90:continue
            scale=float(a['scale_pt_per_mm'])
            if not math.isfinite(scale) or not .02<scale<500:continue
            observations.append({'dimension_id':d['id'],'scale_pt_per_mm':scale,'association':a,
                                 'axis':axis,'nominal_mm':d['nominal_value']})
    best=[];best_key=(-1,-float('inf'))
    for hypothesis in sorted({round(o['scale_pt_per_mm'],6) for o in observations}):
        votes={}
        for observation in observations:
            residual=abs(observation['scale_pt_per_mm']/hypothesis-1)
            if residual>relative_tolerance:continue
            item={**observation,'relative_residual':residual}
            previous=votes.get(item['dimension_id'])
            if previous is None or residual<previous['relative_residual']:votes[item['dimension_id']]=item
        use=list(votes.values());key=(len(use),-sum(o['relative_residual'] for o in use))
        if key>best_key:best_key=key;best=use
    scale=float(np.median([o['scale_pt_per_mm'] for o in best])) if best else None
    return {'method':'local_dimension_consensus_one_vote_per_nominal','bbox_pdf':list(bbox),
            'scale_pt_per_mm':scale,'inliers':best,'inlier_dimension_count':len(best),
            'rejected_observation_count':len(observations)-len(best),
            'status':'consensus' if len(best)>=2 else 'single_evidence' if best else 'unknown',
            'maximum_relative_residual':max((abs(o['scale_pt_per_mm']/scale-1) for o in best),default=None)}
