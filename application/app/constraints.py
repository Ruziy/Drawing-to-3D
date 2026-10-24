"""Topology-gated linear chains. Explicit values are not replaced by image estimates."""
import numpy as np


def solve_chain(observations,origin,tolerance_pt=1.8):
    positions=[origin]
    def node(value):
        for i,p in enumerate(positions):
            if abs(value-p)<=tolerance_pt:return i
        positions.append(value);return len(positions)-1
    edges=[]
    for observation in observations:
        a,b=sorted(observation['endpoints'])
        i,j=node(a),node(b)
        if i!=j:edges.append((i,j,observation['value'],observation['dimension_id']))
    if not edges:return [],[],[]
    # Only the graph component connected to the actual profile start is grounded.
    connected={0}
    while True:
        after=connected|{j for i,j,_,_ in edges if i in connected}|{i for i,j,_,_ in edges if j in connected}
        if after==connected:break
        connected=after
    use=[e for e in edges if e[0] in connected and e[1] in connected]
    if not use:return [],[],[]
    ids=sorted(connected);index={v:i for i,v in enumerate(ids)}
    A=np.zeros((len(use)+1,len(ids)));y=np.zeros(len(use)+1)
    A[0,index[0]]=1
    for row,(a,b,value,_) in enumerate(use,1):
        A[row,index[a]]=-1;A[row,index[b]]=1;y[row]=value
    solution,*_=np.linalg.lstsq(A,y,rcond=None)
    conflicts=[];equations=[]
    for a,b,value,dimension_id in use:
        residual=float(solution[index[b]]-solution[index[a]]-value)
        record={'dimension_id':dimension_id,'from_node':a,'to_node':b,'nominal_mm':value,'residual_mm':residual}
        equations.append(record)
        if abs(residual)>.03:conflicts.append({**record,'reason':'inconsistent_explicit_axial_chain'})
    nodes=[{'pdf_position':positions[i],'position_mm':float(solution[index[i]]),
            'dimension_ids':[d for a,b,_,d in use if i in (a,b)]} for i in ids]
    # Inconsistent equations are reported, never averaged into a trusted profile.
    return ([] if conflicts else nodes),equations,conflicts
