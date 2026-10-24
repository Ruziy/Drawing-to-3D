"""Common part coordinates with explicit unregistered/estimated view relations."""
import numpy as np


class SpatialPartFrame:
    def __init__(self,document,plan):
        self.document=document;self.plan=plan;self.views=[]
        for index,body in enumerate(plan.bodies):
            page=next(p for p in document['pages'] if p['page']==body.page)
            box=np.array(body.bbox)*([page['width'],page['height']]*2);span=box[2:]-box[:2]
            if body.family=='prismatic':
                matrix=[[body.width.value/span[0],0,0,-box[0]*body.width.value/span[0]],
                        [0,body.length.value/span[1],0,-box[1]*body.length.value/span[1]], [0,0,0,0]]
            elif body.axis=='horizontal':
                matrix=[[0,body.width.value/span[1],0,-(box[1]+box[3])/2*body.width.value/span[1]], [0,0,0,0],
                        [body.length.value/span[0],0,0,-box[0]*body.length.value/span[0]]]
            else:
                matrix=[[body.width.value/span[0],0,0,-(box[0]+box[2])/2*body.width.value/span[0]], [0,0,0,0],
                        [0,body.length.value/span[1],0,-box[1]*body.length.value/span[1]]]
            self.views.append({'id':f'body_{index}_page_{body.page}','page':body.page,'bbox_pdf':box.tolist(),
                'role':'main' if index==0 else 'alternative_view_same_part','source_to_part_affine':matrix,
                'status':'estimated_registration','coordinate_unit':'mm','source_coordinate_unit':'pdf_point',
                'normal_coordinate':'unobserved','family':body.family})
        # Correspondence is a hypothesis until coordinates are registered. Never
        # assert that identical numbers alone establish a multi-page transform.
        self.relations=[]
        for i,a in enumerate(document['pages']):
            for b in document['pages'][i+1:]:
                da={round(d['nominal_value'],3) for d in a['dimensions'] if d['type'] in ('linear','diameter') and d['nominal_value']>=5}
                db={round(d['nominal_value'],3) for d in b['dimensions'] if d['type'] in ('linear','diameter') and d['nominal_value']>=5}
                self.relations.append({'pages':[a['page'],b['page']],'shared_nominal_candidates':sorted(da&db),
                    'status':'candidate_correspondence_not_verified','reason':'Cross-page identities require corresponding measured features; common numbers alone are insufficient'})

    def project(self,view_id,point):
        view=next(v for v in self.views if v['id']==view_id)
        return (np.array(view['source_to_part_affine'])@np.array([point[0],point[1],0,1])).tolist()

    def serialize(self):
        return {'part_frame':'right_handed_CAD_mm','views':self.views,'cross_page_relations':self.relations,
                'unregistered_pages':[p['page'] for p in self.document['pages'] if p['page'] not in {v['page'] for v in self.views}]}
