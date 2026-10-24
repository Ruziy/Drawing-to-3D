"""Local view frames and explicit coordinate correspondences, in PDF points."""
import numpy as np


def analyze_views(page):
    axes=[]
    for d in page['dimensions']:
        if d['type'] not in ('diameter','thread') or d['nominal_value']<=0:continue
        for association in d['associations']:
            ends=np.array(association['endpoints_pdf'],float)
            orientation='horizontal' if association['orientation']=='vertical' else 'vertical'
            index=1 if orientation=='horizontal' else 0
            coordinate=float(np.mean(ends[:,index]));scale=association['scale_pt_per_mm']
            axis=next((a for a in axes if a['orientation']==orientation and abs(a['coordinate_pdf']-coordinate)<.8 and abs(a['scale_pt_per_mm']/scale-1)<.018),None)
            if axis is None:
                axis={'id':f'AXIS{len(axes)+1:03d}','orientation':orientation,'coordinate_pdf':coordinate,
                      'scale_pt_per_mm':scale,'dimension_ids':[],'source_endpoints_pdf':[],'status':'candidate'}
                axes.append(axis)
            if d['id'] not in axis['dimension_ids']:
                axis['dimension_ids'].append(d['id']);axis['source_endpoints_pdf'].append(association['endpoints_pdf'])
    for axis in axes:
        if len(axis['dimension_ids'])>=2:axis['status']='dimension_consensus'
    frames=[]
    for profile in page['layout']['profile_candidates']:
        if profile['contour_depth']!=0 or profile['area_pt2']<100:continue
        b=profile['bbox']
        if b[2]-b[0]>page['width']*.85 or b[3]-b[1]>page['height']*.6:continue
        local=[]
        for axis in axes:
            index=1 if axis['orientation']=='horizontal' else 0
            if abs(axis['coordinate_pdf']-(b[index]+b[index+2])/2)<2.5:local.append(axis['id'])
        frames.append({'id':f'VIEW{len(frames)+1:03d}','bbox':b,'source_profile_id':profile['id'],
                       'axis_ids':local,'coordinate_unit':'pdf_point','origin_pdf':[b[0],b[1]],
                       'status':'physical_region_candidate','scale_policy':'local_dimension_consensus_only'})
    links=[]
    for index,a in enumerate(frames):
        for b in frames[index+1:]:
            ba,bb=a['bbox'],b['bbox']
            shared=None
            if max(abs(ba[0]-bb[0]),abs(ba[2]-bb[2]))<2.5 and (ba[3]<bb[1] or bb[3]<ba[1]):shared='x'
            if max(abs(ba[1]-bb[1]),abs(ba[3]-bb[3]))<2.5 and (ba[2]<bb[0] or bb[2]<ba[0]):shared='y'
            if shared:links.append({'views':[a['id'],b['id']],'shared_sheet_axis':shared,
                                    'method':'orthographic_extrema_alignment','status':'candidate_requires_dimension_agreement'})
    intersections=[]
    for a in axes:
        if a['orientation']!='horizontal' or a['status']!='dimension_consensus':continue
        for b in axes:
            if b['orientation']!='vertical' or abs(a['scale_pt_per_mm']/b['scale_pt_per_mm']-1)>.025:continue
            point=[b['coordinate_pdf'],a['coordinate_pdf']]
            for frame in frames:
                box=frame['bbox']
                if box[0]+3<point[0]<box[2]-3 and box[1]+3<point[1]<box[3]-3:
                    intersections.append({'axes':[a['id'],b['id']],'point_pdf':point,'view_id':frame['id'],
                                          'angle_deg':90,'status':'orthogonal_axis_hypothesis_not_a_confirmed_solid',
                                          'scale_pt_per_mm':a['scale_pt_per_mm']})
    labels=[{'text':r['raw_text'],'bbox':r['bbox']} for r in page.get('text_runs',[])
            if r['raw_text'].strip() in ('А-А','A-A','Б-Б','В-В','А–А','Б–Б')]
    return {'axes':axes,'frames':frames,'correspondences':links,'orthogonal_axis_intersections':intersections,
            'section_labels':labels,'warning':'Совпадение проекций — гипотеза; точные размеры не заменяются общим масштабом листа.'}
