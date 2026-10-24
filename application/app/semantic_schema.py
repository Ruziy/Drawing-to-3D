"""Bounded declarative part graph. Model output is data, never executable CAD code."""
from typing import Literal,Annotated
from pydantic import Field, model_validator
from .schemas import StrictModel

Normalized = Annotated[float,Field(ge=0,le=1)]
NormalizedPoint = tuple[Normalized, Normalized]
NormalizedBox = tuple[Normalized, Normalized, Normalized, Normalized]


class SemanticView(StrictModel):
    id: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_]{0,29}$')
    bbox: NormalizedBox
    role: Literal['main','side','end','section','detail','unknown']
    axis: Literal['horizontal','vertical','normal','unknown'] = 'unknown'
    family: Literal['rotational','prismatic','angular','combined','unknown'] = 'unknown'
    related_views: list[str] = Field(default_factory=list,max_length=8)
    shortened: bool = False
    description: str = Field(default='',max_length=300)

    @model_validator(mode='after')
    def valid_box(self):
        a,b,c,d=self.bbox
        if not (0<=a<c<=1 and 0<=b<d<=1):raise ValueError('View bbox requires normalized page coordinates [0,1]')
        return self


class DrawingSemantics(StrictModel):
    part_description: str = Field(max_length=500)
    views: list[SemanticView] = Field(min_length=1,max_length=8)
    expected_features: list[str] = Field(default_factory=list,max_length=20)
    missing_information: list[str] = Field(default_factory=list,max_length=12)


class SemanticParameter(StrictModel):
    role: Literal['length','width','height','thickness','diameter','inner_diameter','depth',
                  'radius','pitch_diameter','angle','offset_x','offset_y','offset_z','count']
    dimension_id: str | None = None
    value: float | None = Field(default=None,gt=0,le=10000)
    evidence: Literal['ocr_dimension','image_dimension','geometric_ratio','visual_estimate'] = 'ocr_dimension'
    source_bbox: list[float] = Field(default_factory=list,max_length=4)
    raw_text: str = Field(default='',max_length=100)


class SemanticObject(StrictModel):
    id: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_]{0,29}$')
    view_id: str
    kind: Literal['cylinder','tube','box','extrusion','revolution','hole','pocket','pattern','chamfer','fillet']
    bbox: NormalizedBox
    axis: Literal['horizontal','vertical','normal'] = 'normal'
    parameters: list[SemanticParameter] = Field(min_length=1,max_length=12)
    profile_points: list[NormalizedPoint] = Field(default_factory=list,max_length=80)
    center: NormalizedPoint | None = None
    description: str = Field(default='',max_length=300)

    @model_validator(mode='before')
    @classmethod
    def parameter_mapping(cls,value):
        # Wire schema uses fixed named parameters. Internally the stable list
        # representation preserves provenance and avoids dict-order semantics.
        if isinstance(value,dict) and isinstance(value.get('parameters'),dict):
            value={**value,'parameters':[{'role':role,**parameter} for role,parameter in value['parameters'].items()]}
        return value

    @model_validator(mode='after')
    def bounded(self):
        a,b,c,d=self.bbox
        if not (0<=a<c<=1 and 0<=b<d<=1):raise ValueError('Object bbox must be normalized and ordered')
        if any(not 0<=v<=1 for p in self.profile_points for v in p):raise ValueError('Profile point outside page')
        if self.center and any(not 0<=v<=1 for v in self.center):raise ValueError('Object center outside page')
        required={'cylinder':{'length','diameter'},'tube':{'length','diameter','inner_diameter'},
                  'box':{'width','length','thickness'},'extrusion':{'width','length','thickness'},
                  'revolution':{'length','diameter'},'hole':{'diameter','depth'},'pocket':{'depth'},
                  'pattern':{'diameter','depth','count','pitch_diameter'},'fillet':{'radius'},'chamfer':{'radius'}}
        roles={p.role for p in self.parameters}
        missing=required[self.kind]-roles
        if missing:raise ValueError(f'{self.id}: numeric parameters missing: {sorted(missing)}; prose description is not executable')
        if len(roles)!=len(self.parameters):raise ValueError('Duplicate parameter role')
        return self


class PartGraph(StrictModel):
    objects: list[SemanticObject] = Field(min_length=1,max_length=16)
    view_relations: list[str] = Field(default_factory=list,max_length=12)
    uncertainties: list[str] = Field(default_factory=list,max_length=20)

    @model_validator(mode='after')
    def unique(self):
        if len({o.id for o in self.objects})!=len(self.objects):raise ValueError('Duplicate new object ID')
        return self


class ConstructionStep(StrictModel):
    object_id: str
    action: Literal['add','subtract','pattern','chamfer','fillet']
    base_id: str | None = None
    description: str = Field(default='',max_length=200)


class ConstructionPlan(StrictModel):
    steps: list[ConstructionStep] = Field(min_length=1,max_length=20)
    result_id: str
    missing_features: list[str] = Field(default_factory=list,max_length=20)

    @model_validator(mode='after')
    def executable_graph(self):
        seen=set()
        for step in self.steps:
            if step.object_id in seen:raise ValueError('Object constructed twice')
            if step.base_id and step.base_id not in seen:raise ValueError('Unknown/future base_id '+step.base_id)
            if step.action!='add' and not step.base_id:raise ValueError('Every subtract/pattern/chamfer/fillet MUST name a previous base_id')
            seen.add(step.object_id)
        if self.result_id not in seen:raise ValueError('Result object is not constructed')
        return self


def graph_schema(page,semantics):
    """Constrain existing pointers; new objects use their own declaration namespace."""
    schema=PartGraph.model_json_schema()
    dimensions=[d['id'] for d in page['dimensions'] if d['type'] not in ('roughness','angle','tolerance')]+page.get('_shared_dimension_ids',[])
    schema['$defs']['SemanticParameter']['properties']['dimension_id']={'anyOf':[{'type':'string','enum':dimensions},{'type':'null'}]} if dimensions else {'type':'null'}
    schema['$defs']['SemanticObject']['properties']['view_id']={'type':'string','enum':[v.id for v in semantics.views]}
    schema['properties']['objects']['maxItems']=6
    schema['$defs']['SemanticObject']['required']=['id','view_id','kind','bbox','axis','parameters']
    schema['$defs']['SemanticParameter']['required']=['role','dimension_id','value','evidence','source_bbox','raw_text']
    # A discriminated construction grammar makes missing numeric roles impossible
    # during decoding. Existing Pydantic validation still independently checks it.
    from copy import deepcopy
    base=deepcopy(schema['$defs']['SemanticObject']);variants=[]
    roles={'tube':['diameter','length','inner_diameter'],'cylinder':['diameter','length'],
           'box':['width','length','thickness'],'extrusion':['width','length','thickness'],
           'revolution':['diameter','length'],'hole':['diameter','depth'],'pocket':['depth'],
           'pattern':['diameter','depth','count','pitch_diameter'],'chamfer':['radius'],'fillet':['radius']}
    for kind,names in roles.items():
        variant=deepcopy(base);variant['properties']['kind']={'const':kind,'type':'string'}
        if kind not in ('extrusion','revolution','pocket'):
            variant['properties']['profile_points']={'type':'array','items':{},'maxItems':0}
        parameters={}
        for role in names:
            parameter=deepcopy(schema['$defs']['SemanticParameter']);parameter['properties'].pop('role')
            parameter['required']=[key for key in parameter['required'] if key!='role']
            parameters[role]=parameter
        variant['properties']['parameters']={'type':'object','properties':parameters,'required':names,'additionalProperties':False}
        variants.append(variant)
    schema['$defs']['SemanticObject']={'oneOf':variants}
    return schema


def construction_schema(graph):
    schema=ConstructionPlan.model_json_schema();ids=[o.id for o in graph.objects]
    schema['$defs']['ConstructionStep']['properties']['object_id']={'type':'string','enum':ids}
    schema['$defs']['ConstructionStep']['properties']['base_id']={'anyOf':[{'type':'string','enum':ids},{'type':'null'}]}
    schema['properties']['result_id']={'type':'string','enum':ids}
    from copy import deepcopy
    base=deepcopy(schema['$defs']['ConstructionStep']);variants=[]
    for action in ('add','subtract','pattern','chamfer','fillet'):
        allowed=[o.id for o in graph.objects if (o.kind not in ('hole','pocket','pattern','chamfer','fillet') if action=='add' else
            o.kind=='pattern' if action=='pattern' else o.kind==action if action in ('chamfer','fillet') else o.kind not in ('pattern','chamfer','fillet'))]
        if not allowed:continue
        variant=deepcopy(base);variant['properties']['action']={'const':action,'type':'string'}
        variant['properties']['object_id']={'type':'string','enum':allowed}
        if action!='add':variant['properties']['base_id']={'type':'string','enum':ids}
        variant['required']=['object_id','action','base_id'];variants.append(variant)
    schema['$defs']['ConstructionStep']={'oneOf':variants}
    return schema
