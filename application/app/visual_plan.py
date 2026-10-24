"""Bounded image-localized construction hypotheses; never executable code."""
from typing import Literal,Annotated
from pydantic import BaseModel,ConfigDict,Field,model_validator,PrivateAttr


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)


Normalized=Annotated[float,Field(ge=0,le=1)]
NormalizedBox=tuple[Normalized,Normalized,Normalized,Normalized]


class Measure(Strict):
    value:float=Field(gt=0,le=10000)
    dimension_id:str|None=None
    estimated:bool=True


class VisualBody(Strict):
    _observed_profile_id:str|None=PrivateAttr(default=None)
    page:int=Field(ge=1,le=20)
    bbox:NormalizedBox
    family:Literal['rotational','prismatic']
    axis:Literal['horizontal','vertical']='horizontal'
    length:Measure
    width:Measure
    depth:Measure
    inner_diameter:Measure|None=None
    origin:tuple[float,float,float]=(0,0,0)
    direction:tuple[float,float,float]=(0,0,1)
    reason:str=Field(max_length=700)

    @model_validator(mode='after')
    def ordered(self):
        x,y,u,v=self.bbox
        if not 0<=x<u<=1 or not 0<=y<v<=1:raise ValueError('bbox must be ordered complete-page coordinates [0,1]')
        if self.inner_diameter and self.inner_diameter.value>=self.width.value:
            raise ValueError('Inner diameter must be smaller than outer diameter')
        if sum(v*v for v in self.direction)<1e-8 or any(abs(v)>10000 for v in self.origin):raise ValueError('Invalid component frame')
        return self


class CompletionFeature(Strict):
    kind:Literal['hole','slot','hole_pattern']
    page:int=Field(ge=1,le=20)
    bbox:NormalizedBox
    origin:tuple[float,float,float]
    direction:tuple[float,float,float]=(0,0,1)
    diameter:float=Field(gt=0,le=10000)
    depth:float=Field(gt=0,le=10000)
    width:float|None=Field(default=None,gt=0,le=10000)
    count:int=Field(default=1,ge=1,le=36)
    pitch_radius:float=Field(default=0,ge=0,le=10000)
    angle_degrees:float=0
    dimension_ids:list[str]=Field(default_factory=list,max_length=8)
    estimated:bool=True
    reason:str=Field(max_length=400)


class VisualPlan(Strict):
    part_description:str=Field(max_length=700)
    bodies:list[VisualBody]=Field(min_length=1,max_length=4)
    composition:Literal['single','fused_components']='single'
    features:list[CompletionFeature]=Field(default_factory=list,max_length=12)
    uncertainties:list[str]=Field(default_factory=list,max_length=20)
