"""Untrusted vision interpretation. No executable code or unconstrained CAD JSON."""
from typing import Literal
from pydantic import Field, model_validator
from .schemas import StrictModel, BBox, Point


class SourceReference(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    page: int = Field(ge=1, le=20)
    view_id: str = Field(default='', max_length=80)
    bbox_pdf: BBox
    dimension_ids: list[str] = Field(default_factory=list, max_length=200)
    geometry_ids: list[str] = Field(default_factory=list, max_length=200)

    @model_validator(mode='after')
    def ordered(self):
        x0,y0,x1,y1 = self.bbox_pdf
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError('Source bbox must be positive, ordered PDF points')
        return self


class ViewInterpretation(StrictModel):
    id: str = Field(max_length=80)
    role: Literal['main', 'top', 'side', 'longitudinal_section', 'cross_section', 'local_section', 'detail', 'unknown']
    source_ref: str
    related_view_ids: list[str] = Field(default_factory=list, max_length=12)


class DimensionInterpretation(StrictModel):
    dimension_id: str
    source_ref: str
    value: float = Field(gt=0, le=10000)
    unit: Literal['mm', 'deg', 'um'] = 'mm'
    role: str = Field(default='', max_length=180)
    proposed_raw_text: str | None = Field(default=None, max_length=120)


class GeometricConstraint(StrictModel):
    type: Literal['coaxial', 'parallel', 'perpendicular', 'symmetric', 'equal', 'dimension_binding', 'view_correspondence']
    source_refs: list[str] = Field(min_length=1, max_length=10)
    dimension_id: str | None = None
    geometry_id: str | None = None
    description: str = Field(default='', max_length=400)


class FeaturePlan(StrictModel):
    id: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_]{0,39}$')
    operation: Literal['Sketch', 'Pad', 'Extrusion', 'Revolution', 'Pocket', 'Hole', 'Cut', 'Fuse', 'Fillet', 'Chamfer']
    source_refs: list[str] = Field(min_length=1, max_length=12)
    profile_id: str | None = None
    # Optional ordered points are sheet coordinates, NOT fabricated mm coordinates.
    profile_points_pdf: list[Point] = Field(default_factory=list, max_length=100)
    axis: Literal['horizontal', 'vertical'] = 'horizontal'
    axis_dimension_id: str | None = None
    dimension_ids: list[str] = Field(default_factory=list, max_length=30)
    description: str = Field(default='', max_length=500)


class UncertainElement(StrictModel):
    description: str = Field(max_length=600)
    source_refs: list[str] = Field(default_factory=list, max_length=12)


class LLMInterpretation(StrictModel):
    part_type: str = Field(max_length=120)
    coordinate_system: Literal['pdf_point_top_left_unrotated']
    views: list[ViewInterpretation] = Field(max_length=16)
    dimensions: list[DimensionInterpretation] = Field(max_length=80)
    geometric_constraints: list[GeometricConstraint] = Field(max_length=40)
    features: list[FeaturePlan] = Field(max_length=16)
    uncertain_elements: list[UncertainElement] = Field(max_length=30)
    source_references: list[SourceReference] = Field(max_length=40)
    validation_notes: list[str] = Field(max_length=30)

    @model_validator(mode='after')
    def references_exist(self):
        refs = [r.id for r in self.source_references]
        if len(set(refs)) != len(refs):
            raise ValueError('Duplicate source reference')
        for item in [*self.views, *self.dimensions, *self.geometric_constraints, *self.features, *self.uncertain_elements]:
            needed = [item.source_ref] if hasattr(item, 'source_ref') else item.source_refs
            if any(r not in refs for r in needed):
                raise ValueError('Unknown source reference')
        view_ids = {v.id for v in self.views}
        if any(i not in view_ids for v in self.views for i in v.related_view_ids):
            raise ValueError('Unknown related view')
        if len({f.id for f in self.features}) != len(self.features):
            raise ValueError('Duplicate feature ID')
        return self
