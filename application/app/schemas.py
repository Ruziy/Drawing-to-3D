from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BBox = tuple[float, float, float, float]
Point = tuple[float, float]
Point3 = tuple[float, float, float]
Positive = Annotated[float, Field(gt=0, le=10000, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TextRun(StrictModel):
    id: str
    raw_text: str
    bbox: BBox
    block_id: str
    font: str
    size: float
    direction: Point
    characters: list[dict] = Field(default_factory=list)
    source: Literal["pdf_text", "ocr_tesseract"] = "pdf_text"
    confidence: float | None = None
    recognition: dict = Field(default_factory=dict)


class Dimension(StrictModel):
    id: str
    raw_text: str
    normalized_text: str
    type: Literal["diameter", "radius", "linear", "angle", "thread", "chamfer", "roughness", "tolerance"]
    nominal_value: float
    unit: Literal["mm", "deg", "um"]
    upper_deviation: float | None = None
    lower_deviation: float | None = None
    fit: str | None = None
    pitch: float | None = None
    angle: float | None = None
    source: Literal["pdf_text", "ocr_tesseract"] = "pdf_text"
    bbox: BBox
    text_run_ids: list[str]
    direction: Point = (1, 0)
    associated_geometry_id: str | None = None
    confidence: float | None = None
    alternatives: list[dict] = Field(default_factory=list)
    associations: list[dict] = Field(default_factory=list)
    quantity: int | None = None
    status: Literal["parsed", "needs_review"] = "parsed"
    warnings: list[str] = Field(default_factory=list)


class GeometryElement(StrictModel):
    id: str
    type: str
    coordinates: list[Point]
    bbox: BBox
    source: Literal["vector", "raster"]
    method: str
    line_class: str = "unknown"
    confidence: float | None = None
    properties: dict = Field(default_factory=dict)


class Region(StrictModel):
    id: str
    type: Literal["unknown", "main_view", "additional_view", "section", "detail", "title_block"]
    bbox: BBox
    method: str
    status: str = "candidate"
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)
    crop_url: str | None = None


class PageAnalysis(StrictModel):
    page: int
    width: float
    height: float
    width_mm: float
    height_mm: float
    orientation: str
    original_rotation: int
    has_text_layer: bool
    content_type: str
    coordinate_system: dict
    images: dict
    embedded_images: list[dict]
    text_blocks: list[dict]
    text_runs: list[TextRun]
    dimensions: list[Dimension]
    geometry: dict[str, list[GeometryElement]]
    views: list[Region]
    ambiguous_elements: list[dict]
    warnings: list[str]
    diagnostics: dict
    ocr: dict = Field(default_factory=dict)
    layout: dict = Field(default_factory=dict)


class DrawingAnalysis(StrictModel):
    schema_version: str = "2.0.0"
    drawing_id: str
    filename: str
    sha256: str
    created_at: str
    units: str = "mm"
    units_status: str = "assumed_engineering_convention_not_verified"
    coordinate_unit: str = "pdf_point"
    metadata: dict
    pages: list[PageAnalysis]
    warnings: list[str]
    statistics: dict


class ProfileSegment(StrictModel):
    kind: Literal['line','arc']
    points: list[Point] = Field(min_length=2,max_length=3)

    @model_validator(mode='after')
    def finite_curve(self):
        if len(self.points)!=(2 if self.kind=='line' else 3):raise ValueError('Line/arc requires two/three ordered points')
        if any(abs(v)>10000 for p in self.points for v in p):raise ValueError('Curve coordinates exceed bounds')
        return self


class Operation(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,39}$")
    type: Literal["cylinder", "box", "revolve", "extrude", "cut", "hole", "fuse", "circular_pattern", "chamfer", "fillet"]
    radius: Positive | None = None
    height: Positive | None = None
    length: Positive | None = None
    width: Positive | None = None
    origin: Point3 = (0, 0, 0)
    direction: Point3 = (0, 0, 1)
    count: int = Field(default=1, ge=1, le=36)
    profile: list[Point] = Field(default_factory=list, max_length=200)
    profile_segments: list[ProfileSegment] = Field(default_factory=list,max_length=200)
    angle: float = Field(default=360, gt=0, le=360)
    base: str | None = None
    tool: str | None = None
    edges: list[int] = Field(default_factory=list, max_length=100)
    edge_selector: dict | None = None
    distance: Positive | None = None

    @model_validator(mode="after")
    def validate_operation(self):
        required = {
            "cylinder": ("radius", "height"), "box": ("length", "width", "height"),
            "hole": ("base", "radius", "height"), "cut": ("base", "tool"),
            "fuse": ("base", "tool"), "extrude": ("height",), "revolve": (),
            "circular_pattern": ("base", "tool"),
            "chamfer": ("base", "distance"), "fillet": ("base", "radius"),
        }
        for name in required[self.type]:
            if getattr(self, name) is None:
                raise ValueError(f"{self.type}: требуется {name}")
        if sum(v*v for v in self.direction)<1e-12:
            raise ValueError('Направление оси не может быть нулевым')
        if self.type in ("chamfer","fillet") and ((not self.edges and not self.edge_selector) or any(i<1 for i in self.edges)):
            raise ValueError("Нужны явные номера рёбер, начиная с 1")
        if self.edge_selector:
            if set(self.edge_selector)!={'point','tolerance_mm','kind'} or self.edge_selector['kind']!='nearest':raise ValueError('Unknown geometric edge selector')
            point=self.edge_selector['point'];tolerance=self.edge_selector['tolerance_mm']
            if len(point)!=3 or not all(isinstance(v,(int,float)) and abs(v)<=10000 for v in point):raise ValueError('Invalid edge selection point')
            if not isinstance(tolerance,(int,float)) or not 0<tolerance<=100:raise ValueError('Invalid edge selection tolerance')
        if self.type in ("extrude", "revolve"):
            if len(self.profile) < 3 or len(set(self.profile)) < 3:
                raise ValueError("Профиль должен содержать минимум три разные точки")
            area = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(self.profile, self.profile[1:]+self.profile[:1]))
            if abs(area) < 1e-8:
                raise ValueError("Профиль имеет нулевую площадь")
            if self.type == "revolve" and any(p[0] < 0 for p in self.profile):
                raise ValueError("Радиус профиля вращения не может быть отрицательным")
        if any(abs(v) > 10000 for v in self.origin) or any(abs(v) > 10000 for p in self.profile for v in p):
            raise ValueError("Координаты вне диапазона +/-10000 мм")
        return self


class CADIntent(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    units: Literal["mm"] = "mm"
    provenance: Literal["manual", "prepared_demo", "automatic_pdf"] = "manual"
    reconstruction_status: Literal["complete", "full_geometry", "partial_geometry", "preliminary_geometry", "manual"] = "manual"
    evidence: dict = Field(default_factory=dict)
    validation_expectations: list[dict] = Field(default_factory=list)
    unresolved_features: list[dict] = Field(default_factory=list)
    description: str = Field(default="Параметры заданы вручную", max_length=1000)
    operations: list[Operation] = Field(min_length=1, max_length=50)
    result: str

    @model_validator(mode="after")
    def check_graph(self):
        seen = set()
        for op in self.operations:
            if op.id in seen:
                raise ValueError("ID операций должны быть уникальны")
            for ref in (op.base, op.tool):
                if ref and ref not in seen:
                    raise ValueError(f"Неизвестная или последующая операция: {ref}")
            seen.add(op.id)
        if self.result not in seen:
            raise ValueError("Результат должен ссылаться на существующую операцию")
        return self
