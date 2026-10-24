"""Conservative annotation parser. PDF characters are evidence, not CAD intent."""
import re

from .schemas import Dimension, TextRun

N = r"\d+(?:[.,]\d+)?"
SIGNED = rf"[+−-]\s*{N}"
FIT = r"(?:[A-HJ-NP-ZÇa-hj-np-z]{1,2}\d{1,2})"
TOL = rf"(?:\s*±\s*{N}|\s*\(?\s*{SIGNED}(?:\s*{SIGNED})?\s*\)?)?"
PATTERNS = [
    ("chamfer", re.compile(rf"(?<![\w.,])(?P<n>{N})\s*[×xх•]\s*(?P<a>{N})\s*°")),
    ("thread", re.compile(rf"(?<![\w])(?:M|Tr)\s*(?P<n>{N})(?:\s*[×xх]\s*(?P<p>{N}))?(?:\s*[-−]\s*\d+[A-Za-zА-Яа-я]+)?")),
    ("roughness", re.compile(rf"(?<!\w)Ra\s*(?P<n>{N})", re.I)),
    ("diameter", re.compile(rf"[⌀Øø∅]\s*(?P<n>{N})(?P<fit>{FIT})?{TOL}")),
    ("radius", re.compile(rf"(?<!\w)R\s*(?P<n>{N}){TOL}")),
    ("angle", re.compile(rf"(?<![\w.,])(?P<n>{N})\s*°(?:\s*{N}\s*['′])?{TOL}")),
]
LINEAR = re.compile(rf"^(?P<n>{N})(?P<fit>{FIT})?{TOL}\s*\*?$")


def number(value):
    return float(value.replace(",", ".").replace("−", "-").replace(" ", ""))


def normalize(text):
    # Do not silently interpret legacy-font Ç or Å as diameter / degrees.
    return text.replace("−", "-").replace("⌀", "Ø").replace("ø", "Ø").replace("∅", "Ø").replace("х", "×").replace("x", "×").replace(",", ".")


def parser_text(run):
    text=run.recognition.get('structured_text',run.raw_text)
    text=re.sub(r'М(?=\s*\d)','M',text)
    if run.source=='pdf_text':
        # Known CAD font character mappings, retained as explicit normalization evidence.
        text=re.sub(r'Ç(?=\s*\d)','Ø',text).replace('Å','°')
    else:
        text=re.sub(r'[ФфΦ](?=\s*\d)','Ø',text)
        # An isolated first numeric word can still be a nominal when the tolerance
        # is garbled. The suffix is retained and flagged, never repaired numerically.
        if not run.recognition.get('structured_text') and not re.search(r'[°×xх]|Ra|R\d|M\d',text):
            m=re.match(r'^\s*(\d+(?:[.,]\d+)?)(?:\s+[^\d].*)?$',text)
            if m:text=m[1]
    return text


def tolerance(text):
    pm = re.search(rf"±\s*({N})", text)
    if pm:
        v = number(pm[1])
        return v, -v, []
    # Limit deviations to text following the nominal / fit, never thread classes.
    signed = [number(m.group()) for m in re.finditer(SIGNED, text)]
    if len(signed) == 2:
        return max(signed), min(signed), []
    if len(signed) == 1:
        v = signed[0]
        return max(v, 0), min(v, 0), []
    return None, None, []


def union_bbox(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def match_bbox(run, start, end):
    chars = run.characters[start:end]
    return union_bbox([c["bbox"] for c in chars]) if chars else run.bbox


def projected_bbox(box, direction):
    ux,uy = direction
    vx,vy = -uy,ux
    points = [(box[0],box[1]),(box[0],box[3]),(box[2],box[1]),(box[2],box[3])]
    u = [x*ux+y*uy for x,y in points]
    v = [x*vx+y*vy for x,y in points]
    return min(u),min(v),max(u),max(v)


def parse_dimensions(runs: list[TextRun], excluded_regions=()):
    dimensions, ambiguous = [], []
    for run in runs:
        raw = parser_text(run)
        if run.source=='ocr_tesseract' and sum(c.isalpha() for c in run.raw_text)>18:
            continue
        cx, cy = (run.bbox[0]+run.bbox[2])/2, (run.bbox[1]+run.bbox[3])/2
        if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in excluded_regions):
            continue
        if any(c in raw for c in ("Ç", "Å", "�")):
            ambiguous.append({"text_run_id": run.id, "raw_text": raw, "bbox": run.bbox,
                              "reason": "legacy_font_or_unicode_mapping_requires_visual_review"})
        occupied = []
        for kind, pattern in PATTERNS:
            for m in pattern.finditer(raw):
                if any(m.start() < end and m.end() > start for start,end in occupied):
                    continue
                occupied.append(m.span())
                nominal = number(m["n"])
                matched = m.group()
                suffix = raw[m.end("n"):m.end()] if kind in ("diameter", "radius", "angle") else ""
                upper, lower, warnings = tolerance(suffix)
                dim = Dimension(id=f"D{len(dimensions)+1:04d}", raw_text=matched,
                    normalized_text=normalize(matched), type=kind, nominal_value=nominal,
                    unit="deg" if kind == "angle" else "um" if kind == "roughness" else "mm",
                    upper_deviation=upper, lower_deviation=lower,
                    fit=m.groupdict().get("fit"), pitch=number(m["p"]) if m.groupdict().get("p") else None,
                    angle=number(m["a"]) if m.groupdict().get("a") else None,
                    bbox=match_bbox(run, *m.span()), text_run_ids=[run.id], warnings=warnings,
                    source=run.source,confidence=run.confidence,direction=run.direction)
                if run.source=='ocr_tesseract' or raw!=run.raw_text:
                    dim.raw_text=run.raw_text
                    dim.normalized_text=normalize(matched)
                    dim.alternatives=[{'raw_text':run.raw_text,'structured_reading':matched,'method':run.recognition or 'legacy_cad_glyph_mapping'}]
                    dim.status='needs_review'
                if kind == "angle":
                    minutes = re.search(rf"°\s*({N})\s*['′]", matched)
                    if minutes:
                        dim.nominal_value += number(minutes[1])/60
                        dim.warnings.append("Угловые минуты переведены в десятичные градусы")
                dimensions.append(dim)
                quantity=re.search(r'(\d+)\s*(?:отв\.?|отверст|отб\.)',raw,re.I)
                if quantity and kind in ('diameter','thread'):dim.quantity=int(quantity[1])
                if kind=='thread' and dim.pitch and dim.pitch>dim.nominal_value*.4:
                    dim.status='needs_review'
                    dim.warnings.append('Шаг резьбы необычно велик относительно диаметра; исходное значение сохранено, требуется проверка')
                    dim.alternatives.append({'pitch':dim.pitch/10,'reason':'possible_missing_decimal_separator','accepted':False})
                if matched.count("(") > matched.count(")"):
                    dim.upper_deviation = dim.lower_deviation = None
                    dim.status = "needs_review"
                    dim.warnings.append("Незавершённая запись допуска: отклонения до объединения не определены")
        if not occupied:
            m = LINEAR.fullmatch(raw.strip())
            if m:
                upper, lower, warnings = tolerance(raw.strip()[m.end("n"):])
                dimensions.append(Dimension(id=f"D{len(dimensions)+1:04d}", raw_text=run.raw_text,
                    normalized_text=normalize(raw), type="linear", nominal_value=number(m["n"]), unit="mm",
                    upper_deviation=upper, lower_deviation=lower, fit=m["fit"], bbox=run.bbox,
                    text_run_ids=[run.id], status="needs_review",
                    warnings=warnings+["Изолированное число: принадлежность размерной надписи не подтверждена"],
                    source=run.source,confidence=run.confidence,
                    direction=run.direction,
                    alternatives=[{'raw_text':run.raw_text,'method':run.recognition}] if raw!=run.raw_text else []))
            elif run.source=='ocr_tesseract':
                # A measured decimal nominal may be followed by a smaller stacked
                # deviation whose SIGN OCR lost. Preserve the nominal; do not
                # invent the missing sign or pretend the complete notation passed.
                split=re.fullmatch(r'\s*(\d+[.,]\d+)\s+(\d+[.,]\d+)\s*',raw)
                words=run.recognition.get('word_evidence',[])
                if split and len(words)==2 and float(words[1]['height'])<float(words[0]['height'])*.85 and number(split[2])<number(split[1])*.3:
                    dimensions.append(Dimension(id=f'D{len(dimensions)+1:04d}',raw_text=run.raw_text,
                        normalized_text=split[1],type='linear',nominal_value=number(split[1]),unit='mm',bbox=run.bbox,
                        text_run_ids=[run.id],source=run.source,confidence=run.confidence,direction=run.direction,status='needs_review',
                        warnings=['Номинал прочитан; знак отдельного отклонения не подтверждён и не восстановлен'],
                        alternatives=[{'raw_text':run.raw_text,'structured_reading':raw,'unparsed_unsigned_deviation':split[2],
                                       'punctuation_evidence':run.recognition.get('punctuation_image_evidence',[]),'tolerance_accepted':False}]))

    # Stacked tolerances can be separate PDF blocks, and can be rotated. Compare
    # in the local text axes against the inline deviation, require one candidate.
    run_map = {r.id:r for r in runs}
    for run in runs:
        value = run.raw_text.strip()
        if not re.fullmatch(rf"\(?\s*{SIGNED}(?:\s*{SIGNED})?\s*\)?", value):
            continue
        candidates = []
        for d in dimensions:
            parent = run_map[d.text_run_ids[0]]
            if d.type not in ("diameter", "linear", "radius") or parent.direction != run.direction:
                continue
            unfinished = d.raw_text.count("(") > d.raw_text.count(")")
            if not unfinished and (parent.block_id != run.block_id or d.upper_deviation is not None):
                continue
            if run.size > parent.size*.95:
                continue
            existing = re.search(SIGNED,parent.raw_text)
            reference = match_bbox(parent,*existing.span()) if existing else parent.bbox
            a = projected_bbox(reference,parent.direction)
            b = projected_bbox(run.bbox,parent.direction)
            ugap = max(0,a[0]-b[2],b[0]-a[2])
            vdistance = abs((a[1]+a[3]-b[1]-b[3])/2)
            if ugap <= parent.size*.5 and vdistance <= parent.size*1.25:
                candidates.append(d)
        if len(candidates) == 1:
            d = candidates[0]
            nums = [number(m.group()) for m in re.finditer(SIGNED,d.raw_text+"\n"+value)]
            if len(nums) > 2:
                ambiguous.append({"text_run_id":run.id,"raw_text":run.raw_text,"bbox":run.bbox,"reason":"too_many_tolerance_values"})
                continue
            d.raw_text += "\n"+run.raw_text
            d.normalized_text = normalize(d.raw_text)
            d.bbox = union_bbox([d.bbox,run.bbox])
            d.text_run_ids.append(run.id)
            if len(nums) == 2:
                d.upper_deviation,d.lower_deviation = max(nums),min(nums)
            elif d.raw_text.count("(") == d.raw_text.count(")"):
                d.upper_deviation,d.lower_deviation = max(nums+[0]),min(nums+[0])
            d.status = "needs_review"
            d.warnings.append("Допуск объединён по координатам в осях надписи; проверить визуально")
        else:
            ambiguous.append({"text_run_id": run.id, "raw_text": run.raw_text, "bbox": run.bbox,
                              "reason": "tolerance_without_unique_nominal"})
    return dimensions, ambiguous
