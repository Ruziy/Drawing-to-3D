import hashlib
from pathlib import Path

import pymupdf as fitz

from .config import DPI, MAX_DOCUMENT_PIXELS, MAX_PAGE_PIXELS, MAX_PAGES
from .schemas import TextRun


def validate_document(path):
    with fitz.open(path) as doc:
        if not doc.is_pdf:
            raise ValueError("Файл не является PDF")
        if doc.needs_pass:
            raise ValueError("PDF защищён паролем")
        if not 1 <= len(doc) <= MAX_PAGES:
            raise ValueError(f"Допустимо от 1 до {MAX_PAGES} страниц")
        total = 0
        for page in doc:
            pixels = int(page.rect.width*DPI/72+1)*int(page.rect.height*DPI/72+1)
            if pixels > MAX_PAGE_PIXELS:
                raise ValueError("Лист слишком большой для 300 DPI (лимит 96 млн пикселей)")
            total += pixels
        if total > MAX_DOCUMENT_PIXELS:
            raise ValueError("Превышен общий лимит 220 млн пикселей при 300 DPI")
        return len(doc)


def inspect_pdf(path: Path, output: Path):
    count = validate_document(path)
    output.mkdir(parents=True, exist_ok=True)
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            rotation = page.rotation
            page.set_rotation(0)  # in-memory only; never save or modify uploaded bytes
            page.get_pixmap(dpi=110, alpha=False).save(output / f"preview-{i+1}.png")
            pages.append({"page":i+1,"width":page.rect.width,"height":page.rect.height,
                          "original_rotation":rotation,"has_text_layer":bool(page.get_text().strip())})
    return {"page_count":count,"pages":pages,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def text_content(page):
    runs, blocks = [], []
    for bi, block in enumerate(page.get_text("rawdict", flags=fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]):
        if block["type"] != 0:
            continue
        block_id = f"T{bi+1:04d}"
        raw_lines, ids = [], []
        for line in block["lines"]:
            chars = [c for span in line["spans"] for c in span["chars"]]
            text = "".join(c["c"] for c in chars)
            if not text.strip():
                continue
            run = TextRun(id=f"TR{len(runs)+1:04d}", raw_text=text,
                bbox=tuple(line["bbox"]), block_id=block_id,
                font=line["spans"][0]["font"], size=line["spans"][0]["size"],
                direction=tuple(line["dir"]),
                characters=[{"text":c["c"],"bbox":list(c["bbox"]),"origin":list(c["origin"])} for c in chars])
            runs.append(run)
            raw_lines.append(text)
            ids.append(run.id)
        blocks.append({"id":block_id,"bbox":list(block["bbox"]),"raw_text":"\n".join(raw_lines),
                       "text_run_ids":ids,"source":"pdf_text"})
    return runs, blocks


def embedded_images(page, directory, base_url):
    result = []
    for i, info in enumerate(page.get_image_info(xrefs=True)):
        xref = info.get("xref", 0)
        record = {k:list(v) if isinstance(v,tuple) else v for k,v in info.items() if k != "digest"}
        if xref:
            raw = page.parent.extract_image(xref)
            name = f"embedded-{i+1}.{raw['ext']}"
            (directory/name).write_bytes(raw["image"])
            record["original_file_url"] = f"{base_url}/{name}"
            record["extraction"] = "pymupdf_extract_image_original_stream_representation"
        result.append(record)
    return result
