from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument
from app.core.errors import AppError

ALLOWED = {".pdf", ".docx", ".txt"}

async def extract_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix not in ALLOWED:
        raise AppError("Unsupported file type. Use PDF, DOCX or TXT.", 415, "UNSUPPORTED_FILE")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page_number": i, "section": None, "text": text})
        return pages

    if suffix == ".docx":
        doc = DocxDocument(str(path))
        blocks = []
        section = None
        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            if p.style and p.style.name.lower().startswith("heading"):
                section = text
                continue
            blocks.append({"page_number": None, "section": section, "text": text})
        return blocks

    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"page_number": None, "section": None, "text": text}]
