from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from app.core.config import get_settings
from app.services.embeddings import get_embedder

settings = get_settings()

@dataclass
class Chunk:
    text: str
    page_number: int | None
    section: str | None
    index: int
    chunk_type: str


def structure_aware_documents(pages: list[dict]) -> list[Document]:
    """Keep page and heading boundaries as metadata before semantic splitting."""
    docs: list[Document] = []
    current_section = ""
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        sections: list[tuple[str, list[str]]] = []
        buf: list[str] = []
        for line in lines:
            is_heading = (
                len(line) <= 100
                and (line.isupper() or line.endswith(":") or line.startswith("#"))
            )
            if is_heading and buf:
                sections.append((current_section, buf)); buf = []
                current_section = line.lstrip("# ").strip()
            elif is_heading:
                current_section = line.lstrip("# ").strip()
            else:
                buf.append(line)
        if buf:
            sections.append((current_section, buf))
        for section, body in sections or [(current_section, lines)]:
            docs.append(Document(
                page_content="\n".join(body),
                metadata={"page_number": page.get("page_number"), "section": section or ""},
            ))
    return docs

async def build_chunks(pages: list[dict]) -> list[Chunk]:
    docs = structure_aware_documents(pages)
    if not docs:
        return []

    semantic = SemanticChunker(
        get_embedder(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=settings.semantic_breakpoint_percentile,
    )
    split_docs: list[Document] = []
    for doc in docs:
        text = doc.page_content.strip()
        if len(text) < settings.chunk_min_chars:
            split_docs.append(doc)
            continue
        parts = await semantic.atransform_documents([doc])
        split_docs.extend(parts)

    # Safety splitter prevents very large semantic chunks.
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_max_chars,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    final_docs = await recursive.atransform_documents(split_docs)

    chunks = []
    for i, doc in enumerate(final_docs):
        text = doc.page_content.strip()
        if not text:
            continue
        chunks.append(Chunk(
            text=text,
            page_number=doc.metadata.get("page_number"),
            section=doc.metadata.get("section") or "",
            index=i,
            chunk_type="semantic+structure",
        ))
    return chunks
