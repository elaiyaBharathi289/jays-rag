from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


settings = get_settings()


@dataclass
class Chunk:
    text: str
    page_number: int | None
    section: str | None
    index: int
    chunk_type: str


def structure_aware_documents(
    pages: list[dict],
) -> list[Document]:
    """
    Preserve page and heading boundaries before
    recursive chunking.

    This avoids running an embedding model during
    the chunking stage.
    """

    docs: list[Document] = []

    current_section = ""

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        lines = [
            x.strip()
            for x in text.splitlines()
            if x.strip()
        ]

        sections: list[
            tuple[str, list[str]]
        ] = []

        buf: list[str] = []

        for line in lines:

            is_heading = (
                len(line) <= 100
                and (
                    line.isupper()
                    or line.endswith(":")
                    or line.startswith("#")
                )
            )

            if is_heading and buf:

                sections.append(
                    (
                        current_section,
                        buf,
                    )
                )

                buf = []

                current_section = (
                    line
                    .lstrip("# ")
                    .strip()
                )

            elif is_heading:

                current_section = (
                    line
                    .lstrip("# ")
                    .strip()
                )

            else:

                buf.append(line)

        if buf:

            sections.append(
                (
                    current_section,
                    buf,
                )
            )

        for section, body in (
            sections
            or [(current_section, lines)]
        ):

            docs.append(
                Document(
                    page_content="\n".join(body),
                    metadata={
                        "page_number": page.get(
                            "page_number"
                        ),
                        "section": section or "",
                    },
                )
            )

    return docs


async def build_chunks(
    pages: list[dict],
) -> list[Chunk]:

    print(
        "[CHUNKING] Starting structure-aware chunking...",
        flush=True,
    )

    docs = structure_aware_documents(pages)

    print(
        f"[CHUNKING] Structure-aware documents: "
        f"{len(docs)}",
        flush=True,
    )

    if not docs:
        return []

    # ---------------------------------------------------------
    # Recursive splitter
    # ---------------------------------------------------------
    #
    # We intentionally do NOT use SemanticChunker here.
    #
    # SemanticChunker requires embeddings during the chunking
    # stage. That was causing the embedding model to be loaded
    # before the normal embedding pipeline.
    #
    # Embeddings will happen later in vector_store.py.
    #

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_max_chars,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    print(
        "[CHUNKING] Starting recursive splitting...",
        flush=True,
    )

    final_docs = await recursive.atransform_documents(
        docs
    )

    print(
        f"[CHUNKING] Recursive splitting completed: "
        f"{len(final_docs)} chunks",
        flush=True,
    )

    # ---------------------------------------------------------
    # Convert to application Chunk objects
    # ---------------------------------------------------------

    chunks: list[Chunk] = []

    for i, doc in enumerate(final_docs):

        text = doc.page_content.strip()

        if not text:
            continue

        chunks.append(
            Chunk(
                text=text,
                page_number=doc.metadata.get(
                    "page_number"
                ),
                section=doc.metadata.get(
                    "section"
                ) or "",
                index=i,
                chunk_type="structure+recursive",
            )
        )

    print(
        f"[CHUNKING] Final chunks: {len(chunks)}",
        flush=True,
    )

    return chunks