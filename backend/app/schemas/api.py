from typing import Optional, List
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    duplicate: bool = False
    message: str

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_id: Optional[str] = None
    chat_id: Optional[str] = None

class Source(BaseModel):
    document_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_id: str
    score: float
    text_preview: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    rewritten_query: str
    chat_id: str
    cache_hit: bool = False
