from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    content: str = Field(min_length=1)
    corpus: str = "default"
    metadata: dict = Field(default_factory=dict)


class DocumentOut(BaseModel):
    doc_id: str
    corpus: str
    chunks_created: int


class DocumentSummary(BaseModel):
    doc_id: str
    corpus: str
    chunk_id: str
    text: str
    metadata: dict


class DocumentList(BaseModel):
    documents: list[DocumentSummary]


class DeleteResponse(BaseModel):
    doc_id: str
    chunks_removed: int
