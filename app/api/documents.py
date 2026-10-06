from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.schemas.documents import (
    DeleteResponse,
    DocumentIn,
    DocumentList,
    DocumentOut,
    DocumentSummary,
)
from app.services.rag.service import RagService

router = APIRouter()


@router.post("/documents", response_model=DocumentOut)
def ingest_document(
    document: DocumentIn, rag_service: RagService = Depends(get_rag_service)
) -> DocumentOut:
    doc_id, chunks_created = rag_service.ingest(
        document.content, corpus=document.corpus, metadata=document.metadata
    )
    return DocumentOut(doc_id=doc_id, corpus=document.corpus, chunks_created=chunks_created)


@router.get("/documents", response_model=DocumentList)
def list_documents(
    corpus: str | None = None, rag_service: RagService = Depends(get_rag_service)
) -> DocumentList:
    chunks = rag_service.list_documents(corpus)
    return DocumentList(
        documents=[
            DocumentSummary(
                doc_id=chunk.doc_id,
                corpus=chunk.corpus,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]
    )


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(
    doc_id: str, rag_service: RagService = Depends(get_rag_service)
) -> DeleteResponse:
    removed = rag_service.delete_document(doc_id)
    return DeleteResponse(doc_id=doc_id, chunks_removed=removed)
