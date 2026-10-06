def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping fixed-size character chunks.

    Whitespace is normalized so chunk boundaries don't depend on incidental
    formatting in the source document.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start += step
    return chunks
