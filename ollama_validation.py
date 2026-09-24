from ollama_basemodels import *

def validate_chunk(
    chunk: FairytaleChunk,
    original_fairytale: Fairytale,
):
    full_text = original_fairytale.text

    # Offset of Chunk
    if not (
        0 <= chunk.start < chunk.end <= len(full_text)
    ):
        raise ValueError(
            f"Invalid chunk offsets: "
            f"{chunk.start}:{chunk.end}"
        )

    # Chunk == Original text
    extracted_text = full_text[chunk.start:chunk.end]

    if extracted_text != chunk.text:
        raise ValueError(
            f"Chunk text does not match original text "
            f"for chunk {chunk.chunk_id}."
        )

# Validation spans for Ollama LLMs
def validate_chunk_spans(chunk: FairytaleChunk, original_fairytale: Fairytale) -> list[dict]:

    errors = []

    full_text = original_fairytale.text

    for i, span in enumerate(chunk.spans):

        # Offset in Fairytale

        if not (
            0 <= span.start < span.end <= len(full_text)
        ):
            errors.append({
                "khm": chunk.khm,
                "chunk_id": chunk.chunk_id,
                "span_index": i,
                "reason": "invalid_offset_range",
                "start": span.start,
                "end": span.end,
                "span_text": span.text,
            })
            continue

        # Offset is span text

        extracted_text = full_text[span.start:span.end]

        if extracted_text != span.text:
            errors.append({
                "khm": chunk.khm,
                "chunk_id": chunk.chunk_id,
                "span_index": i,
                "reason": "text_mismatch",
                "start": span.start,
                "end": span.end,
                "span_text": span.text,
                "original_text": extracted_text,
            })

        # span in chunk

        if not (
            chunk.start <= span.start
            and span.end <= chunk.end
        ):
            errors.append({
                "khm": chunk.khm,
                "chunk_id": chunk.chunk_id,
                "span_index": i,
                "reason": "span_outside_chunk",
                "start": span.start,
                "end": span.end,
                "chunk_start": chunk.start,
                "chunk_end": chunk.end,
                "span_text": span.text,
            })

    return errors