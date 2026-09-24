from ollama_basemodels import *
from spacy.lang.de import German


def split_fairytale_to_chunks(fairytale: Fairytale, number_of_sentences: int = 1):
    """
    Takes 1 fairytale and splits it into chunks of n sentences.
    Returns a list of chunks with khm, title, start+end index of SpaCy sentences and a chunk_id.
    """
    nlp = German()  # contains tokenizer, no other SpaCy Pipeline Components
    nlp.add_pipe("sentencizer")  # add segmentation by sentence

    doc = nlp(fairytale.text)
    sentences = list(doc.sents) # split to sentences as SpaCy Objects

    chunks = []

    # combine n sentences for more context
    # after: https://stackoverflow.com/a/73958383
    #text_chunks = [" ".join(sentences[i:i + number_of_sentences]) for i in range(0, len(sentences), number_of_sentences)]
    for i in range(0, len(sentences), number_of_sentences):
        chunk_sentences = sentences[i:i + number_of_sentences]

        # save SpaCy start, end indexes for each chunk
        start = chunk_sentences[0].start_char
        end = chunk_sentences[-1].end_char

        chunk_text = fairytale.text[start:end]

        chunks.append(
            FairytaleChunk(
                khm=fairytale.khm,
                title=fairytale.title,
                text=chunk_text,
                start=start,
                end=end,
                chunk_id=i // number_of_sentences,
            )
        )

    return chunks


def find_label_offsets(annotation: LLMFairytaleAnnotation, chunk: FairytaleChunk):

    chunk_text = chunk.text
    chunk_start = chunk.start
    chunk_id = chunk.chunk_id
    khm = chunk.khm

    annotated_spans = []
    span_errors = []

    for span in annotation.spans:
        local_start = chunk_text.lower().find(span.text.lower())

        if local_start == -1:
            span_errors.append(
                {
                    "khm": khm,
                    "chunk_id": chunk_id,
                    "span": span.text,
                    "chunk_text": chunk_text,
                    "reason": "span_not_found_in_chunk",
                }
            )
            #raise ValueError(
            #    f"Span not found in chunk: {span.text!r}"
            #)
            continue

        local_end = local_start + len(span.text)

        absolute_start = chunk_start + local_start
        absolute_end = chunk_start + local_end

        annotated_spans.append(
            Span(
                start=absolute_start,
                end=absolute_end,
                text=span.text,
                label=span.label,
            )
        )

    return chunk.model_copy(update={"spans": annotated_spans}), span_errors