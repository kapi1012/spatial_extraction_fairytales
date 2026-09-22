import os
import json

INPUT_DIR = "data/grimm_fairytales_json"

def count_words(input_dir):
    word_count = 0
    for filename in os.listdir(input_dir):
        with open(os.path.join(input_dir, filename)) as json_file:
            fairytale = json.load(json_file)
            text = fairytale["text"]
            words = text.split()
            word_count += len(words)
    return word_count

overall_word_count = count_words(INPUT_DIR)
print(overall_word_count)


# Validation spans for Ollama LLMs
def validate_spans(annotation):
    text = annotation.fairytale.text

    for span in annotation.spans:
        if not (0 <= span.start < span.end <= len(text)):
            raise ValueError(
                f"Invalid span offsets: {span.start}:{span.end}"
            )

        extracted_text = text[span.start:span.end]

        if extracted_text != span.text:
            raise ValueError(
                f"Span text does not match offsets: "
                f"{span.start}:{span.end} -> "
                f"{span.text!r} != {extracted_text!r}"
            )


def validate_span_offsets(
        annotation,
):
    text = annotation.fairytale.text

    errors = []

    for i, span in enumerate(annotation.spans):

        if not (0 <= span.start < span.end <= len(text)):
            errors.append({
                "index": i,
                "reason": "invalid_offset_range",
                "start": span.start,
                "end": span.end,
                "span_text": span.text,
            })
            continue

        extracted = text[span.start:span.end]

        if extracted != span.text:
            errors.append({
                "index": i,
                "reason": "text_mismatch",
                "start": span.start,
                "end": span.end,
                "span_text": span.text,
                "original_text": extracted,
            })

    return errors