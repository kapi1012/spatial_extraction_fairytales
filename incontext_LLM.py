import string

import markdown
import json
from pydantic import BaseModel
from typing import Literal
from ollama import chat


""""
Ollama in Python: https://www.youtube.com/watch?v=AGAETsxjg0o 

Testing two models: Gemma and Mistral

Start with Gemma 3, 1 b Parameters: https://ollama.com/library/gemma3 -> too small

Structured json output: https://docs.ollama.com/capabilities/structured-outputs#python
- output like (simplified) label-studio output
- define pydantic model

streaming for in-context-learning: https://docs.ollama.com/capabilities/streaming

- Prompt with annotator guidelines
    - Prompt in config as systemprompt -> no need to send it everytime
    
- split Data in sentences
    - SpaCy sentecizer 
    - for context, take x sentences
        -> try 1 sentence, 3, 5, up to 10 as context
        - save id for sentence per fairytale 
    - model returns labeling sentence-wise
    
- assign labels to test data
    - return text + labels as json
    
- function to assign text + labels to offset chars
    -> models can't count offsets 
    - check sentence from original fairytale (id for sentence -> SpaCy split should be deterministic)
    - substring search -> find label text
        - count offset
        - add offset to label

- examples from train data "in context-learning"
    - RAG + Agent?
    - "for each Sentence/Chunk, look up similar texts from train data"
    - how to split train data to match test data?
    
- compute metrics

"""""

# Define pydantic BaseModel similar to Dataclass
# for better JSON output
Label = Literal["VAGUE_LOC", "CONCEPTUAL_LOC"]

class Fairytale(BaseModel):
    khm: int
    title: str
    text: str

class Span(BaseModel):
    start: int
    end: int
    text: str
    context: str
    label: Label

class FairytaleAnnotation(BaseModel):
    fairytale: Fairytale
    spans: list[Span]


# load systemprompt with annotation guidelines
with open("./ollama_files/systemprompt.md", "r") as f:
    systemprompt = f.read()

# load fairytale-json
with open("data/grimm_fairytales_json/grimm_der_nagel.json", "r") as ft:
    fairytale_dict = json.load(ft)
fairytale = Fairytale.model_validate(fairytale_dict)

# todo: Chunk longer fairytales
# load all test-fairytales by khm

fairytale_str = json.dumps(
    fairytale.model_dump(),
    ensure_ascii=False
)

response = chat(
    model='gemma3:4b',
    messages=[
        {
            'role': 'system',
            'content': systemprompt,
        },

        {
            'role': 'user',
            'content': fairytale_str,
        }
    ],
    format=FairytaleAnnotation.model_json_schema(),
    options={
        "temperature": 0,
    }
)

annotation = FairytaleAnnotation.model_validate_json(response.message.content)
print(annotation)

def validate_fairytale(
    input_fairytale: Fairytale,
    annotation: FairytaleAnnotation,
):
    output_fairytale = annotation.fairytale

    if output_fairytale.khm != input_fairytale.khm:
        raise ValueError("KHM was changed.")

    if output_fairytale.title != input_fairytale.title:
        raise ValueError("Title was changed.")

    if output_fairytale.text != input_fairytale.text:
        raise ValueError("Text was changed.")

validate_fairytale(fairytale, annotation)

def validate_spans(annotation: FairytaleAnnotation):
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

#validate_spans(annotation)

def validate_span_offsets(
    annotation: FairytaleAnnotation,
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

span_errors = validate_span_offsets(annotation)

print(f"Spans: {len(annotation.spans)}")
print(f"Offset errors: {len(span_errors)}")

for error in span_errors[:20]:
    print(error)
# evaluation
# check, if offsets are correct:
# assert annotation.fairytale.khm == original.fairytale.khm
# assert annotation.fairytale.text == original.fairytale.text