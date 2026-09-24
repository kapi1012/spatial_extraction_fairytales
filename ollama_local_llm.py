import json

from ollama import chat

from data_split import data_split_on_khm
from ollama_validation import *
from ollama_chunks import *
from ollama_basemodels import *

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

ANALYSIS_PATH = "data/analysis_dataset/analysis_dataset.csv"
train_khm, val_khm, test_khm = data_split_on_khm(ANALYSIS_PATH, 42)

# load systemprompt with annotation guidelines
with open("./ollama_files/systemprompt.md", "r") as f:
    systemprompt = f.read()

# load fairytale-json
with open("./ollama_files/fairytales.json", "r", encoding="utf-8") as ft:
    fairytales = json.load(ft)
#todo load all test fairytales by khm
#todo split all test fairytales

# 1 fairytale for pipeline testing
fairytale_data = next(item for item in fairytales if item["khm"] == 1)
fairytale = Fairytale.model_validate(fairytale_data)


chunked_fairytale = split_fairytale_to_chunks(fairytale, 3)

#chunked_fairytale_test = chunked_fairytale[0]
#chunked_fairytale_test_json = json.dumps(    {
#        "khm": chunked_fairytale_test.khm,
#        "title": chunked_fairytale_test.title,
#        "text": chunked_fairytale_test.text,
#    }, ensure_ascii=False)
#print(chunked_fairytale_test)
#validate_chunk(chunked_fairytale_test, fairytale)

# annotate all chunk per 1 fairytale
annotated_chunks = []
all_span_errors = []

for chunk in chunked_fairytale:

    chunked_fairytale_json = json.dumps(
        {
            "khm": chunk.khm,
            "title": chunk.title,
            "text": chunk.text,
        },
        ensure_ascii=False)

    response = chat(
        model='gemma3:4b',
        messages=[
            {
                'role': 'system',
                'content': systemprompt,
            },

            {
                'role': 'user',
                'content': chunked_fairytale_json,
            }
        ],
        format=LLMFairytaleAnnotation.model_json_schema(),
        options={
            "temperature": 0,
        }
    )


    annotation = LLMFairytaleAnnotation.model_validate_json(response.message.content)
    #print(annotation)

    annotated_chunk, spans_not_found = find_label_offsets(annotation, chunk)
    print(annotated_chunk)

    span_errors = validate_chunk_spans(annotated_chunk, fairytale)

    if span_errors:
        print("Span errors:")
        for error in span_errors:
            print(error)
    else:
        print("All spans valid.")

    annotated_chunks.append(annotated_chunk)
    all_span_errors.append(spans_not_found)

# build Fairytale object of all annotated spans

all_spans = []

for chunk in annotated_chunks:
    all_spans.extend(chunk.spans)

annotated_fairytale = fairytale.model_copy(update={"spans": all_spans})

print(annotated_fairytale)
print("ALLE SPAN ERRORS:")
print(json.dumps(
    all_span_errors,
    ensure_ascii=False,
    indent=2
))