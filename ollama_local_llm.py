import json

from datetime import datetime
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

def llm_annotate_fairytale(fairytale_chunks, fairytale, model_name):
    annotated_chunks = []
    all_span_errors = []

    for chunk in fairytale_chunks:

        chunked_fairytale_json = json.dumps(
            {
                "khm": chunk.khm,
                "title": chunk.title,

                "text": chunk.text,
            },
            ensure_ascii=False)

        response = chat(
            model=model_name,
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

        annotated_chunk, spans_not_found = find_label_offsets(annotation, chunk)
        # print(annotated_chunk)

        span_errors = validate_chunk_spans(annotated_chunk, fairytale)

        if span_errors:
            all_span_errors.extend(span_errors)
        else:
            print("All spans valid.")

        all_span_errors.extend(spans_not_found)
        annotated_chunks.append(annotated_chunk)

    return annotated_chunks, all_span_errors


# build Fairytale object of all annotated chunks
def combine_chunks_to_fairytale(annotated_chunks):
    all_spans = []

    for chunk in annotated_chunks:
        all_spans.extend(chunk.spans)

    annotated_fairytale = Fairytale(
        khm=fairytale.khm,
        title=fairytale.title,
        text=fairytale.text,
        spans=all_spans,
    )

    return annotated_fairytale


# save results and errors
def save_results_and_errors(annotated_fairytales, all_span_errors, output_dir):
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # convert to dict
    annotated_fairytales_data = [
        fairytale.model_dump()
        for fairytale in annotated_fairytales
    ]

    with open(f"{output_dir}annotated_fairytales_{current_time}.json", "w", encoding="utf-8") as f:
        json.dump(annotated_fairytales_data, f, ensure_ascii=False, indent=4)

    with open(f"{output_dir}annotation_errors_{current_time}.json", "w", encoding="utf-8") as f:
        json.dump(all_span_errors, f, ensure_ascii=False, indent=4)


OUTPUT_DIR = "./ollama_files/"
ANALYSIS_PATH = "data/analysis_dataset/analysis_dataset.csv"
train_khm, val_khm, test_khm = data_split_on_khm(ANALYSIS_PATH, 42)

# load systemprompt with annotation guidelines
with open("./ollama_files/systemprompt.md", "r") as f:
    systemprompt = f.read()

# load fairytale-json
with open("./ollama_files/fairytales.json", "r", encoding="utf-8") as ft:
    fairytales = json.load(ft)

model_name = "gemma3:4b"
# load all test fairytales by khm
fairytales_data = [
    item for item in fairytales
    if item["khm"] in test_khm
]

all_annotated_fairytales = []
all_span_errors = []

for data in fairytales_data:
    fairytale = Fairytale.model_validate(data)
    chunked_fairytale = split_fairytale_to_chunks(fairytale, 3)
    annotated_chunks, span_errors = llm_annotate_fairytale(chunked_fairytale, fairytale, model_name)
    annotated_fairytale = combine_chunks_to_fairytale(annotated_chunks)

    all_annotated_fairytales.append(annotated_fairytale)
    all_span_errors.extend(span_errors)

save_results_and_errors(all_annotated_fairytales, all_span_errors, OUTPUT_DIR)


# 1 fairytale for pipeline testing
#fairytale_data = next(item for item in fairytales if item["khm"] == 1)
#fairytale = Fairytale.model_validate(fairytale_data)
#chunked_fairytale = split_fairytale_to_chunks(fairytale, 3)

# annotate all chunk per 1 fairytale
#annotated_chunks, all_span_errors = llm_annotate_fairytale(chunked_fairytale, fairytale, model_name)
#annotated_fairytale = combine_chunks_to_fairytale(annotated_chunks)
#save_results_and_errors(annotated_fairytale, all_span_errors, OUTPUT_DIR)
