import json
import time

from pydantic import ValidationError
from datetime import datetime
from ollama import chat, ResponseError
from data_split import data_split_on_khm, split_dataset_by_khm
from ollama_validation import *
from ollama_chunks import *
from ollama_basemodels import *
from evaluation.evaluation import span_metrics, extract_test_spans, save_local_llm_test_results
from dataset_preprocessing import load_dataset

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

        annotation = None
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"KHM={chunk.khm}, "
                    f"chunk={chunk.chunk_id}, "
                    f"chars={len(chunk.text)}, "
                    f"words={len(chunk.text.split())}"
                )
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

                raw_response = response.message.content

                print(
                    f"Output chars={len(raw_response)}"
                )

                # Parsing successful
                break

            except ValidationError as e:
                print(
                    f"[{model_name}] JSON parsing failed " f"for KHM {chunk.khm}, chunk {chunk.chunk_id} " f"(attempt {attempt}/{max_retries})")
                print(e)
                if attempt < max_retries:
                    time.sleep(1)

            except ResponseError as e:
                print(
                    f"[{model_name}] Ollama error for " f"KHM {chunk.khm}, chunk {chunk.chunk_id} " f"(attempt {attempt}/{max_retries})")
                print(e)
                if attempt < max_retries:
                    time.sleep(5)


        if annotation is None:
            all_span_errors.append(
            {"khm": chunk.khm, "chunk_id": chunk.chunk_id, "reason": "llm_json_validation_failed", })
            #skip chunk
            continue


        annotated_chunk, spans_not_found = find_label_offsets(annotation, chunk)

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

    # keep only unique spans
    # there are duplicates because of overlapping chunks
    unique_spans = {}
    for span in all_spans:
        # combination of span-attributes as key
        key = (span.start, span.end, span.text, span.label)
        # same key -> overwrites duplicate
        unique_spans[key] = span

    # convert back to list
    deduplicated_spans = list(unique_spans.values())

    annotated_fairytale = Fairytale(
        khm=fairytale.khm,
        title=fairytale.title,
        text=fairytale.text,
        spans=deduplicated_spans,
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
DATA_PATH = "data/grimm_fairytales_annotated/vague_loc_after_iaa_29-9-26.json"
NUMBER_SENTENCES = 3

if __name__ == "__main__":
    train_khm, val_khm, test_khm = data_split_on_khm(ANALYSIS_PATH, 42)

    # load dataset for testing
    dataset = load_dataset(DATA_PATH)
    train_fairytales, val_fairytales, test_fairytales = split_dataset_by_khm(dataset, train_khm, val_khm, test_khm, )

    # load systemprompt with annotation guidelines
    with open("./ollama_files/systemprompt.md", "r") as f:
        systemprompt = f.read()

    # load fairytale-json
    with open("./ollama_files/fairytales.json", "r", encoding="utf-8") as ft:
        fairytales = json.load(ft)

    model_name = "mistral-small3.2:24b"
    # load all test fairytales by khm; json not the dataclass
    fairytales_data = [
        item for item in fairytales
        if item["khm"] in test_khm
    ]

    all_annotated_fairytales = []
    all_span_errors = []

    start_time_tracking = time.time()
    for data in fairytales_data:
        fairytale = Fairytale.model_validate(data)
        chunked_fairytale = split_fairytale_to_chunks(fairytale, NUMBER_SENTENCES)
        annotated_chunks, span_errors = llm_annotate_fairytale(chunked_fairytale, fairytale, model_name)
        annotated_fairytale = combine_chunks_to_fairytale(annotated_chunks)

        all_annotated_fairytales.append(annotated_fairytale)
        all_span_errors.extend(span_errors)

    end_annotation_time = time.time()
    annotation_time = end_annotation_time - start_time_tracking

    save_results_and_errors(all_annotated_fairytales, all_span_errors, OUTPUT_DIR)
    all_true_spans, all_pred_spans, evaluation_errors = extract_test_spans(test_fairytales, all_annotated_fairytales)
    span_precision, span_recall, span_f1 = span_metrics(all_true_spans, all_pred_spans)
    save_local_llm_test_results(span_precision, span_recall, span_f1, evaluation_errors, OUTPUT_DIR, model_name, annotation_time, NUMBER_SENTENCES)
