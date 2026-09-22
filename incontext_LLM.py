import json
from pydantic import BaseModel
from typing import Literal
from ollama import chat
from spacy.lang.de import German

from helpers import validate_span_offsets, validate_spans

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


#fairytale_str = json.dumps(fairytale.model_dump(),ensure_ascii=False)

# Chunk longer fairytales
nlp = German()  # contains tokenizer, no other SpaCy Pipeline Components
nlp.add_pipe("sentencizer")  # add segmentation by sentence

doc = nlp(fairytale.text)
sentences = [sent.text for sent in doc.sents]
#print(sentences)


# combine n sentences for more context
# after: https://stackoverflow.com/a/73958383
def combine_sentences(sentences, number_of_sentences):
    n = number_of_sentences
    n_sentences = [' '.join(map(str, sentences[i:i + n])) for i in range(0, len(sentences), n)]
    return n_sentences


combined_sentences = combine_sentences(sentences, 3)
#print(combined_sentences)

# add title and khm to combined_sentences as json
chunked_fairytale = {"text": combined_sentences, "khm": fairytale.khm, "title": fairytale.title}
chunked_fairytale = json.dumps(chunked_fairytale)

chunked_fairytale_test = {"text": combined_sentences[0], "khm": fairytale.khm, "title": fairytale.title}

print(chunked_fairytale_test)
#todo load all test fairytales by khm
#todo split all test fairytales


chunked_fairytale_test = Fairytale.model_validate(chunked_fairytale_test)
chunked_fairytale_str = json.dumps(chunked_fairytale_test.model_dump(), ensure_ascii=False)

response = chat(
    model='gemma3:4b',
    messages=[
        {
            'role': 'system',
            'content': systemprompt,
        },

        {
            'role': 'user',
            #'content': fairytale_str,
            'content': chunked_fairytale_str,
        }
    ],
    format=FairytaleAnnotation.model_json_schema(),
    options={
        "temperature": 0,
    }
)


annotation = FairytaleAnnotation.model_validate_json(response.message.content)
print(annotation)


def find_label_offsets(annotation: FairytaleAnnotation, original_fairytale: Fairytale):
    label_dict = {"khm": annotation.fairytale.khm, "spans": []}  # contains: khm, span: start, end, text, label
    complete_fairytale_text = original_fairytale.text

    for span in annotation.spans:
        # find start, end from sentences in complete fairytale
        substring = annotation.fairytale.text
        start_index_substring = complete_fairytale_text.index(substring)
        end_index_substring = start_index_substring + len(substring)

        # search only within start, end for label-text
        label_text = span.text
        start_index_label = complete_fairytale_text.index(label_text, start_index_substring, end_index_substring)
        end_index_label = start_index_label + len(label_text)

        # overwrite false annotations
        span.start = start_index_label
        span.end = end_index_label

        # build span
        new_span = Span(start=start_index_label, end=end_index_label, text=span.text, label=span.label)
        # add all to dict
        label_dict["spans"].append(new_span)

    return label_dict, annotation

labels, annotation = find_label_offsets(annotation, fairytale)
print(labels)
print(annotation)


validate_spans(annotation)
span_errors = validate_span_offsets(annotation)

print(f"Spans: {len(annotation.spans)}")
print(f"Offset errors: {len(span_errors)}")
