import os
import json
import spacy

from spacy.matcher import Matcher

INPUT_DIR = "./data/grimm_fairytales_json"
OUTPUT_DIR = "./data/grimm_fairytales_preannotated"

def load_fairytales(directory):
    fairytales = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as json_file:
            fairytales.append(json.load(json_file))
    return fairytales

fairytales = load_fairytales(INPUT_DIR)
nlp = spacy.load("de_core_news_sm")
#doc = nlp(fairytales[1]['text'])

# Design Matcher to find vague locations & prepositions
# initialize Matcher
matcher = Matcher(nlp.vocab)

# pattern: preposition + optional determiner, adjective, adverb + lemma
# second pattern: indefinite article + optional adjective, adverb + lemma
FUZZY_LOCATIONS = ["Wald", "Straße", "Schloss", "Dorf", "Weide", "Wiese", "Herberge", "Haus", "Kirche", "Scheune", "Erde", "Daheim", "Grund", "Fluss", "See", "Wasser"]
pattern = [
    [
    {"LOWER": {"IN": ["im", "in", "am", "auf", "bei", "beim", "durch", "zu", "zur", "unter", "über", "vor", "hinter", "zwischen"]}},
     {"POS": {"IN": ["DET", "ADJ", "ADV"]}, "OP": "*"},
     {"LEMMA": {"IN": FUZZY_LOCATIONS}}
    ],
    [
        {"LEMMA": "ein"},
        {"POS": {"IN": ["ADV", "ADJ"]}, "OP": "*"},
        {"LEMMA": {"IN": FUZZY_LOCATIONS}}
    ]
]

matcher.add("FUZZY_LOC", pattern)


def to_labelstudio(text, spans):
    return {
        "data": {"text": text},
        "predictions": [
            {
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": s["start"],
                            "end": s["end"],
                            "text": s["text"],
                            "labels": [s["label"]]
                        }
                    }
                    for s in spans
                ]
            }
        ]
    }


for idx, fairytale in enumerate(fairytales):
    doc = nlp(fairytale['text'])
    matches = matcher(doc)

    # Extract spans from matches
    all_spans = []
    for match_id, start, end in matches:
        span = doc[start:end]
        all_spans.append({
            "text": span.text,
            "start": span.start_char,
            "end": span.end_char,
            "label": "FUZZY_LOC"
        })

    # Create output structure with original metadata
    output_data = {
        "khm": fairytale['khm'],
        "title": fairytale['title'],
        "text": fairytale['text'],
        "annotations": to_labelstudio(fairytale['text'], all_spans)
    }

    # Save with original filename
    output_filename = f"grimm_{fairytale['title'].lower().replace(' ', '_')}_annotated.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
