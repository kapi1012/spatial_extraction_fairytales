import os
import json
import spacy

from spacy.matcher import Matcher
from spacy.lookups import Lookups

INPUT_DIR = "./data/grimm_fairytales_json"
OUTPUT_DIR = "./label-studio-data/grimm_fairytales_preannotated"

def load_fairytales(directory):
    fairytales = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as json_file:
            fairytales.append(json.load(json_file))
    return fairytales

fairytales = load_fairytales(INPUT_DIR)
nlp = spacy.load("de_core_news_sm")


# Lookups for old spelling
lookups = Lookups()
lookups.add_table("lemma_lookup", {
    "Thal": "Tal",
    "Schloß": "Schloss",
    "Thor": "Tor",
    "Tahl": "Tal",
    "rothe": "rote",
    "Wirthshaus": "Wirtshaus",
    "Thurm": "Turm",
    "Walde": "Wald"

})

# Design Matcher to find vague locations & prepositions
# initialize Matcher
matcher = Matcher(nlp.vocab)

# pattern: preposition + optional determiner, adjective, adverb + lemma
# second pattern: indefinite article + optional adjective, adverb + lemma
FUZZY_LOCATIONS = ["Wald", "Straße", "Weg", "Schloss", "Dorf", "Weide", "Wiese", "Herberge", "Haus", "Kirche", "Scheune",
                   "Erde", "Daheim", "Grund", "Fluss", "See", "Wasser", "Haselbaum", "Baum", "Grab", "Friedhof", "Galgen",
                   "Brunnen", "Boden", "Gasthof", "Brücke", "Berg", "Hügel", "Tal", "Hain", "Garten", "Hof", "Feld",
                   "Burg", "Gefängnis", "Kerker", "Tor", "Schmiede", "Hütte", "Scheune", "Wirtshaus", "Küche", "Kammer",
                   "Keller", "Treppe", "Bett", "Palast", "Mühle", "Stube", "Stall", "Kutsche", "Acker", "Insel",
                   ]
pattern = [
    [
    {"LOWER": {"IN": ["im", "in", "ins", "am", "an", "auf", "aus", "bei", "beim", "durch", "zu", "zur", "zum", "unter",
                      "über", "vor", "hinter", "zwischen", "um", "ums", "neben"]}},
     {"POS": {"IN": ["DET", "ADJ", "ADV"]}, "OP": "*"},
     {"LEMMA": {"IN": FUZZY_LOCATIONS}}
    ],
    [
        {
            "POS": "DET",
            "LEMMA": {"IN": ["ein", "mein", "dein", "sein", "ihr", "unser", "der"]}
        },
        {"POS": {"IN": ["ADV", "ADJ"]}, "OP": "*"},
        {"LEMMA": {"IN": FUZZY_LOCATIONS}}
    ]
]

matcher.add("FUZZY_LOC", pattern)


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

    # Create output structure - data and annotations at top level
    output_data = {
        "data": {"text": fairytale['text'],
                 # Metadata
                 "khm": fairytale['khm'],
                 "title": fairytale['title']
                 },
        "predictions": [    # predictions for pre-annotation
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
                    for s in all_spans
                ]
            }
        ],
    }

    # Save with original filename
    output_filename = f"grimm_{fairytale['title'].lower().replace(' ', '_')}_annotated.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
