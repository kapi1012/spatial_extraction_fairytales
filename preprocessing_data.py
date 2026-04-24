import os
import json
import spacy
import de_core_news_sm

from spacy.matcher import Matcher

INPUT_DIR = "./data/grimm_fairytales_json"


def load_fairytales(directory):
    fairytales = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as json_file:
            fairytales.append(json.load(json_file))
    return fairytales

fairytales = load_fairytales(INPUT_DIR)
nlp = spacy.load("de_core_news_sm")
doc = nlp(fairytales[1]['text'])

for token in doc:
    print(token.text, token.pos_)

# Design Matcher to find vague locations & prepositions
matcher = Matcher(nlp.vocab)

# pattern: preposition + optional determiner, adjective, adverb + lemma
pattern = [[
    {"LOWER": {"IN": ["im", "in", "am", "auf", "bei", "beim", "durch", "zu", "zur", "unter", "über", "vor", "hinter", "zwischen"]}},
     {"POS": {"IN": ["DET", "ADJ", "ADV"]}, "OP": "*"},
     {"LEMMA": {"IN": ["Wald", "Straße", "Schloss", "Dorf", "Weide", "Wiese", "Herberge", "Haus", "Kirche", "Scheune", "Erde", "Daheim", "Grund", "Fluss", "See", "Wasser"]}}
    ]]

#Todo Spacy Spans extrahieren, Pattern Matcher, dann Label Studio JSON bauen

