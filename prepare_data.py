import json
import os

INPUT_DIR = "data/grimm_fairytales/"
OUTPUT_DIR = "data/grimm_fairytales_json/"

def convert_fairytale(input_dir, output_dir):
# Convert txt fairytale to json
# KHM = Kinder- und Hausmärchen ID

    with open(input_dir, 'r', encoding='utf-8-sig') as f: #utf-8-sig to handle BOM byte
        khm_number = f.readline()
        text = f.readlines() # all other lines as list

        khm_number = khm_number.replace(".", "").strip()
        khm_number = int(khm_number)

        title = text[0].replace(".", "").strip()
        text = ' '.join(text[1:]).strip() # combine all lines other than title to one string

        fairytale = {'khm': khm_number, 'title': title, 'text': text}

    # utf encoding for Umlaute
    with open(output_dir, "w", encoding='utf-8') as outfile:
        json.dump(fairytale, outfile, indent=3)

for f in os.listdir(INPUT_DIR):
    input_dir = os.path.join(INPUT_DIR, f)
    output_dir = os.path.join(OUTPUT_DIR, f)
    output_dir =output_dir.replace(".txt", ".json")
    convert_fairytale(input_dir, output_dir)