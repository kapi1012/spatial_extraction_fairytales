import os
import json

INPUT_DIR = "data/data_preparations/grimm_fairytales_json"

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


def combine_json_files(input_dir):
    json_files = []
    for filename in os.listdir(input_dir):
        with open(os.path.join(input_dir, filename)) as json_file:
            json_files.append(json.load(json_file))
    json_files.sort(key=lambda x: x["khm"])
    with open("models/ollama_files/fairytales.json", "w") as file:
        json.dump(json_files, file, indent=4)

combine_json_files(INPUT_DIR)

