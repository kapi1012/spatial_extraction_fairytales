import json
import os

INPUT_DIR = "grimm_fairytales/"
OUTPUT_DIR = "grimm_fairytales_json/"

# List with fairytales not in standard german, but a dialect
KHM_NON_STANDARD_GERMAN = [19, 47, 66, 68, 82, 91, 95, 96, 113, 126, 137, 138, 139, 140, 143, 165, 167, 187, 190, 196, 203]

def convert_fairytale(input_dir, output_dir):
# Convert txt fairytale to json
# KHM = Kinder- und Hausmärchen ID

    with open(input_dir, 'r', encoding='utf-8-sig') as f: #utf-8-sig to handle BOM byte
        khm_number = f.readline()
        text = f.readlines() # all other lines as list

        khm_number = khm_number.replace(".", "").strip()
        khm_number = int(khm_number)

        title = text[0].replace(".", "").strip()
        #if 'ß' in title:
        #    title = title.replace('ß' , 'ss')
        text = ' '.join(text[1:]).strip() # combine all lines other than title to one string

        fairytale = {'khm': khm_number, 'title': title, 'text': text}

    # keeping only the title as filename
    output_filename = f"grimm_{fairytale['title'].lower().replace(',', '').replace(' ', '_')}.json"

    umlaute = str.maketrans({"ß": "ss", "ä": "ae", "ö": "oe", "ü": "ue"})
    output_filename = output_filename.translate(umlaute)

    # utf encoding for Umlaute
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding='utf-8') as outfile:
        json.dump(fairytale, outfile, indent=3)

def delete_non_standard_german(directory, khm_list):
    path_to_json_files = directory
    # get all JSON file names as a list
    json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]

    for json_file_name in json_file_names:
            with open(os.path.join(path_to_json_files, json_file_name)) as json_file:
                fairytale = json.load(json_file)
                if fairytale['khm'] in khm_list:
                    os.remove(os.path.join(path_to_json_files, json_file_name))


for f in os.listdir(INPUT_DIR):
    input_dir = os.path.join(INPUT_DIR, f)
    convert_fairytale(input_dir, OUTPUT_DIR)

delete_non_standard_german(OUTPUT_DIR, KHM_NON_STANDARD_GERMAN)