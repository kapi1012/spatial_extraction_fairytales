import markdown
import json
from ollama import chat

""""
Ollama in Python: https://www.youtube.com/watch?v=AGAETsxjg0o 

Testing two models: Gemma and Mistral

Start with Gemma 3, 1 b Parameters: https://ollama.com/library/gemma3

Structured json output: https://docs.ollama.com/capabilities/structured-outputs#python
- output like (simplified) label-studio output

streaming for in-context-learning: https://docs.ollama.com/capabilities/streaming

- Prompt with annotator guidelines
- examples from train data
- assign labels to test data
- compute metrics

"""""
# load systemprompt with annotation guidelines
with open("./ollama_files/systemprompt.md", "r") as f:
    systemprompt = f.read()

# load fairytale-json
with open("data/grimm_fairytales_json/grimm_der_nagel.json", "r") as ft:
    fairytale = json.load(ft)
fairytale = json.dumps(fairytale)

response = chat(
    model='gemma3:1b',
    messages=[
        {
            'role': 'system',
            'content': systemprompt,
        },

        {
            'role': 'user',
            'content': fairytale,
        }
    ],

)
print(response.message.content)