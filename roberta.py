import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForTokenClassification

# model: https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-german
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-german")
model = AutoModelForTokenClassification.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-german", device_map="auto")



# check: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ