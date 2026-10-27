# Spatial information extraction from German folk tales
Computer Science Masterthesis 2026: "Extraction of vague spatial information from German folk tales" at HKA.

## Abstract
In the field of location extraction, descriptions of locations still pose a challenge, particularly longer sequences that do not include proper nouns. Descriptions of locations in folk tales, which refer almost exclusively to non-real locations, serve as an example of this. 

Drawing on concepts of space and place from literary studies, two terms for vague location expressions are defined: Vague Spatial Expression and Conceptual Spatial Expression. Using the two resulting labels, Vague Location and Conceptual Location, 190 texts from the Brothers Grimm’s Children’s and Household Tales are annotated. 

To extract vague location expressions using sequence tagging, this study compares three methods: two BiLSTM-CRF models, a fine-tuned RoBERTa Transformer, and four large language models using in-context learning. The RoBERTa model achieved the best results at the span and token levels, while the in-context learning models lagged behind the baseline defined by the BiLSTM. 

An analysis of the metrics achieved at the token level and an examination of typical annotation errors lead to the conclusion that the size of the dataset is generally sufficient for model training and fine-tuning; however, the low frequency of the Conceptual Location label in absolute terms has a negative impact on the metrics.

Future work could focus on expanding the fairy tale dataset or testing additional machine learning methods, particularly through extensions of in-context learning. Furthermore, the annotated dataset can be used from a literary studies perspective to analyze the occurrence of location expressions in fairy tales.

# Project structure

- Final dataset: in `data` 
- Prepare txt for Label Studio: `data-preparations`
- Prepare dataset for modeltraining: `dataset_preprocessing`
- BiLSTM-CRF, RoBERTa and local LLMs: `models`
- Inter Annotator Agreement (Krippendorffs Alpha and Data): `inter_annotator_agreement`
- Evaluation, model predictions as json and error analysis: `evaluation`
- use `label-studio-data` for data upload for local Label Studio use

# Project setup
This project uses uv as virtual environment manager and Python 3.13.

## uv
To install uv, run the following command:
```bash
pip install uv
```
Or check the documentation here: https://docs.astral.sh/uv/getting-started/installation/

To create a virtual environment, run:
```bash
.venv
```
Alternatively, select a new uv Python Interpreter in you IDE (e.g. https://www.jetbrains.com/help/pycharm/uv.html)

To activate the virtual environment, run:
```bash
source .venv/bin/activate
```
To synchronize the virtual environment with the dependencies listed in `pyproject.toml`, run:
```bash
uv sync
```
To deactivate the virtual environment, run:
```bash
deactivate
```

# Dataset
One main goal is to generate a dataset of German folk tales with annotated spatial information. The dataset will be created by scraping German folk tales from the web and manually annotating them with vague spatial information. 
The dataset contains tagged spatial information as "forest", "castle".

## Spacy
For using spacy with uv run in console:

```bash
uv add pip
```

```bash
uv run -- spacy download de_core_news_sm
```

If you're not using uv, you can simply import it in the Python script after installing it:

```import de_core_news_sm```


## Label Studio
Open-Source tool for data labeling: https://labelstud.io/
Quickstart: https://labelstud.io/quick-start/
To launch Label Studio, run:
```bash
label-studio
```

To use local data in Label Studio, create a directory named `data` or `label-studio-data` in the project root and place your data files there. 

### Labeling Interface

```
<View>
  
  <Header value="$title"/>
  
  <Text name="meta" value="KHM: $khm"/>
  
  <Labels name="label" toName="text">
    <Label value="VAGUE_LOC" background="RED"/>
    <Label value="CONCEPTUAL_LOC" background="BLUE"/>
  </Labels>
  
  <Text name="text" value="$text"/>

</View>
```

## Setup for IAA

- Hetzner Server
  - 2 VCPU
  - 4 GB RAM
  - 40 GB
  - 4,75 € / month

- using subdomain ``https://ortsreferenzen-annotation.lektorat-kasper.de/``

## Setup Local LLM

- Install Ollama locally ``curl -fsSL https://ollama.com/install.sh | sh``
- add ollama to uv
- start Ollama in uv: ``ollama serve``
- if "Error: listen tcp 127.0.0.1:11434: bind: address already in use" appears, Ollama server is added as a system service (Linux):
- ``sudo systemctl stop ollama`` on system console
- ``sudo systemctl status ollama`` check if process is inactive

### Start local model
- choose model from https://ollama.com/search
- download model:
- ``ollama pull <model name>``
- start model locally to chat:
- ``ollama run <model name>``
- type ``/bye`` to leave
- use curl for requests on local server: https://docs.ollama.com/quickstart#ollama-api-2


- check installed models:
- ``ollama list``
- remove models by name:
- ``ollama rm <model name>``
- The models are also stored locally, so find the directory: 
- ``du -sh ~/.ollama/models/``
- delete models