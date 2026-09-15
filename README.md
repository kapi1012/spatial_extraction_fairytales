# Spatial information extraction from German folk tales
Computer Science Masterthesis 2026: "Extraction of vague spatial information from German folk tales" at HKA.

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
  <Labels name="label" toName="text">
    <Label value="FUZZY_LOC" background="#FFA39E"/>
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
- Login to the server via SSH:

``ssh root@62.238.6.54``

``cd label-studio``

``docker compose up -d``

- using subdomain ``https://ortsreferenzen-annotation.lektorat-kasper.de/``

## Setup Local LLM

- Install Ollama locally ``curl -fsSL https://ollama.com/install.sh | sh``
- add ollama to uv
- start Ollama in uv: ``ollama serve``
- if "Error: listen tcp 127.0.0.1:11434: bind: address already in use" appears, Ollama server is added as a system service (Linux):
- ``systemctl stop ollama`` on system console

### Start local model
- choose model from https://ollama.com/search
- download model:
- ``ollama pull <model name>``
- start model locally to chat:
- ``ollama run <model name>``
- type ``/bye`` to leave
- use curl for requests on local server: https://docs.ollama.com/quickstart#ollama-api-2
- 

- check installed models:
- ``ollama list``
- remove models by name:
- ``ollama rm <model name>``
- The models are also stored locally, so find the directory: 
- ``du -sh ~/.ollama/models/``
- delete models