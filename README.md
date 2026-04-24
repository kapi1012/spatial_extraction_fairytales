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

```uv add pip```

```uv run -- spacy download de_core_news_sm```


## Label Studio
Open-Source tool for data labeling: https://labelstud.io/
Quickstart: https://labelstud.io/quick-start/
To launch Label Studio, run:
```bash
label-studio
```
