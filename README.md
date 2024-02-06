# Topic extraction

The source code for the article "Topic extraction in chatbot applications".

## Project structure
```
├── 00-cluster-selection.ipynb
├── 01_topic-extraction.ipynb
├── README.md
├── data
│   └── dev-v2.0.json
├── figures
├── requirements.txt
└── src
    ├── misc.py
    └── nlp.py
```

## Data
The illustrative example is the "dev" dataset from the Stanford Question Answering Dataset. The dataset can be donwloaded from https://rajpurkar.github.io/SQuAD-explorer/. The .json file should be placed in the `data/` folder.

## Dependencies
Create a conda environment from `requirements.txt`:
```bash
conda create python=3.10 --name topic-extraction-env
conda activate topic-extraction-env
pip install -r requirements.txt
```