# Census Income Project

Predict whether an individual's income exceeds $50K/year using 1994 US Census
data (Kohavi & Becker, UCI ML Repository).

## Folder structure

```
census-income-project/
├── .vscode/
│   └── settings.json        # editor + Python interpreter settings
├── data/                     # (optional) local copies of the dataset
├── notebooks/
│   └── exploration.ipynb     # exploratory / scratch notebook
├── outputs/
│   ├── figures/               # generated PNG plots
│   └── models/                # saved model artifacts (.pkl)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # load + clean the raw CSV
│   ├── eda.py                 # exploratory plots
│   ├── preprocess.py          # encoding, train/test split, scaling
│   ├── train.py                # model definitions + training loop
│   ├── evaluate.py             # confusion matrix, ROC, feature importance
│   └── main.py                  # pipeline entrypoint
├── tests/
│   └── __init__.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.main
```

This loads the dataset, runs EDA, trains four models (Logistic Regression,
Decision Tree, Random Forest, Gradient Boosting), and saves comparison
metrics and plots to `outputs/`.

## Results

Gradient Boosting was the best performer: **87.1% accuracy, 0.93 ROC-AUC**.
Top predictive features: relationship status, capital gain, education level.
