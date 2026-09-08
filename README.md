# URL Sentinel

A Streamlit proof-of-concept for lightweight phishing-URL detection.

The application extracts five lexical URL features, applies a five-feature XGBoost model, and displays a phishing probability plus local SHAP explanations.

## Features

- `slash_count`
- `is_https`
- `digit_count`
- `letter_count`
- `url_length`

## Run locally

Use Python 3.14.7 or a compatible Python environment:

```powershell
C:/Users/USER/AppData/Local/Python/pythoncore-3.14-64/python.exe -m pip install -r requirements.txt
C:/Users/USER/AppData/Local/Python/pythoncore-3.14-64/python.exe -m streamlit run app.py
```

Open `http://localhost:8501`.

## Reproduce the prototype model

```powershell
C:/Users/USER/AppData/Local/Python/pythoncore-3.14-64/python.exe train_prototype_model.py
C:/Users/USER/AppData/Local/Python/pythoncore-3.14-64/python.exe test_prototype.py
C:/Users/USER/AppData/Local/Python/pythoncore-3.14-64/python.exe prototype_validation.py
```

The trained deployment files are stored in `prototype_artifacts/`. The validation report is stored in `prototype_validation_results/`.

## Project structure

- `app.py`: Streamlit evaluator interface
- `prototype.py`: reusable prediction and SHAP explanation class
- `common_url_features.py`: shared raw-URL feature extractor
- `train_prototype_model.py`: deployment artifact trainer
- `prototype_validation.py`: curated validation runner
- `prototype_artifacts/`: trained five-feature XGBoost model and metadata
- `requirements.txt`: pinned Python dependencies

## Research scope

The model uses URL lexical and structural features only. It does not visit URLs, inspect page content, query reputation services, or prove that a website is safe or malicious. An unfamiliar legitimate URL can be flagged. The system is a research proof of concept, not a production security service.

## Data and GitHub

Raw datasets are intentionally excluded from Git tracking by `.gitignore` because they may be large or subject to source licensing terms. Keep the following files in the local project directory when reproducing experiments:

- `PhiUSIIL_Phishing_URL_Dataset.csv`
- `url_features_extracted1.csv`

Generated research outputs are also excluded except for the deployment artifacts and curated validation report needed by the demo. Use Git LFS or an approved external data store if the dataset source permits redistribution.
