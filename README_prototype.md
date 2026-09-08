# URL Phishing Prototype

This prototype uses the final five-feature XGBoost model:

- `slash_count`
- `is_https`
- `digit_count`
- `letter_count`
- `url_length`

It returns a PHISHING or LEGITIMATE prediction, phishing probability, extracted features, and local SHAP contributions.

## Run

Train the deployment artifact once:

```powershell
C:/Users/USER/AppData/Local/Python/pythoncore-3.14-64/python.exe train_prototype_model.py
```

Classify a URL:

```powershell
C:/Users/USER/AppData/Local/Python/pythoncore-3.14-64/python.exe prototype.py "https://example.com"
```

Run the smoke test:

```powershell
C:/Users/USER/AppData/Local/Python/pythoncore-3.14-64/python.exe test_prototype.py
```

Run the curated validation set:

```powershell
C:/Users/USER/AppData/Local/Python/pythoncore-3.14-64/python.exe prototype_validation.py
```

The results are written to `prototype_validation_results/` as CSV and Markdown.

## Web app

Start the evaluator-facing page:

```powershell
C:/Users/USER/AppData/Local/Python/pythoncore-3.14-64/python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in a browser. Enter a URL, choose **Analyze URL**, and show the prediction, probability, five extracted features, and local SHAP contributions.

This is a research prototype, not a browser reputation service. It uses lexical and structural URL features only, so an unfamiliar legitimate URL can be flagged and predictions should not be treated as proof of safety or maliciousness.
