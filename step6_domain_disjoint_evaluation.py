from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from common_url_features import extract_features

RANDOM_STATE = 42
DATASET_A_PATH = Path("PhiUSIIL_Phishing_URL_Dataset.csv")
DATASET_B_PATH = Path("url_features_extracted1.csv")
MODEL_DIR = Path("common_cross_dataset_results")
OUTPUT_DIR = Path("domain_disjoint_results")

MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


def normalize_domain(url):
    text = str(url).strip()
    parsed = urlparse(text if "://" in text else "http://" + text)
    return (parsed.hostname or "").lower().rstrip(".")


def load_clean(path, label_column):
    df = pd.read_csv(path).dropna(subset=["URL", label_column]).copy()
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True).copy()
    raw_labels = set(df[label_column].astype(int).unique())
    if raw_labels != {0, 1}:
        raise ValueError(f"{path} labels must be 0 and 1, found {sorted(raw_labels)}")
    # Source labels are 0=phishing and 1=legitimate; use 1=phishing consistently.
    df["label"] = 1 - df[label_column].astype(int)
    df["domain"] = df["URL"].map(normalize_domain)
    return df


def evaluate(model_name, model, X, y, scaler=None):
    transformed = scaler.transform(X) if scaler is not None else X
    predictions = model.predict(transformed)
    return {
        "Model": model_name,
        "Rows": len(y),
        "Legitimate": int((y == 0).sum()),
        "Phishing": int((y == 1).sum()),
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
    }


def main():
    dataset_a = load_clean(DATASET_A_PATH, "label")
    dataset_b = load_clean(DATASET_B_PATH, "ClassLabel")

    features_a = extract_features(dataset_a["URL"])
    X_train, X_test, y_train, y_test = train_test_split(
        features_a,
        dataset_a["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=dataset_a["label"],
    )
    dataset_a_test_domains = set(dataset_a.loc[X_test.index, "domain"])
    dataset_a_urls = set(dataset_a["URL"])
    dataset_a_domains = set(dataset_a["domain"])

    b_full = dataset_b.copy()
    b_url_disjoint = dataset_b[~dataset_b["URL"].isin(dataset_a_urls)].copy()
    b_domain_disjoint = dataset_b[~dataset_b["domain"].isin(dataset_a_domains)].copy()
    evaluations = {
        "Dataset A test": (X_test, y_test),
        "Dataset B full": (
            extract_features(b_full["URL"]),
            b_full["label"],
        ),
        "Dataset B URL-disjoint": (
            extract_features(b_url_disjoint["URL"]),
            b_url_disjoint["label"],
        ),
        "Dataset B domain-disjoint": (
            extract_features(b_domain_disjoint["URL"]),
            b_domain_disjoint["label"],
        ),
    }

    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODELS.items()
    }
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    result_rows = []
    for evaluation_name, (features, labels) in evaluations.items():
        for model_name, model in models.items():
            model_scaler = scaler if model_name == "Logistic Regression" else None
            row = evaluate(model_name, model, features, labels, model_scaler)
            row["Evaluation"] = evaluation_name
            result_rows.append(row)

    results = pd.DataFrame(result_rows)[
        [
            "Evaluation",
            "Model",
            "Rows",
            "Legitimate",
            "Phishing",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]
    ]
    baseline = results.loc[results["Evaluation"] == "Dataset A test", ["Model", "F1"]]
    baseline = baseline.rename(columns={"F1": "Dataset_A_F1"})
    comparison = results.merge(baseline, on="Model")
    comparison["F1_Change_from_A"] = comparison["F1"] - comparison["Dataset_A_F1"]
    comparison["F1_Drop_from_A"] = comparison["Dataset_A_F1"] - comparison["F1"]

    counts = pd.DataFrame(
        [
            {
                "Evaluation": name,
                "Rows": len(labels),
                "Legitimate": int((labels == 0).sum()),
                "Phishing": int((labels == 1).sum()),
                "Unique_domains": int(
                    len(dataset_a_test_domains)
                    if name == "Dataset A test"
                    else {
                        "Dataset B full": b_full,
                        "Dataset B URL-disjoint": b_url_disjoint,
                        "Dataset B domain-disjoint": b_domain_disjoint,
                    }[name]["domain"].nunique()
                ),
            }
            for name, (_, labels) in evaluations.items()
        ]
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "all_evaluation_results.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "generalization_comparison.csv", index=False)
    counts.to_csv(OUTPUT_DIR / "evaluation_counts.csv", index=False)

    print("Evaluation counts:")
    print(counts.to_string(index=False))
    print("\nF1 comparison:")
    print(
        comparison[
            ["Evaluation", "Model", "F1", "Dataset_A_F1", "F1_Change_from_A"]
        ].to_string(index=False)
    )
    print(f"\nSaved results in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
