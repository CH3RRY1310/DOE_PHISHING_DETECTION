from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common_url_features import COMMON_FEATURES, extract_features

RANDOM_STATE = 42
DATASET_A_PATH = Path("PhiUSIIL_Phishing_URL_Dataset.csv")
DATASET_B_PATH = Path("url_features_extracted1.csv")
OUTPUT_DIR = Path("common_cross_dataset_results")


def clean_dataset(path, label_column):
    df = pd.read_csv(path)
    required_columns = {"URL", label_column}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    missing_labels = int(df[label_column].isna().sum())
    df = df.dropna(subset=["URL", label_column]).copy()
    before = len(df)
    df = df.drop_duplicates(subset=["URL"]).copy()
    duplicate_urls = before - len(df)
    raw_labels = set(df[label_column].astype(int).unique())
    labels = raw_labels
    if labels != {0, 1}:
        raise ValueError(f"{path} must contain labels 0 and 1, found {sorted(labels)}")
    # Both source files use 0=phishing and 1=legitimate; make phishing the positive class.
    df[label_column] = 1 - df[label_column].astype(int)
    return df, missing_labels, duplicate_urls


def evaluate_model(name, model, X, y, scaler=None):
    transformed = scaler.transform(X) if scaler is not None else X
    predictions = model.predict(transformed)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
    }
    return metrics, predictions


def print_evaluation(title, results, y, predictions):
    print(f"\n{title}")
    print(pd.DataFrame([results]).to_string(index=False))
    print(classification_report(y, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))


def main():
    dataset_a, a_missing_labels, a_duplicates = clean_dataset(DATASET_A_PATH, "label")
    dataset_b, b_missing_labels, b_duplicates = clean_dataset(DATASET_B_PATH, "ClassLabel")

    print(f"Dataset A after cleaning: {dataset_a.shape}")
    print(f"Dataset A missing labels removed: {a_missing_labels}; duplicate URLs removed: {a_duplicates}")
    print(f"Dataset B after cleaning: {dataset_b.shape}")
    print(f"Dataset B missing labels removed: {b_missing_labels}; duplicate URLs removed: {b_duplicates}")
    print("Dataset A labels after mapping (1=phishing, 0=legitimate):")
    print(dataset_a["label"].value_counts().sort_index())
    print("Dataset B labels after mapping (1=phishing, 0=legitimate):")
    print(dataset_b["ClassLabel"].value_counts().sort_index())

    X_a = extract_features(dataset_a["URL"])
    y_a = dataset_a["label"]
    dataset_a_urls = set(dataset_a["URL"])
    overlap = len(dataset_a_urls & set(dataset_b["URL"]))
    dataset_b_external = dataset_b[~dataset_b["URL"].isin(dataset_a_urls)].copy()
    X_b = extract_features(dataset_b_external["URL"])
    y_b = dataset_b_external["ClassLabel"]

    if X_a.isnull().sum().sum() or X_b.isnull().sum().sum():
        raise ValueError("The common URL extractor produced missing feature values")
    print(f"URLs shared by Dataset A and Dataset B: {overlap}")
    print(f"Dataset B non-overlapping evaluation rows: {len(dataset_b_external)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_a,
        y_a,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_a,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    models["Logistic Regression"].fit(X_train_scaled, y_train)
    models["Random Forest"].fit(X_train, y_train)
    models["XGBoost"].fit(X_train, y_train)

    dataset_a_results = []
    dataset_b_results = []
    confusion_matrices = []
    for name, model in models.items():
        model_scaler = scaler if name == "Logistic Regression" else None
        a_metrics, a_predictions = evaluate_model(name, model, X_test, y_test, model_scaler)
        b_metrics, b_predictions = evaluate_model(name, model, X_b, y_b, model_scaler)
        print_evaluation(f"Dataset A - {name}", a_metrics, y_test, a_predictions)
        print_evaluation(f"Dataset B - {name}", b_metrics, y_b, b_predictions)
        dataset_a_results.append(a_metrics)
        dataset_b_results.append(b_metrics)
        for dataset_name, labels, predictions in (
            ("Dataset A", y_test, a_predictions),
            ("Dataset B unseen", y_b, b_predictions),
        ):
            true_negative, false_positive, false_negative, true_positive = confusion_matrix(
                labels,
                predictions,
            ).ravel()
            confusion_matrices.append(
                {
                    "Dataset": dataset_name,
                    "Model": name,
                    "TN": true_negative,
                    "FP": false_positive,
                    "FN": false_negative,
                    "TP": true_positive,
                }
            )

    dataset_a_results = pd.DataFrame(dataset_a_results).sort_values("F1", ascending=False)
    dataset_b_results = pd.DataFrame(dataset_b_results).sort_values("F1", ascending=False)
    comparison = dataset_a_results[["Model", "F1"]].rename(columns={"F1": "Dataset_A_F1"}).merge(
        dataset_b_results[["Model", "F1"]].rename(columns={"F1": "Dataset_B_F1"}),
        on="Model",
    )
    comparison["Generalization_Drop"] = comparison["Dataset_A_F1"] - comparison["Dataset_B_F1"]

    importance = pd.DataFrame(
        {
            "Feature": COMMON_FEATURES,
            "Importance": models["Random Forest"].feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, OUTPUT_DIR / f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    dataset_a_results.to_csv(OUTPUT_DIR / "dataset_a_results.csv", index=False)
    dataset_b_results.to_csv(OUTPUT_DIR / "dataset_b_results.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "generalization_comparison.csv", index=False)
    pd.DataFrame(confusion_matrices).to_csv(
        OUTPUT_DIR / "confusion_matrices.csv",
        index=False,
    )
    importance.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False)
    joblib.dump(
        {
            "features": COMMON_FEATURES,
            "random_state": RANDOM_STATE,
            "label_convention": "0=legitimate, 1=phishing; source labels were 0=phishing, 1=legitimate",
            "dataset_a_rows": len(dataset_a),
            "dataset_b_rows": len(dataset_b),
            "dataset_b_external_rows": len(dataset_b_external),
            "shared_urls": overlap,
        },
        OUTPUT_DIR / "metadata.pkl",
    )

    print("\nGeneralization comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSaved frozen models and results in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
