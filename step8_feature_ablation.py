from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common_url_features import COMMON_FEATURES, extract_features

RANDOM_STATE = 42
DATASET_A_PATH = Path("PhiUSIIL_Phishing_URL_Dataset.csv")
DATASET_B_PATH = Path("url_features_extracted1.csv")
OUTPUT_DIR = Path("feature_ablation_results")

FEATURE_SETS = {
    "all_18": COMMON_FEATURES,
    "without_slash_count": [feature for feature in COMMON_FEATURES if feature != "slash_count"],
    "without_slash_and_https": [
        feature for feature in COMMON_FEATURES if feature not in {"slash_count", "is_https"}
    ],
    "top_5": ["slash_count", "is_https", "digit_count", "letter_count", "url_length"],
}


def normalize_domain(url):
    text = str(url).strip()
    parsed = urlparse(text if "://" in text else "http://" + text)
    return (parsed.hostname or "").lower().rstrip(".")


def load_clean(path, label_column):
    df = pd.read_csv(path).dropna(subset=["URL", label_column]).copy()
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    raw_labels = set(df[label_column].astype(int).unique())
    if raw_labels != {0, 1}:
        raise ValueError(f"{path} labels must be 0 and 1, found {sorted(raw_labels)}")
    df["label"] = 1 - df[label_column].astype(int)
    df["domain"] = df["URL"].map(normalize_domain)
    return df


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
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


def evaluate(model, model_name, features, labels, feature_names, evaluation_name, scaler=None):
    selected_features = features[feature_names]
    start = perf_counter()
    transformed = scaler.transform(selected_features) if scaler is not None else selected_features
    predictions = model.predict(transformed)
    inference_seconds = perf_counter() - start
    row_count = len(labels)
    return {
        "Feature_Set": evaluation_name[0],
        "Evaluation": evaluation_name[1],
        "Model": model_name,
        "Feature_Count": len(feature_names),
        "Rows": row_count,
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "Inference_Seconds": inference_seconds,
        "Inference_Milliseconds_Per_Row": inference_seconds / row_count * 1000,
    }


def main():
    dataset_a = load_clean(DATASET_A_PATH, "label")
    dataset_b = load_clean(DATASET_B_PATH, "ClassLabel")
    features_a = extract_features(dataset_a["URL"])
    features_b = extract_features(dataset_b["URL"])

    X_train, X_test, y_train, y_test = train_test_split(
        features_a,
        dataset_a["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=dataset_a["label"],
    )
    dataset_a_domains = set(dataset_a["domain"])
    dataset_b_domain_disjoint = dataset_b[~dataset_b["domain"].isin(dataset_a_domains)].copy()
    features_b_unseen = extract_features(dataset_b_domain_disjoint["URL"])
    y_b_unseen = dataset_b_domain_disjoint["label"]

    result_rows = []
    importance_rows = []
    for feature_set_name, feature_names in FEATURE_SETS.items():
        selected_train = X_train[feature_names]
        selected_test = X_test[feature_names]
        selected_b = features_b_unseen[feature_names]
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(selected_train)
        models = build_models()
        models["Logistic Regression"].fit(train_scaled, y_train)
        models["Random Forest"].fit(selected_train, y_train)
        models["XGBoost"].fit(selected_train, y_train)

        for model_name, model in models.items():
            model_scaler = scaler if model_name == "Logistic Regression" else None
            for evaluation_name, features, labels in (
                ((feature_set_name, "Dataset A test"), selected_test, y_test),
                ((feature_set_name, "Dataset B domain-disjoint"), selected_b, y_b_unseen),
            ):
                result_rows.append(
                    evaluate(
                        model,
                        model_name,
                        features,
                        labels,
                        feature_names,
                        evaluation_name,
                        model_scaler,
                    )
                )

            if model_name == "XGBoost":
                for feature, importance in zip(feature_names, model.feature_importances_):
                    importance_rows.append(
                        {
                            "Feature_Set": feature_set_name,
                            "Feature": feature,
                            "XGBoost_Importance": importance,
                        }
                    )

    results = pd.DataFrame(result_rows)
    importance = pd.DataFrame(importance_rows)
    baseline = results[results["Feature_Set"] == "all_18"][["Evaluation", "Model", "F1"]]
    baseline = baseline.rename(columns={"F1": "All_18_F1"})
    results = results.merge(baseline, on=["Evaluation", "Model"], how="left")
    results["F1_Change_from_All_18"] = results["F1"] - results["All_18_F1"]

    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "xgboost_ablation_importance.csv", index=False)

    summary = results[
        [
            "Feature_Set",
            "Evaluation",
            "Model",
            "Feature_Count",
            "F1",
            "Precision",
            "Recall",
            "Inference_Milliseconds_Per_Row",
            "F1_Change_from_All_18",
        ]
    ].sort_values(["Evaluation", "Model", "Feature_Count"])
    summary.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)

    print("Ablation summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved results in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
