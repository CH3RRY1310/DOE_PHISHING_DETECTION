import json
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from common_url_features import COMMON_FEATURES, extract_features

RANDOM_STATE = 42
DATASET_A_PATH = Path("PhiUSIIL_Phishing_URL_Dataset.csv")
DATASET_B_PATH = Path("url_features_extracted1.csv")
MODEL_DIR = Path("common_cross_dataset_results")
OUTPUT_DIR = Path("shap_results")
SAMPLE_SIZE = 5000


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


def normalize_shap_values(values):
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    if hasattr(values, "values"):
        values = values.values
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def global_shap_report(shap_values, features):
    return (
        pd.DataFrame(
            {
                "Feature": COMMON_FEATURES,
                "Mean_Absolute_SHAP": abs(shap_values).mean(axis=0),
                "Mean_SHAP": shap_values.mean(axis=0),
            }
        )
        .sort_values("Mean_Absolute_SHAP", ascending=False)
        .reset_index(drop=True)
    )


def local_shap_report(urls, labels, features, shap_values, predictions):
    rows = []
    for row_index in range(len(features)):
        order = abs(shap_values[row_index]).argsort()[::-1][:5]
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "URL": urls.iloc[row_index],
                    "Actual_Label": int(labels.iloc[row_index]),
                    "Predicted_Label": int(predictions[row_index]),
                    "Rank": rank,
                    "Feature": COMMON_FEATURES[feature_index],
                    "Feature_Value": features.iloc[row_index, feature_index],
                    "SHAP_Value": shap_values[row_index, feature_index],
                    "Direction": "toward_phishing"
                    if shap_values[row_index, feature_index] > 0
                    else "toward_legitimate",
                }
            )
    return pd.DataFrame(rows)


def main():
    dataset_a = load_clean(DATASET_A_PATH, "label")
    dataset_b = load_clean(DATASET_B_PATH, "ClassLabel")
    _, dataset_a_test = train_test_split(
        dataset_a,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=dataset_a["label"],
    )
    dataset_a_test_features = extract_features(dataset_a_test["URL"])

    dataset_a_domains = set(dataset_a["domain"])
    dataset_b_unseen = dataset_b[~dataset_b["domain"].isin(dataset_a_domains)].copy()
    dataset_b_sample = dataset_b_unseen.sample(
        n=min(SAMPLE_SIZE, len(dataset_b_unseen)),
        random_state=RANDOM_STATE,
    )
    dataset_b_sample_features = extract_features(dataset_b_sample["URL"])

    model = joblib.load(MODEL_DIR / "xgboost.pkl")
    explainer = shap.TreeExplainer(model)

    datasets = {
        "dataset_a_test": (
            dataset_a_test["URL"],
            dataset_a_test["label"],
            dataset_a_test_features,
        ),
        "dataset_b_domain_disjoint": (
            dataset_b_sample["URL"],
            dataset_b_sample["label"],
            dataset_b_sample_features,
        ),
    }

    summary = {
        "model": "XGBoost",
        "feature_count": len(COMMON_FEATURES),
        "sample_size_per_evaluation": SAMPLE_SIZE,
        "dataset_a_test_rows": len(dataset_a_test),
        "dataset_b_domain_disjoint_rows": len(dataset_b_unseen),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)

    native_importance = pd.DataFrame(
        {
            "Feature": COMMON_FEATURES,
            "Gain_Importance": model.feature_importances_,
        }
    ).sort_values("Gain_Importance", ascending=False)
    native_importance.to_csv(OUTPUT_DIR / "xgboost_native_importance.csv", index=False)

    for name, (urls, labels, features) in datasets.items():
        shap_values = normalize_shap_values(explainer.shap_values(features))
        predictions = model.predict(features)
        global_report = global_shap_report(shap_values, features)
        global_report.to_csv(OUTPUT_DIR / f"{name}_global_shap.csv", index=False)
        local_report = local_shap_report(
            urls.reset_index(drop=True),
            labels.reset_index(drop=True),
            features.reset_index(drop=True),
            shap_values,
            predictions,
        )
        local_report.to_csv(OUTPUT_DIR / f"{name}_local_shap_top5.csv", index=False)
        summary[f"{name}_mean_abs_shap_top5"] = global_report.head(5).to_dict("records")

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("XGBoost native importance:")
    print(native_importance.to_string(index=False))
    for name in datasets:
        print(f"\n{name} global SHAP ranking:")
        print(pd.read_csv(OUTPUT_DIR / f"{name}_global_shap.csv").head(10).to_string(index=False))
    print(f"\nSaved SHAP analysis in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
