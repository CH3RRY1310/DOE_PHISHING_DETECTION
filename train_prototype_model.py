from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from common_url_features import extract_features

RANDOM_STATE = 42
DATASET_A_PATH = Path("PhiUSIIL_Phishing_URL_Dataset.csv")
OUTPUT_DIR = Path("prototype_artifacts")
FEATURES = ["slash_count", "is_https", "digit_count", "letter_count", "url_length"]


def main():
    dataset = pd.read_csv(DATASET_A_PATH).dropna(subset=["URL", "label"]).copy()
    dataset = dataset.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    raw_labels = set(dataset["label"].astype(int).unique())
    if raw_labels != {0, 1}:
        raise ValueError(f"Dataset A labels must be 0 and 1, found {sorted(raw_labels)}")

    # Dataset A uses 0=phishing and 1=legitimate; the prototype reports 1=phishing.
    labels = 1 - dataset["label"].astype(int)
    features = extract_features(dataset["URL"])[FEATURES]
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(features, labels)

    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "top5_xgboost.pkl")
    joblib.dump(
        {
            "features": FEATURES,
            "label_convention": "0=legitimate, 1=phishing",
            "source_label_mapping": "Dataset A source: 0=phishing, 1=legitimate",
            "training_rows": len(dataset),
            "random_state": RANDOM_STATE,
        },
        OUTPUT_DIR / "metadata.pkl",
    )
    print(f"Trained top-5 XGBoost on {len(dataset)} cleaned Dataset A URLs")
    print(f"Saved prototype artifacts in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
