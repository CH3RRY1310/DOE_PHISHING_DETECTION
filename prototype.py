import argparse
from pathlib import Path

import joblib
import pandas as pd
import shap

from common_url_features import extract_features

ARTIFACT_DIR = Path("prototype_artifacts")


class URLPhishingPrototype:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)
        self.model = joblib.load(artifact_dir / "top5_xgboost.pkl")
        metadata = joblib.load(artifact_dir / "metadata.pkl")
        self.features = metadata["features"]
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, url, explanation_count=5):
        feature_frame = extract_features([url])[self.features]
        probability = float(self.model.predict_proba(feature_frame)[0, 1])
        predicted_label = int(probability >= 0.5)
        shap_values = self.explainer.shap_values(feature_frame)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        contributions = []
        order = abs(shap_values[0]).argsort()[::-1][:explanation_count]
        for index in order:
            value = float(shap_values[0, index])
            contributions.append(
                {
                    "feature": self.features[index],
                    "value": float(feature_frame.iloc[0, index]),
                    "shap_value": value,
                    "direction": "toward_phishing" if value > 0 else "toward_legitimate",
                }
            )
        return {
            "url": url,
            "prediction": "PHISHING" if predicted_label else "LEGITIMATE",
            "phishing_probability": probability,
            "features": {
                name: float(feature_frame.iloc[0][name]) for name in self.features
            },
            "explanation": contributions,
        }


def main():
    parser = argparse.ArgumentParser(description="Classify a URL with the top-5 XGBoost prototype")
    parser.add_argument("url", nargs="?", help="URL to classify")
    args = parser.parse_args()
    url = args.url or input("Enter a URL: ").strip()
    result = URLPhishingPrototype().predict(url)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Phishing probability: {result['phishing_probability']:.4f}")
    print("Features:")
    print(pd.Series(result["features"]).to_string())
    print("\nTop SHAP contributions:")
    print(pd.DataFrame(result["explanation"]).to_string(index=False))


if __name__ == "__main__":
    main()
