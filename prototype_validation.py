from pathlib import Path

import pandas as pd

from common_url_features import extract_features
from prototype import URLPhishingPrototype

DATASET_A_PATH = Path("PhiUSIIL_Phishing_URL_Dataset.csv")
OUTPUT_DIR = Path("prototype_validation_results")


def load_labeled_urls():
    dataset = pd.read_csv(DATASET_A_PATH).dropna(subset=["URL", "label"]).copy()
    dataset = dataset.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    dataset["expected"] = dataset["label"].map({0: "PHISHING", 1: "LEGITIMATE"})
    features = extract_features(dataset["URL"])
    return pd.concat([dataset, features], axis=1)


def select_cases(dataset):
    phishing = dataset[dataset["expected"] == "PHISHING"]
    legitimate = dataset[dataset["expected"] == "LEGITIMATE"]
    candidate_groups = [
        ("Legitimate", legitimate.sort_values("url_length")),
        ("Phishing", phishing.sort_values("url_length")),
        ("IP-based phishing", phishing[phishing["is_domain_ip"] == 1]),
        ("Long phishing URL", phishing.sort_values("url_length", ascending=False)),
        ("Digit-heavy phishing URL", phishing.sort_values("digit_count", ascending=False)),
        (
            "Suspicious structure",
            phishing.assign(
                structure_score=phishing["slash_count"]
                + phishing["digit_count"]
                + phishing["special_char_count"]
            ).sort_values("structure_score", ascending=False),
        ),
    ]
    selected = []
    used_urls = set()
    for category, candidates in candidate_groups:
        for _, row in candidates.iterrows():
            if row["URL"] not in used_urls:
                selected.append((category, row))
                used_urls.add(row["URL"])
                break
    return selected


def main():
    dataset = load_labeled_urls()
    prototype = URLPhishingPrototype()
    rows = []
    for category, row in select_cases(dataset):
        result = prototype.predict(row["URL"])
        top_factors = "; ".join(
            f"{item['feature']} ({item['direction']}, SHAP={item['shap_value']:.3f})"
            for item in result["explanation"][:3]
        )
        rows.append(
            {
                "Category": category,
                "URL": row["URL"],
                "Expected": row["expected"],
                "Prediction": result["prediction"],
                "Phishing_Probability": result["phishing_probability"],
                "Correct": result["prediction"] == row["expected"],
                "Top_SHAP_Factors": top_factors,
                "slash_count": result["features"]["slash_count"],
                "is_https": result["features"]["is_https"],
                "digit_count": result["features"]["digit_count"],
                "letter_count": result["features"]["letter_count"],
                "url_length": result["features"]["url_length"],
            }
        )

    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "prototype_validation.csv", index=False)
    results.to_markdown(OUTPUT_DIR / "prototype_validation.md", index=False)
    print(results.to_string(index=False))
    print(f"\nAccuracy on curated cases: {results['Correct'].mean():.1%}")
    print(f"Saved validation results in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
