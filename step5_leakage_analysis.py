import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from common_url_features import COMMON_FEATURES, extract_features

DATASET_A_PATH = Path("PhiUSIIL_Phishing_URL_Dataset.csv")
DATASET_B_PATH = Path("url_features_extracted1.csv")
OUTPUT_DIR = Path("leakage_analysis")


def normalize_domain(url):
    text = str(url).strip()
    parsed = urlparse(text if "://" in text else "http://" + text)
    return (parsed.hostname or "").lower().rstrip(".")


def load_clean(path, label_column):
    df = pd.read_csv(path).dropna(subset=["URL", label_column]).copy()
    df = df.drop_duplicates(subset=["URL"]).copy()
    raw_labels = set(df[label_column].astype(int).unique())
    if raw_labels != {0, 1}:
        raise ValueError(f"{path} labels must be 0 and 1, found {sorted(raw_labels)}")
    df["phishing_label"] = 1 - df[label_column].astype(int)
    df["domain"] = df["URL"].map(normalize_domain)
    return df


def feature_leakage_report(features, labels):
    rows = []
    mutual_information = mutual_info_classif(
        features,
        labels,
        random_state=42,
        discrete_features="auto",
    )
    for index, feature in enumerate(COMMON_FEATURES):
        legitimate = features.loc[labels == 0, feature]
        phishing = features.loc[labels == 1, feature]
        pooled_std = features[feature].std()
        rows.append(
            {
                "Feature": feature,
                "Absolute_Pearson_Correlation": abs(features[feature].corr(labels)),
                "Mutual_Information": mutual_information[index],
                "Legitimate_Mean": legitimate.mean(),
                "Phishing_Mean": phishing.mean(),
                "Mean_Difference": phishing.mean() - legitimate.mean(),
                "Pooled_Std": pooled_std,
                "Constant": features[feature].nunique() <= 1,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Mutual_Information", "Absolute_Pearson_Correlation"],
        ascending=False,
    )


def domain_label_report(dataset_a, dataset_b):
    domains_a = set(dataset_a["domain"])
    domains_b = set(dataset_b["domain"])
    shared_domains = domains_a & domains_b
    rows = []
    for domain in shared_domains:
        labels_a = sorted(dataset_a.loc[dataset_a["domain"] == domain, "phishing_label"].unique())
        labels_b = sorted(dataset_b.loc[dataset_b["domain"] == domain, "phishing_label"].unique())
        rows.append(
            {
                "domain": domain,
                "dataset_a_labels": ",".join(map(str, labels_a)),
                "dataset_b_labels": ",".join(map(str, labels_b)),
                "cross_dataset_label_conflict": bool(set(labels_a) != set(labels_b)),
                "dataset_a_url_count": int((dataset_a["domain"] == domain).sum()),
                "dataset_b_url_count": int((dataset_b["domain"] == domain).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["cross_dataset_label_conflict", "dataset_a_url_count", "dataset_b_url_count"],
        ascending=False,
    )


def main():
    dataset_a = load_clean(DATASET_A_PATH, "label")
    dataset_b = load_clean(DATASET_B_PATH, "ClassLabel")

    urls_a = set(dataset_a["URL"])
    urls_b = set(dataset_b["URL"])
    overlapping_urls = urls_a & urls_b
    overlap_a = dataset_a[dataset_a["URL"].isin(overlapping_urls)][["URL", "phishing_label"]]
    overlap_b = dataset_b[dataset_b["URL"].isin(overlapping_urls)][["URL", "phishing_label"]]
    overlap_labels = overlap_a.merge(overlap_b, on="URL", suffixes=("_A", "_B"))
    overlap_labels["label_conflict"] = (
        overlap_labels["phishing_label_A"] != overlap_labels["phishing_label_B"]
    )

    shared_domain_report = domain_label_report(dataset_a, dataset_b)
    features_a = extract_features(dataset_a["URL"])
    feature_report_a = feature_leakage_report(features_a, dataset_a["phishing_label"])
    features_b = extract_features(dataset_b["URL"])
    feature_report_b = feature_leakage_report(features_b, dataset_b["phishing_label"])

    model_importance_path = Path("common_cross_dataset_results/random_forest_feature_importance.csv")
    model_importance = pd.read_csv(model_importance_path) if model_importance_path.exists() else pd.DataFrame()

    summary = {
        "dataset_a_rows": len(dataset_a),
        "dataset_b_rows": len(dataset_b),
        "exact_url_overlap": len(overlapping_urls),
        "exact_url_overlap_rate_of_a": len(overlapping_urls) / len(dataset_a),
        "exact_url_overlap_rate_of_b": len(overlapping_urls) / len(dataset_b),
        "overlap_label_conflicts": int(overlap_labels["label_conflict"].sum()),
        "dataset_a_unique_domains": len(set(dataset_a["domain"])),
        "dataset_b_unique_domains": len(set(dataset_b["domain"])),
        "shared_domains": len(shared_domain_report),
        "shared_domains_with_label_conflicts": int(
            shared_domain_report["cross_dataset_label_conflict"].sum()
        ),
        "dataset_a_feature_rows_with_missing_values": int(features_a.isna().any(axis=1).sum()),
        "dataset_b_feature_rows_with_missing_values": int(features_b.isna().any(axis=1).sum()),
        "dataset_a_highest_mutual_information_feature": feature_report_a.iloc[0]["Feature"],
        "dataset_b_highest_mutual_information_feature": feature_report_b.iloc[0]["Feature"],
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    overlap_labels.to_csv(OUTPUT_DIR / "exact_url_overlap_labels.csv", index=False)
    shared_domain_report.to_csv(OUTPUT_DIR / "shared_domain_report.csv", index=False)
    feature_report_a.to_csv(OUTPUT_DIR / "dataset_a_feature_leakage.csv", index=False)
    feature_report_b.to_csv(OUTPUT_DIR / "dataset_b_feature_leakage.csv", index=False)
    if not model_importance.empty:
        model_importance.to_csv(OUTPUT_DIR / "random_forest_importance_snapshot.csv", index=False)

    print("Leakage and overlap summary:")
    print(json.dumps(summary, indent=2))
    print("\nTop Dataset A feature associations:")
    print(feature_report_a.head(10).to_string(index=False))
    print("\nTop Dataset B feature associations:")
    print(feature_report_b.head(10).to_string(index=False))
    print(f"\nSaved analysis in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
