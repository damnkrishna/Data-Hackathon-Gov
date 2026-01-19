"""
STEP 3: District Case Builder

Creates interpretable summaries for high-risk districts:
- Dominant anomaly type
- Persistence vs spike behavior
- Timeline of abnormality
"""

import pandas as pd
from pathlib import Path
import yaml


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def classify_dataset(file_name):
    if "biometric" in file_name:
        return "BIOMETRIC"
    if "demographic" in file_name:
        return "DEMOGRAPHIC"
    if "enrol" in file_name or "enroll" in file_name:
        return "ENROLMENT"
    if "cross" in file_name:
        return "CROSS_DATASET"
    return "OTHER"


def normalize_columns(df):
    rename_map = {
        "State": "state",
        "District": "district",
        "Month": "month",
        "Date": "date"
    }
    df = df.rename(columns=rename_map)

    if "month" not in df.columns and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").astype(str)

    return df


def build_case_summaries(anomalies_df, risk_df):
    anomalies_df["dataset"] = anomalies_df["source_file"].apply(classify_dataset)

    records = []

    for _, row in risk_df.iterrows():
        state = row["state"]
        district = row["district"]

        subset = anomalies_df[
            (anomalies_df["state"] == state) &
            (anomalies_df["district"] == district)
        ]

        if subset.empty:
            continue

        active_months = subset["month"].nunique()
        start_month = subset["month"].min()
        end_month = subset["month"].max()

        dominant_dataset = (
            subset["dataset"]
            .value_counts()
            .idxmax()
        )

        pattern = "PERSISTENT" if active_months > 2 else "SPIKE"

        records.append({
            "state": state,
            "district": district,
            "risk_score": row["risk_score"],
            "risk_tier": row["risk_tier"],
            "dominant_dataset": dominant_dataset,
            "active_months": active_months,
            "start_month": start_month,
            "end_month": end_month,
            "pattern_type": pattern
        })

    return pd.DataFrame(records).sort_values(
        "risk_score", ascending=False
    )


def main():
    config = load_config()
    output_dir = Path(config["data"]["output_dir"])

    anomaly_dir = output_dir / "anomalies"
    risk_path = output_dir / "risk_scores" / "district_risk_scores.csv"
    case_dir = output_dir / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 3: DISTRICT CASE INTERPRETATION")
    print("=" * 60)

    anomalies = []
    for f in anomaly_dir.glob("*.csv"):
        df = pd.read_csv(f)
        df["source_file"] = f.name
        anomalies.append(df)

    anomalies_df = normalize_columns(
        pd.concat(anomalies, ignore_index=True)
    )

    risk_df = pd.read_csv(risk_path)

    case_summaries = build_case_summaries(anomalies_df, risk_df)

    output_path = case_dir / "district_case_summaries.csv"
    case_summaries.to_csv(output_path, index=False)

    print(f"\n✓ District case studies generated: {len(case_summaries)}")
    print(f"✓ Saved to: {output_path}")

    print("\nTop District Case Summaries:")
    print(case_summaries.head(10))


if __name__ == "__main__":
    main()
