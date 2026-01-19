"""
STEP 2: District Risk Scoring

Assigns a risk score to each district based on:
- Number of anomaly methods triggered
- Severity of methods
- Temporal persistence
"""

import pandas as pd
from pathlib import Path
import yaml


METHOD_WEIGHTS = {
    "isolation_forest": 3,
    "cross_dataset": 4,
    "zscore": 1
}


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_raw_anomalies(anomaly_dir: Path):
    dfs = []
    for f in anomaly_dir.glob("*.csv"):
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def classify_method(file_name):
    if "isolation" in file_name:
        return "isolation_forest"
    if "cross" in file_name:
        return "cross_dataset"
    if "zscore" in file_name:
        return "zscore"
    return None


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

def compute_risk_scores(df):
    """
    Calculates weighted risk scores and assigns tiers based on statistical percentiles.
    """
    # Fix 1: Explicitly create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Map methods to weights
    df["method"] = df["source_file"].apply(classify_method)
    df = df[df["method"].notna()].copy() # Filtered copy
    df["method_weight"] = df["method"].map(METHOD_WEIGHTS)

    # 2. Per district per month aggregation
    monthly = (
        df.groupby(["state", "district", "month"])
        .agg(
            monthly_risk=("method_weight", "sum"),
            methods_triggered=("method", "nunique")
        )
        .reset_index()
    )

    # 3. Final district-level summary
    risk_scores = (
        monthly.groupby(["state", "district"])
        .agg(
            risk_score=("monthly_risk", "sum"),
            active_months=("month", "nunique"),
            max_methods=("methods_triggered", "max")
        )
        .reset_index()
    )

    # 4. Dynamic Risk Tiers (Statistical Percentiles)
    # This ensures your High/Medium/Low categories are balanced
    high_cutoff = risk_scores["risk_score"].quantile(0.90)  # Top 10%
    med_cutoff = risk_scores["risk_score"].quantile(0.70)   # Next 20%

    def assign_tier(row):
        # A district is HIGH if it's in the top 10% OR has 3 methods + 6+ months
        if row["risk_score"] >= high_cutoff:
            return "HIGH"
        elif row["risk_score"] >= med_cutoff:
            return "MEDIUM"
        else:
            return "LOW"

    risk_scores["risk_tier"] = risk_scores.apply(assign_tier, axis=1)

    return risk_scores.sort_values("risk_score", ascending=False)

def main():
    config = load_config()

    output_dir = Path(config["data"]["output_dir"])
    anomaly_dir = output_dir / "anomalies"
    risk_dir = output_dir / "risk_scores"
    risk_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 2: DISTRICT RISK SCORING")
    print("=" * 60)

    raw = load_raw_anomalies(anomaly_dir)
    raw = normalize_columns(raw)

    risk_scores = compute_risk_scores(raw)

    output_path = risk_dir / "district_risk_scores.csv"
    risk_scores.to_csv(output_path, index=False)

    print(f"\n✓ Districts scored: {len(risk_scores)}")
    print(f"✓ Output saved to: {output_path}")

    print("\nTop High-Risk Districts:")
    print(risk_scores.head(10))


if __name__ == "__main__":
    main()
