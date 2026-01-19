"""
STEP 1: High-Confidence Anomaly Intersection

Keeps only anomalies that:
- Appear in MULTIPLE strong detection methods
- Persist across MULTIPLE months
"""

import pandas as pd
from pathlib import Path
import yaml


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_anomaly_files(anomaly_dir: Path):
    files = list(anomaly_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No anomaly CSV files found")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def normalize_columns(df):
    """
    Ensure consistent column names across anomaly files
    """
    rename_map = {
        "State": "state",
        "District": "district",
        "Month": "month",
        "Date": "date"
    }

    df = df.rename(columns=rename_map)

    # Create month column if missing
    if "month" not in df.columns and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").astype(str)

    required_cols = {"state", "district", "month"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def classify_method(file_name):
    if "isolation" in file_name:
        return "isolation_forest"
    if "zscore" in file_name:
        return "zscore"
    if "cross" in file_name:
        return "cross_dataset"
    return "other"

def build_high_confidence_anomalies(df):
    """
    Identifies districts flagged by multiple models across multiple months.
    """
    # 1. Map filenames to method names
    df["method"] = df["source_file"].apply(classify_method)

    # 2. Filter for strong signals only
    strong_methods = ["isolation_forest", "zscore", "cross_dataset"]
    df_filtered = df[df["method"].isin(strong_methods)].copy()

    # 3. Aggregate directly to District level
    # We count UNIQUE months and UNIQUE methods per district
    high_conf = (
        df_filtered.groupby(["state", "district"])
        .agg(
            active_months=("month", "nunique"),
            max_methods=("method", "nunique")
        )
        .reset_index()
    )

    # 4. Apply High-Confidence Logic
    # District must have 2+ different methods AND 2+ different months of anomalies
    mask = (high_conf["max_methods"] >= 2) & (high_conf["active_months"] >= 2)
    
    # 5. Sort by severity (most methods first, then most months)
    return high_conf[mask].sort_values(
        by=["max_methods", "active_months"], 
        ascending=False
    )
def main():
    config = load_config()

    output_dir = Path(config["data"]["output_dir"])
    anomaly_dir = output_dir / "anomalies"
    high_conf_dir = output_dir / "high_confidence"
    high_conf_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 1: HIGH-CONFIDENCE ANOMALY INTERSECTION")
    print("=" * 60)

    # Load and clean the raw data
    raw_anomalies = load_anomaly_files(anomaly_dir)
    raw_anomalies = normalize_columns(raw_anomalies)
    
    # Ensure we don't process the exact same row twice from different files
    raw_anomalies = raw_anomalies.drop_duplicates()

    # Generate the high confidence list
    high_conf = build_high_confidence_anomalies(raw_anomalies)

    # Save to CSV
    output_path = high_conf_dir / "high_confidence_anomalies.csv"
    high_conf.to_csv(output_path, index=False)

    # Clean terminal reporting
    print(f"\n✓ High-confidence districts identified: {len(high_conf)}")
    print(f"✓ Saved to: {output_path}")
    print("\n--- TOP 10 ANOMALOUS DISTRICTS ---")
    
    # to_string(index=False) makes the output look like a clean table
    if not high_conf.empty:
        print(high_conf.head(10).to_string(index=False))
    else:
        print("No districts met the high-confidence threshold.")
        
    print("=" * 60)

if __name__ == "__main__":
    main()