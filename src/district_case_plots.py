import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
PROCESSED_DIR = Path("processed")
OUTPUT_DIR = Path("output/case_studies/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DISTRICTS = [
    ("Maharashtra", "Nashik"),
    ("Maharashtra", "Pune"),
    ("Maharashtra", "Jalgaon"),
    ("Maharashtra", "Ahmadnagar"),
    ("Maharashtra", "Amravati"),
]

# -----------------------------
# Load data
# -----------------------------
csv_files = list(PROCESSED_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("❌ No CSV files found in processed/")

print("Available processed files:")
for f in csv_files:
    print(" -", f.name)

DATA_PATH = max(csv_files, key=lambda f: f.stat().st_size)
print(f"\n✓ Using data file: {DATA_PATH.name}")

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Detect time column
# -----------------------------
if "date" in df.columns:
    df["time"] = pd.to_datetime(df["date"])
else:
    raise ValueError("❌ No date column found")

print("✓ Using time column: date")

# -----------------------------
# Compute totals (using correct prefixes)
# -----------------------------
bio_cols = [c for c in df.columns if c.startswith("bio_")]
demo_cols = [c for c in df.columns if c.startswith("demo_")]

if not bio_cols or not demo_cols:
    raise ValueError("❌ Biometric or Demographic columns missing")

df["biometric_total"] = df[bio_cols].sum(axis=1)
df["demographic_total"] = df[demo_cols].sum(axis=1)

# Avoid divide-by-zero
df = df[df["demographic_total"] > 0]

# -----------------------------
# Plotting
# -----------------------------
for state, district in TARGET_DISTRICTS:
    ddf = df[(df["state"] == state) & (df["district"] == district)].copy()

    if ddf.empty:
        print(f"⚠️ No data for {district}, {state}")
        continue

    ddf = ddf.sort_values("time")

    # ---- Plot 1: Biometric Volume ----
    plt.figure(figsize=(10, 4))
    plt.plot(ddf["time"], ddf["biometric_total"], marker="o")
    plt.title(f"Biometric Authentication Volume\n{district}, {state}")
    plt.xlabel("Date")
    plt.ylabel("Total Biometric Authentications")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{district}_biometric_volume.png")
    plt.close()

    # ---- Plot 2: Biometric / Demographic Ratio ----
    ddf["bio_demo_ratio"] = ddf["biometric_total"] / ddf["demographic_total"]

    plt.figure(figsize=(10, 4))
    plt.plot(ddf["time"], ddf["bio_demo_ratio"], marker="o")
    plt.title(f"Biometric-to-Demographic Ratio\n{district}, {state}")
    plt.xlabel("Date")
    plt.ylabel("Biometric / Demographic Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{district}_bio_demo_ratio.png")
    plt.close()

    print(f"✓ Plots generated for {district}, {state}")

print("\n✅ All district plots generated successfully.")
