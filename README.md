# Aadhar Anomaly Detection System

Detect meaningful patterns, trends, and anomalies in Aadhar biometric, demographic, and enrolment data to support informed decision-making and system improvements.

## 🚀 Quick Start


### **Phase 1: Environment Setup**

Ensure all necessary libraries are installed to avoid runtime errors during script execution.

```bash
pip install -r requirements.txt

```

### **Phase 2: Data Loading & Validation**

This phase prepares the raw Aadhaar datasets (Biometric, Demographic, and Enrolment) for analysis.

1. **Load Data:** Aggregates raw CSV chunks from `raw/` into `processed/`.
```bash
python src/data_loader.py

```


2. **Validate Data:** Performs integrity checks to ensure the cleaned data is complete and accurate.
```bash
python src/data_validation.py

```



### **Phase 3: Core Anomaly Detection**

This phase applies statistical and machine learning models to identify outliers.

1. **Initial Detection:** Runs primary models such as Isolation Forest and Z-score.
```bash
python src/anomaly_detection.py

```


2. **Refine Confidence:** Filters overlapping signals from multiple models to reduce false positives.
```bash
python src/high_confidence.py

```



### **Phase 4: Risk Scoring & Ranking**

This phase quantifies the severity of detected anomalies to prioritize administrative action.

1. **Calculate Scores:** Assigns risk weights based on deviation magnitude and frequency.
```bash
python src/risk_scoring.py

```


2. **District Prioritization:** Ranks districts by their overall risk profiles.
```bash
python src/district_case_builder.py

```



### **Phase 5: Visual Reporting**

Generate judge-ready visualizations and launch the interactive dashboard.

1. **Generate Plots:** Creates district-specific visual evidence (e.g., `Nashik_bio_demo_ratio.png`).
```bash
python src/district_case_plot.py

```


2. **Launch Dashboard:** Review all findings in a real-time interactive environment.
```bash
streamlit run src/visualization_dashboard.py

```




## 📁 Project Structure

```
Data-Hackathon-Gov/
├── raw/                           # Raw data (CSV chunks)
│   ├── api_data_aadhar_biometric/
│   ├── api_data_aadhar_demographic/
│   └── api_data_aadhar_enrolment/
├── processed/                     # Cleaned datasets
├── output/                        # Analysis results
├── src/                           # Source code
│   ├── data_loader.py            # Data loading & cleaning
│   ├── eda_analysis.py           # Exploratory analysis
│   ├── feature_engineering.py    # Feature creation
│   ├── anomaly_detection.py      # Anomaly detection models
│   ├── pattern_analysis.py       # Pattern mining
│   ├── visualization_dashboard.py # Interactive dashboard
│   └── report_generator.py       # Report creation
├── config.yaml                    # Configuration
└── requirements.txt               # Dependencies
```

## 📊 Datasets

### 1. Biometric Data (~1.86M records)
- Biometric authentication by age groups (5-17, 17+)
- Location: state, district, pincode
- Temporal: daily records

### 2. Demographic Data (~2.07M records)
- Demographic authentication by age groups (5-17, 17+)
- Same geographic and temporal structure

### 3. Enrolment Data (~1M records)
- New Aadhar enrolments by age groups (0-5, 5-17, 18+)
- Same geographic and temporal structure

## 🎯 Anomaly Detection Methods

1. **Statistical Methods**
   - Z-score detection
   - IQR (Interquartile Range)

2. **Time Series Methods**
   - Seasonal decomposition
   - ARIMA residuals
   - Prophet anomaly detection

3. **Machine Learning Methods**
   - Isolation Forest
   - Local Outlier Factor (LOF)
   - Autoencoder (Neural Network)

4. **Geographic Methods**
   - Spatial clustering (DBSCAN)
   - Hotspot analysis

5. **Cross-Dataset Analysis**
   - Biometric/Demographic ratio anomalies
   - Coverage gap detection
   - Enrolment vs authentication mismatches

## 🔍 Key Insights

The system identifies:
- **Fraud Indicators**: Suspicious authentication patterns
- **System Issues**: Outages, failures, performance problems
- **Data Quality Issues**: Missing or inconsistent data
- **Infrastructure Gaps**: Underserved regions
- **Capacity Planning**: Peak usage patterns

## 📝 Configuration

Edit `config.yaml` to customize:
- Data paths
- Anomaly detection thresholds
- Feature engineering parameters
- Output preferences

## 🛠️ Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run individual modules
python src/data_loader.py
python src/eda_analysis.py
python src/anomaly_detection.py

# Or run full pipeline
python main.py
```

## 📈 Output

- Processed datasets: `processed/`
- Anomaly reports: `output/anomalies/`
- Visualizations: `output/plots/`
- Summary reports: `output/reports/`
