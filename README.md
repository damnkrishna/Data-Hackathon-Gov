# Aadhar Anomaly Detection System

Detect meaningful patterns, trends, and anomalies in Aadhar biometric, demographic, and enrolment data to support informed decision-making and system improvements.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Load and Process Data

```bash
python src/data_loader.py
```

This will:
- Load all CSV chunks from `raw/` directory
- Clean and validate data
- Create processed datasets in `processed/` directory
- Generate cross-dataset aggregation

### 3. Run Exploratory Data Analysis

```bash
python src/eda_analysis.py
```

### 4. Detect Anomalies

```bash
python src/anomaly_detection.py
```

### 5. Launch Dashboard

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
