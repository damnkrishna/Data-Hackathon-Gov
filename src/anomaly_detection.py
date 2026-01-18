"""
Anomaly Detection Module for Aadhar Data
Implements multiple anomaly detection methods:
1. Statistical (Z-score, IQR)
2. Time Series (seasonal decomposition)
3. Machine Learning (Isolation Forest, One-Class SVM)
4. Cross-dataset ratio analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


class AadharAnomalyDetector:
    """Comprehensive anomaly detection for Aadhar datasets"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize detector with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.output_dir = Path(self.config['data']['output_dir'])
        self.anomaly_dir = self.output_dir / 'anomalies'
        self.anomaly_dir.mkdir(parents=True, exist_ok=True)
        
        self.z_threshold = self.config['anomaly_detection']['z_score_threshold']
        self.iqr_multiplier = self.config['anomaly_detection']['iqr_multiplier']
        self.spike_threshold = self.config['anomaly_detection']['spike_threshold']
        self.contamination = self.config['anomaly_detection']['contamination']
        
    def load_data(self):
        """Load processed datasets"""
        print("\n" + "="*60)
        print("Loading Processed Datasets")
        print("="*60)
        
        datasets = {}
        for name in ['biometric', 'demographic', 'enrolment', 'combined_aggregated']:
            file_path = self.processed_dir / f"{name}.csv"
            if file_path.exists():
                datasets[name] = pd.read_csv(file_path, parse_dates=['date'])
                print(f"✓ Loaded {name}: {datasets[name].shape}")
        
        return datasets
    
    def statistical_anomalies_zscore(self, df, dataset_name):
        """
        Detect anomalies using Z-score method
        Values beyond z_threshold standard deviations are flagged
        """
        print(f"\n{'='*60}")
        print(f"Z-Score Anomaly Detection - {dataset_name.upper()}")
        print(f"{'='*60}")
        
        df = df.copy()
        
        # Get numeric value columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_cols = [col for col in numeric_cols if col not in ['pincode', 'week', 'has_bio', 'has_demo', 'has_enrol']]
        
        anomalies = pd.DataFrame()
        
        for col in value_cols:
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
            
            # Flag anomalies
            mask = z_scores > self.z_threshold
            col_anomalies = df[mask].copy()
            col_anomalies['anomaly_column'] = col
            col_anomalies['z_score'] = z_scores[mask]
            col_anomalies['value'] = df.loc[mask, col]
            
            anomalies = pd.concat([anomalies, col_anomalies], ignore_index=True)
            
            print(f"{col}: {mask.sum()} anomalies ({mask.sum()/len(df)*100:.2f}%)")
        
        # Save anomalies
        if len(anomalies) > 0:
            output_path = self.anomaly_dir / f'{dataset_name}_zscore_anomalies.csv'
            anomalies.to_csv(output_path, index=False)
            print(f"\n✓ Saved {len(anomalies)} z-score anomalies to: {output_path.name}")
        
        return anomalies
    
    def statistical_anomalies_iqr(self, df, dataset_name):
        """
        Detect anomalies using IQR (Interquartile Range) method
        Values outside [Q1 - k*IQR, Q3 + k*IQR] are flagged
        """
        print(f"\n{'='*60}")
        print(f"IQR Anomaly Detection - {dataset_name.upper()}")
        print(f"{'='*60}")
        
        df = df.copy()
        
        # Get numeric value columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_cols = [col for col in numeric_cols if col not in ['pincode', 'week', 'has_bio', 'has_demo', 'has_enrol']]
        
        anomalies = pd.DataFrame()
        
        for col in value_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            
            # Flag anomalies
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            col_anomalies = df[mask].copy()
            col_anomalies['anomaly_column'] = col
            col_anomalies['value'] = df.loc[mask, col]
            col_anomalies['iqr_lower'] = lower_bound
            col_anomalies['iqr_upper'] = upper_bound
            
            anomalies = pd.concat([anomalies, col_anomalies], ignore_index=True)
            
            print(f"{col}: {mask.sum()} anomalies ({mask.sum()/len(df)*100:.2f}%)")
            print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Save anomalies
        if len(anomalies) > 0:
            output_path = self.anomaly_dir / f'{dataset_name}_iqr_anomalies.csv'
            anomalies.to_csv(output_path, index=False)
            print(f"\n✓ Saved {len(anomalies)} IQR anomalies to: {output_path.name}")
        
        return anomalies
    
    def temporal_spike_detection(self, df, dataset_name):
        """
        Detect sudden spikes in daily aggregations
        Flags days with values > spike_threshold times the median
        """
        print(f"\n{'='*60}")
        print(f"Temporal Spike Detection - {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Get numeric value columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_cols = [col for col in numeric_cols if col not in ['pincode', 'week', 'has_bio', 'has_demo', 'has_enrol']]
        
        # Daily aggregation
        daily = df.groupby('date')[value_cols].sum()
        
        spikes = []
        
        for col in value_cols:
            median_val = daily[col].median()
            
            # Flag spikes
            mask = daily[col] > (median_val * self.spike_threshold)
            spike_dates = daily[mask]
            
            for date, value in spike_dates[col].items():
                spikes.append({
                    'date': date,
                    'column': col,
                    'value': value,
                    'median': median_val,
                    'spike_ratio': value / median_val if median_val > 0 else np.inf
                })
            
            if mask.sum() > 0:
                print(f"{col}: {mask.sum()} spike days detected")
                for date, value in spike_dates[col].head().items():
                    ratio = value / median_val if median_val > 0 else np.inf
                    print(f"  {date.date()}: {value:,.0f} ({ratio:.1f}x median)")
        
        spikes_df = pd.DataFrame(spikes)
        
        if len(spikes_df) > 0:
            output_path = self.anomaly_dir / f'{dataset_name}_temporal_spikes.csv'
            spikes_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved {len(spikes_df)} temporal spikes to: {output_path.name}")
        
        return spikes_df
    
    def geographic_outliers(self, df, dataset_name):
        """
        Detect geographic outliers - locations with unusually high/low values
        Per-state analysis to handle regional variations
        """
        print(f"\n{'='*60}")
        print(f"Geographic Outlier Detection - {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Get numeric value columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_cols = [col for col in numeric_cols if col not in ['pincode', 'week', 'has_bio', 'has_demo', 'has_enrol']]
        
        # Aggregate by location
        location_agg = df.groupby(['state', 'district', 'pincode'])[value_cols].sum()
        
        outliers = []
        
        # Per-state analysis to account for regional differences
        for state in df['state'].unique():
            state_data = location_agg.loc[state] if state in location_agg.index else pd.DataFrame()
            
            if len(state_data) < 10:  # Skip states with too few locations
                continue
            
            for col in value_cols:
                if col not in state_data.columns:
                    continue
                    
                # IQR method per state
                Q1 = state_data[col].quantile(0.25)
                Q3 = state_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                upper_bound = Q3 + self.iqr_multiplier * IQR
                
                # Flag high outliers
                state_outliers = state_data[state_data[col] > upper_bound]
                
                for idx, row in state_outliers.iterrows():
                    district, pincode = idx if isinstance(idx, tuple) else (idx, None)
                    outliers.append({
                        'state': state,
                        'district': district,
                        'pincode': pincode,
                        'column': col,
                        'value': row[col],
                        'state_median': state_data[col].median(),
                        'state_upper_bound': upper_bound
                    })
        
        outliers_df = pd.DataFrame(outliers)
        
        if len(outliers_df) > 0:
            print(f"Detected {len(outliers_df)} geographic outliers")
            print(f"\nTop 10 outliers:")
            print(outliers_df.nlargest(10, 'value')[['state', 'district', 'column', 'value']])
            
            output_path = self.anomaly_dir / f'{dataset_name}_geographic_outliers.csv'
            outliers_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved to: {output_path.name}")
        
        return outliers_df
    
    def ml_isolation_forest(self, df, dataset_name):
        """
        Machine Learning anomaly detection using Isolation Forest
        Unsupervised outlier detection on multivariate data
        """
        print(f"\n{'='*60}")
        print(f"Isolation Forest Detection - {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Get numeric value columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_cols = [col for col in numeric_cols if col not in ['pincode', 'week', 'has_bio', 'has_demo', 'has_enrol']]
        
        if len(value_cols) == 0:
            print("No suitable columns for ML detection")
            return pd.DataFrame()
        
        # Prepare features
        X = df[value_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        print(f"Training Isolation Forest (contamination={self.contamination})...")
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        predictions = iso_forest.fit_predict(X_scaled)
        
        # -1 indicates anomaly, 1 indicates normal
        anomaly_mask = predictions == -1
        
        anomalies = df[anomaly_mask].copy()
        anomalies['anomaly_score'] = iso_forest.decision_function(X_scaled)[anomaly_mask]
        
        print(f"Detected {anomaly_mask.sum()} anomalies ({anomaly_mask.sum()/len(df)*100:.2f}%)")
        
        if len(anomalies) > 0:
            output_path = self.anomaly_dir / f'{dataset_name}_isolation_forest.csv'
            anomalies.to_csv(output_path, index=False)
            print(f"✓ Saved to: {output_path.name}")
        
        return anomalies
    
    def cross_dataset_ratio_anomalies(self, combined_df):
        """
        Detect anomalies in biometric/demographic ratios
        Unusual ratios indicate potential fraud or system issues
        """
        print(f"\n{'='*60}")
        print("Cross-Dataset Ratio Anomaly Detection")
        print(f"{'='*60}")
        
        # Filter records with both bio and demo data
        subset = combined_df[(combined_df['has_bio'] == 1) & (combined_df['has_demo'] == 1)].copy()
        
        if len(subset) == 0:
            print("No records with both biometric and demographic data")
            return pd.DataFrame()
        
        # Calculate totals
        subset['bio_total'] = subset['bio_bio_age_5_17'] + subset['bio_bio_age_17_']
        subset['demo_total'] = subset['demo_demo_age_5_17'] + subset['demo_demo_age_17_']
        
        # Remove zeros
        subset = subset[(subset['bio_total'] > 0) & (subset['demo_total'] > 0)]
        
        # Calculate ratio
        subset['bio_demo_ratio'] = subset['bio_total'] / subset['demo_total']
        
        # Detect anomalies using IQR
        Q1 = subset['bio_demo_ratio'].quantile(0.25)
        Q3 = subset['bio_demo_ratio'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.iqr_multiplier * IQR
        upper_bound = Q3 + self.iqr_multiplier * IQR
        
        mask = (subset['bio_demo_ratio'] < lower_bound) | (subset['bio_demo_ratio'] > upper_bound)
        
        anomalies = subset[mask].copy()
        
        print(f"Detected {len(anomalies)} ratio anomalies ({len(anomalies)/len(subset)*100:.2f}%)")
        print(f"Ratio bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        if len(anomalies) > 0:
            print(f"\nTop 10 extreme ratios:")
            top_anomalies = anomalies.nlargest(10, 'bio_demo_ratio')[
                ['date', 'state', 'district', 'bio_total', 'demo_total', 'bio_demo_ratio']
            ]
            print(top_anomalies)
            
            output_path = self.anomaly_dir / 'cross_dataset_ratio_anomalies.csv'
            anomalies.to_csv(output_path, index=False)
            print(f"\n✓ Saved to: {output_path.name}")
        
        return anomalies
    
    def data_quality_issues(self, datasets):
        """
        Identify data quality issues like inconsistent naming, missing values
        """
        print(f"\n{'='*60}")
        print("Data Quality Issue Detection")
        print(f"{'='*60}")
        
        issues = []
        
        for name, df in datasets.items():
            if name == 'combined_aggregated':
                continue
            
            # State name inconsistencies
            state_variants = df.groupby('state').size().sort_values(ascending=True)
            
            # Find potential duplicates (case-insensitive)
            state_lower = df['state'].str.lower().str.strip()
            state_counts = state_lower.value_counts()
            
            for state_key, count in state_counts.items():
                variants = df[state_lower == state_key]['state'].unique()
                if len(variants) > 1:
                    issues.append({
                        'dataset': name,
                        'issue_type': 'state_name_inconsistency',
                        'state_variants': ', '.join(sorted(variants)),
                        'count': count
                    })
        
        issues_df = pd.DataFrame(issues)
        
        if len(issues_df) > 0:
            print(f"Detected {len(issues_df)} data quality issues")
            print("\nState name inconsistencies:")
            for _, row in issues_df.head(10).iterrows():
                print(f"  {row['dataset']}: {row['state_variants']}")
            
            output_path = self.anomaly_dir / 'data_quality_issues.csv'
            issues_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved to: {output_path.name}")
        
        return issues_df
    
    def generate_anomaly_summary(self):
        """Generate summary report of all detected anomalies"""
        print(f"\n{'='*60}")
        print("Generating Anomaly Summary Report")
        print(f"{'='*60}")
        
        summary_path = self.anomaly_dir / 'anomaly_summary.txt'
        
        # Count anomaly files
        anomaly_files = list(self.anomaly_dir.glob('*.csv'))
        
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("AADHAR ANOMALY DETECTION - SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total anomaly files generated: {len(anomaly_files)}\n\n")
            
            for file_path in sorted(anomaly_files):
                df = pd.read_csv(file_path)
                f.write(f"\n{file_path.name}\n")
                f.write(f"  Anomalies detected: {len(df)}\n")
                
                if 'date' in df.columns:
                    f.write(f"  Date range: {df['date'].min()} to {df['date'].max()}\n")
                
                if 'state' in df.columns:
                    f.write(f"  Affected states: {df['state'].nunique()}\n")
                    f.write(f"  Top states: {', '.join(df['state'].value_counts().head(5).index.tolist())}\n")
        
        print(f"✓ Saved summary to: {summary_path.name}")
    
    def run_all_detections(self):
        """Run all anomaly detection methods"""
        print("\n" + "="*60)
        print("AADHAR ANOMALY DETECTION")
        print("="*60)
        
        # Load data
        datasets = self.load_data()
        
        # Run detection on individual datasets
        for name in ['biometric', 'demographic', 'enrolment']:
            if name in datasets:
                df = datasets[name]
                
                # Statistical methods
                self.statistical_anomalies_zscore(df, name)
                self.statistical_anomalies_iqr(df, name)
                
                # Temporal analysis
                self.temporal_spike_detection(df, name)
                
                # Geographic analysis
                self.geographic_outliers(df, name)
                
                # ML method
                self.ml_isolation_forest(df, name)
        
        # Cross-dataset analysis
        if 'combined_aggregated' in datasets:
            self.cross_dataset_ratio_anomalies(datasets['combined_aggregated'])
        
        # Data quality issues
        self.data_quality_issues(datasets)
        
        # Generate summary
        self.generate_anomaly_summary()
        
        print("\n" + "="*60)
        print("ANOMALY DETECTION COMPLETE!")
        print("="*60)
        print(f"\nAll anomalies saved to: {self.anomaly_dir}")
        print("\nNext steps:")
        print("  1. Review anomaly files in output/anomalies/")
        print("  2. Launch dashboard: streamlit run src/visualization_dashboard.py")


def main():
    """Main execution function"""
    detector = AadharAnomalyDetector(config_path='config.yaml')
    detector.run_all_detections()


if __name__ == "__main__":
    main()
