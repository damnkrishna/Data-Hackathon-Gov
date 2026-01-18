"""
Data Loader Module for Aadhar Anomaly Detection
Loads and consolidates biometric, demographic, and enrolment datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import os


class AadharDataLoader:
    """Load and process Aadhar datasets"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize data loader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.date_format = self.config['temporal']['date_format']
        
    def load_dataset_chunks(self, dataset_type):
        """
        Load all CSV chunks for a specific dataset type
        
        Args:
            dataset_type: 'biometric', 'demographic', or 'enrolment'
            
        Returns:
            Consolidated pandas DataFrame
        """
        # Get directory path
        dir_key = f"{dataset_type}_dir"
        dataset_dir = Path(self.config['data'][dir_key])
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Find all CSV files
        csv_files = sorted(list(dataset_dir.glob("*.csv")))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
        
        print(f"\n{'='*60}")
        print(f"Loading {dataset_type.upper()} dataset")
        print(f"{'='*60}")
        print(f"Found {len(csv_files)} CSV files")
        
        # Load and concatenate all chunks
        dfs = []
        total_rows = 0
        
        for csv_file in tqdm(csv_files, desc=f"Loading {dataset_type}"):
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
                total_rows += len(df)
                print(f"  ✓ {csv_file.name}: {len(df):,} rows")
            except Exception as e:
                print(f"  ✗ Error loading {csv_file.name}: {e}")
        
        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        print(f"\nTotal records loaded: {total_rows:,}")
        print(f"Combined dataset shape: {combined_df.shape}")
        
        return combined_df
    
    def clean_dataset(self, df, dataset_type):
        """
        Clean and validate dataset
        
        Args:
            df: Input dataframe
            dataset_type: Dataset type for specific cleaning rules
            
        Returns:
            Cleaned dataframe
        """
        print(f"\n{'='*60}")
        print(f"Cleaning {dataset_type.upper()} dataset")
        print(f"{'='*60}")
        
        initial_rows = len(df)
        
        # Convert date column to datetime
        date_col = self.config['temporal']['date_column']
        df[date_col] = pd.to_datetime(df[date_col], format=self.date_format, errors='coerce')
        
        # Check for parsing errors
        invalid_dates = df[date_col].isna().sum()
        if invalid_dates > 0:
            print(f"⚠ Warning: {invalid_dates} invalid dates found and removed")
            df = df.dropna(subset=[date_col])
        
        # Remove duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"⚠ Warning: {duplicates} duplicate rows found and removed")
            df = df.drop_duplicates().copy()  # Create explicit copy to avoid warning
        
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_counts = df[numeric_cols].isna().sum()
        
        if missing_counts.sum() > 0:
            print(f"\nMissing values in numeric columns:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")
            
            # Fill missing numeric values with 0 (assuming no activity)
            df[numeric_cols] = df[numeric_cols].fillna(0)
            print("  → Filled with 0 (assuming no activity)")
        
        # Validate geographic columns
        geo_cols = ['state', 'district', 'pincode']
        for col in geo_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    print(f"⚠ Warning: {missing} missing values in {col}")
                    df = df.dropna(subset=[col])
        
        # Convert numeric columns to appropriate types
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        
        print(f"\nCleaning summary:")
        print(f"  Initial rows: {initial_rows:,}")
        print(f"  Final rows: {final_rows:,}")
        print(f"  Rows removed: {rows_removed:,} ({rows_removed/initial_rows*100:.2f}%)")
        
        return df
    
    def save_processed_dataset(self, df, dataset_type):
        """Save processed dataset to CSV"""
        output_path = self.processed_dir / f"{dataset_type}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved processed dataset to: {output_path}")
        return output_path
    
    def get_dataset_summary(self, df, dataset_type):
        """Generate summary statistics for dataset"""
        print(f"\n{'='*60}")
        print(f"{dataset_type.upper()} Dataset Summary")
        print(f"{'='*60}")
        
        # Basic info
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Date range
        date_col = self.config['temporal']['date_column']
        print(f"\nDate range: {df[date_col].min()} to {df[date_col].max()}")
        print(f"Number of unique dates: {df[date_col].nunique()}")
        
        # Geographic coverage
        print(f"\nGeographic coverage:")
        print(f"  States: {df['state'].nunique()}")
        print(f"  Districts: {df['district'].nunique()}")
        print(f"  Pincodes: {df['pincode'].nunique()}")
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\nNumeric columns statistics:")
        print(df[numeric_cols].describe())
        
        return {
            'shape': df.shape,
            'date_range': (df[date_col].min(), df[date_col].max()),
            'unique_dates': df[date_col].nunique(),
            'unique_states': df['state'].nunique(),
            'unique_districts': df['district'].nunique(),
            'unique_pincodes': df['pincode'].nunique()
        }
    
    def create_aggregated_dataset(self, bio_df, demo_df, enrol_df):
        """
        Create aggregated cross-dataset for relationship analysis
        
        Args:
            bio_df: Biometric dataframe
            demo_df: Demographic dataframe
            enrol_df: Enrolment dataframe
            
        Returns:
            Aggregated dataframe with all datasets merged
        """
        print(f"\n{'='*60}")
        print("Creating Cross-Dataset Aggregation")
        print(f"{'='*60}")
        
        # Aggregation keys
        agg_keys = ['date', 'state', 'district', 'pincode']
        
        # Aggregate biometric
        bio_agg = bio_df.groupby(agg_keys).sum().reset_index()
        bio_agg = bio_agg.add_prefix('bio_').rename(columns={
            'bio_date': 'date',
            'bio_state': 'state',
            'bio_district': 'district',
            'bio_pincode': 'pincode'
        })
        print(f"Biometric aggregated: {bio_agg.shape}")
        
        # Aggregate demographic
        demo_agg = demo_df.groupby(agg_keys).sum().reset_index()
        demo_agg = demo_agg.add_prefix('demo_').rename(columns={
            'demo_date': 'date',
            'demo_state': 'state',
            'demo_district': 'district',
            'demo_pincode': 'pincode'
        })
        print(f"Demographic aggregated: {demo_agg.shape}")
        
        # Aggregate enrolment
        enrol_agg = enrol_df.groupby(agg_keys).sum().reset_index()
        enrol_agg = enrol_agg.add_prefix('enrol_').rename(columns={
            'enrol_date': 'date',
            'enrol_state': 'state',
            'enrol_district': 'district',
            'enrol_pincode': 'pincode'
        })
        print(f"Enrolment aggregated: {enrol_agg.shape}")
        
        # Outer join to keep all records
        combined = bio_agg.merge(demo_agg, on=agg_keys, how='outer')
        combined = combined.merge(enrol_agg, on=agg_keys, how='outer')
        
        # Fill NaN with 0 (no activity in that dataset)
        combined = combined.fillna(0)
        
        # Add mismatch indicators
        combined['has_bio'] = (combined[[c for c in combined.columns if c.startswith('bio_')]].sum(axis=1) > 0).astype(int)
        combined['has_demo'] = (combined[[c for c in combined.columns if c.startswith('demo_')]].sum(axis=1) > 0).astype(int)
        combined['has_enrol'] = (combined[[c for c in combined.columns if c.startswith('enrol_')]].sum(axis=1) > 0).astype(int)
        
        print(f"\nCombined dataset shape: {combined.shape}")
        print(f"Records with all three datasets: {((combined['has_bio'] + combined['has_demo'] + combined['has_enrol']) == 3).sum()}")
        print(f"Records with only biometric: {((combined['has_bio'] == 1) & (combined['has_demo'] == 0) & (combined['has_enrol'] == 0)).sum()}")
        print(f"Records with only demographic: {((combined['has_bio'] == 0) & (combined['has_demo'] == 1) & (combined['has_enrol'] == 0)).sum()}")
        print(f"Records with only enrolment: {((combined['has_bio'] == 0) & (combined['has_demo'] == 0) & (combined['has_enrol'] == 1)).sum()}")
        
        # Save combined dataset
        output_path = self.processed_dir / "combined_aggregated.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n✓ Saved combined dataset to: {output_path}")
        
        return combined
    
    def load_all_datasets(self):
        """
        Load all three datasets and create both individual and combined versions
        
        Returns:
            Dictionary with all loaded datasets
        """
        # Load individual datasets
        biometric_df = self.load_dataset_chunks('biometric')
        biometric_df = self.clean_dataset(biometric_df, 'biometric')
        bio_summary = self.get_dataset_summary(biometric_df, 'biometric')
        bio_path = self.save_processed_dataset(biometric_df, 'biometric')
        
        demographic_df = self.load_dataset_chunks('demographic')
        demographic_df = self.clean_dataset(demographic_df, 'demographic')
        demo_summary = self.get_dataset_summary(demographic_df, 'demographic')
        demo_path = self.save_processed_dataset(demographic_df, 'demographic')
        
        enrolment_df = self.load_dataset_chunks('enrolment')
        enrolment_df = self.clean_dataset(enrolment_df, 'enrolment')
        enrol_summary = self.get_dataset_summary(enrolment_df, 'enrolment')
        enrol_path = self.save_processed_dataset(enrolment_df, 'enrolment')
        
        # Create combined aggregated dataset
        combined_df = self.create_aggregated_dataset(biometric_df, demographic_df, enrolment_df)
        
        return {
            'biometric': {
                'df': biometric_df,
                'summary': bio_summary,
                'path': bio_path
            },
            'demographic': {
                'df': demographic_df,
                'summary': demo_summary,
                'path': demo_path
            },
            'enrolment': {
                'df': enrolment_df,
                'summary': enrol_summary,
                'path': enrol_path
            },
            'combined': {
                'df': combined_df,
                'path': self.processed_dir / "combined_aggregated.csv"
            }
        }


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("AADHAR ANOMALY DETECTION - DATA LOADER")
    print("="*60)
    
    # Initialize loader
    loader = AadharDataLoader(config_path='config.yaml')
    
    # Load all datasets
    datasets = loader.load_all_datasets()
    
    print("\n" + "="*60)
    print("DATA LOADING COMPLETE!")
    print("="*60)
    print("\nProcessed datasets saved to:", loader.processed_dir)
    print("\nNext steps:")
    print("  1. Run EDA analysis: python src/eda_analysis.py")
    print("  2. Engineer features: python src/feature_engineering.py")
    print("  3. Detect anomalies: python src/anomaly_detection.py")
    

if __name__ == "__main__":
    main()
