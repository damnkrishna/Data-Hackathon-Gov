"""
Exploratory Data Analysis (EDA) for Aadhar Anomaly Detection
Generates comprehensive statistical analysis and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import warnings
from data_validation import DataValidator  # Import validator
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class AadharEDA:
    """Exploratory Data Analysis for Aadhar datasets"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize EDA with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.output_dir = Path(self.config['data']['output_dir'])
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize validator
        self.validator = DataValidator()
        
    def load_processed_data(self):
        """Load processed datasets with validation"""
        print("\n" + "="*60)
        print("Loading Processed Datasets")
        print("="*60)
        
        datasets = {}
        
        for name in ['biometric', 'demographic', 'enrolment', 'combined_aggregated']:
            file_path = self.processed_dir / f"{name}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, parse_dates=['date'])
                
                # Apply validation and cleaning (except for combined)
                if name != 'combined_aggregated':
                    df, report = self.validator.validate_dataset(df, name, verbose=False)
                    # Store validation report
                    datasets[f'{name}_validation_report'] = report
                
                datasets[name] = df
                print(f"✓ Loaded {name}: {df.shape}")
            else:
                print(f"✗ File not found: {file_path}")
        
        return datasets
    
    def temporal_analysis(self, df, dataset_name, validation_report=None):
        """Analyze temporal patterns with robust baselines"""
        print(f"\n{'='*60}")
        print(f"Temporal Analysis - {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Get numeric columns (exclude date and geo columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        geo_cols = ['pincode']
        value_cols = [col for col in numeric_cols if col not in geo_cols]
        
        # Daily aggregation
        daily = df.groupby('date')[value_cols].sum()
        
        print(f"\nDate Range: {daily.index.min()} to {daily.index.max()}")
        print(f"Number of days: {len(daily)}")
        
        # Report robust baselines if available
        if validation_report and 'baselines' in validation_report:
            print(f"\n{'='*60}")
            print("ROBUST BASELINE STATISTICS (for anomaly detection)")
            print(f"{'='*60}")
            for col in value_cols:
                if col in validation_report['baselines']:
                    baseline = validation_report['baselines'][col]
                    print(f"\n{col}:")
                    print(f"  Robust mean (baseline): {baseline['robust']['mean']:,.2f}")
                    print(f"  Robust std: {baseline['robust']['std']:,.2f}")
                    print(f"  Full mean (corrupted): {baseline['full']['mean']:,.2f}")
                    print(f"  Corruption factor: {baseline['corruption_factor']:.2f}x")
        
        print(f"\nDaily Statistics:")
        print(daily.describe())
        
        # Plot daily trends
        fig, axes = plt.subplots(len(value_cols), 1, figsize=(14, 4*len(value_cols)))
        if len(value_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(value_cols):
            axes[i].plot(daily.index, daily[col], linewidth=1.5)
            axes[i].set_title(f'{col} - Daily Trend', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Count')
            axes[i].grid(True, alpha=0.3)
            
            # Add robust baseline line if available
            if validation_report and 'baselines' in validation_report:
                if col in validation_report['baselines']:
                    robust_mean = validation_report['baselines'][col]['robust']['mean']
                    axes[i].axhline(y=robust_mean, color='green', linestyle='--', 
                                  label=f'Robust Baseline: {robust_mean:,.0f}', alpha=0.7)
                    axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{dataset_name}_daily_trends.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: {dataset_name}_daily_trends.png")
        plt.close()
        
        # Weekly aggregation
        df['week'] = df['date'].dt.isocalendar().week
        weekly = df.groupby('week')[value_cols].sum()
        
        print(f"\nWeekly Statistics:")
        print(weekly.describe())
        
        return daily, weekly
    
    def geographic_analysis(self, df, dataset_name):
        """Analyze geographic patterns"""
        print(f"\n{'='*60}")
        print(f"Geographic Analysis - {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # State-level analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        geo_cols = ['pincode']
        value_cols = [col for col in numeric_cols if col not in geo_cols]
        
        state_agg = df.groupby('state')[value_cols].sum().sort_values(by=value_cols[0], ascending=False)
        
        print(f"\nTop 10 States by {value_cols[0]}:")
        print(state_agg.head(10))
        
        print(f"\nBottom 10 States by {value_cols[0]}:")
        print(state_agg.tail(10))
        
        # Plot top 20 states
        fig, ax = plt.subplots(figsize=(14, 8))
        state_agg.head(20).plot(kind='barh', ax=ax)
        ax.set_title(f'Top 20 States - {dataset_name.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
        ax.set_ylabel('State')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{dataset_name}_top_states.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: {dataset_name}_top_states.png")
        plt.close()
        
        # District-level analysis
        district_agg = df.groupby(['state', 'district'])[value_cols].sum().sort_values(by=value_cols[0], ascending=False)
        
        print(f"\nTop 10 Districts by {value_cols[0]}:")
        print(district_agg.head(10))
        
        return state_agg, district_agg
    
    def distribution_analysis(self, df, dataset_name):
        """Analyze value distributions"""
        print(f"\n{'='*60}")
        print(f"Distribution Analysis - {dataset_name.upper()}")
        print(f"{'='*60}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        geo_cols = ['pincode']
        value_cols = [col for col in numeric_cols if col not in geo_cols]
        
        # Create distribution plots
        fig, axes = plt.subplots(len(value_cols), 2, figsize=(14, 4*len(value_cols)))
        if len(value_cols) == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(value_cols):
            # Histogram
            axes[i, 0].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
            axes[i, 0].set_title(f'{col} - Distribution', fontweight='bold')
            axes[i, 0].set_xlabel('Value')
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].set_yscale('log')
            
            # Box plot
            axes[i, 1].boxplot(df[col])
            axes[i, 1].set_title(f'{col} - Box Plot', fontweight='bold')
            axes[i, 1].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{dataset_name}_distributions.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: {dataset_name}_distributions.png")
        plt.close()
        
        # Statistical summary with percentiles
        print(f"\nDetailed Statistics:")
        for col in value_cols:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Median: {df[col].median():.2f}")
            print(f"  Std: {df[col].std():.2f}")
            print(f"  Min: {df[col].min():.0f}")
            print(f"  Max: {df[col].max():.0f}")
            print(f"  95th percentile: {df[col].quantile(0.95):.2f}")
            print(f"  99th percentile: {df[col].quantile(0.99):.2f}")
            print(f"  Zeros: {(df[col] == 0).sum()} ({(df[col] == 0).sum()/len(df)*100:.2f}%)")
    
    def cross_dataset_analysis(self, datasets):
        """Analyze relationships between datasets"""
        print(f"\n{'='*60}")
        print("Cross-Dataset Analysis")
        print(f"{'='*60}")
        
        combined = datasets['combined_aggregated']
        
        # Check for column existence patterns
        print("\nDataset Coverage Patterns:")
        print(f"  Records with all three datasets: {((combined['has_bio'] == 1) & (combined['has_demo'] == 1) & (combined['has_enrol'] == 1)).sum():,}")
        print(f"  Records with bio + demo only: {((combined['has_bio'] == 1) & (combined['has_demo'] == 1) & (combined['has_enrol'] == 0)).sum():,}")
        print(f"  Records with bio + enrol only: {((combined['has_bio'] == 1) & (combined['has_demo'] == 0) & (combined['has_enrol'] == 1)).sum():,}")
        print(f"  Records with demo + enrol only: {((combined['has_bio'] == 0) & (combined['has_demo'] == 1) & (combined['has_enrol'] == 1)).sum():,}")
        print(f"  Records with only bio: {((combined['has_bio'] == 1) & (combined['has_demo'] == 0) & (combined['has_enrol'] == 0)).sum():,}")
        print(f"  Records with only demo: {((combined['has_bio'] == 0) & (combined['has_demo'] == 1) & (combined['has_enrol'] == 0)).sum():,}")
        print(f"  Records with only enrol: {((combined['has_bio'] == 0) & (combined['has_demo'] == 0) & (combined['has_enrol'] == 1)).sum():,}")
        
        # Correlation analysis on records with both bio and demo
        subset = combined[(combined['has_bio'] == 1) & (combined['has_demo'] == 1)].copy()
        
        if len(subset) > 0:
            # Calculate bio/demo ratio
            subset['bio_total'] = subset['bio_bio_age_5_17'] + subset['bio_bio_age_17_']
            subset['demo_total'] = subset['demo_demo_age_5_17'] + subset['demo_demo_age_17_']
            
            # Remove zeros to avoid division issues
            subset_nonzero = subset[(subset['bio_total'] > 0) & (subset['demo_total'] > 0)]
            
            if len(subset_nonzero) > 0:
                subset_nonzero['bio_demo_ratio'] = subset_nonzero['bio_total'] / subset_nonzero['demo_total']
                
                print(f"\nBiometric/Demographic Ratio Statistics:")
                print(f"  Mean: {subset_nonzero['bio_demo_ratio'].mean():.3f}")
                print(f"  Median: {subset_nonzero['bio_demo_ratio'].median():.3f}")
                print(f"  Std: {subset_nonzero['bio_demo_ratio'].std():.3f}")
                print(f"  Min: {subset_nonzero['bio_demo_ratio'].min():.3f}")
                print(f"  Max: {subset_nonzero['bio_demo_ratio'].max():.3f}")
                
                # Plot ratio distribution
                fig, ax = plt.subplots(figsize=(12, 6))
                # Clip extreme values for better visualization
                ratio_clipped = subset_nonzero['bio_demo_ratio'].clip(0, 5)
                ax.hist(ratio_clipped, bins=100, edgecolor='black', alpha=0.7)
                ax.set_title('Biometric/Demographic Ratio Distribution (clipped at 5)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Ratio')
                ax.set_ylabel('Frequency')
                ax.axvline(subset_nonzero['bio_demo_ratio'].median(), color='red', linestyle='--', label=f'Median: {subset_nonzero["bio_demo_ratio"].median():.2f}')
                ax.legend()
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'bio_demo_ratio.png', dpi=150, bbox_inches='tight')
                print(f"\n✓ Saved: bio_demo_ratio.png")
                plt.close()
    
    def generate_summary_report(self, datasets):
        """Generate comprehensive EDA summary report"""
        print(f"\n{'='*60}")
        print("Generating Summary Report")
        print(f"{'='*60}")
        
        report_path = self.output_dir / 'eda_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("AADHAR ANOMALY DETECTION - EDA SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            for name, df in datasets.items():
                # Skip validation reports and combined dataset
                if name == 'combined_aggregated' or '_validation_report' in name:
                    continue
                
                # Skip if not a DataFrame
                if not isinstance(df, pd.DataFrame):
                    continue
                    
                f.write(f"\n{name.upper()} Dataset\n")
                f.write("-"*60 + "\n")
                f.write(f"Records: {len(df):,}\n")
                f.write(f"Date Range: {df['date'].min()} to {df['date'].max()}\n")
                f.write(f"States: {df['state'].nunique()}\n")
                f.write(f"Districts: {df['district'].nunique()}\n")
                f.write(f"Pincodes: {df['pincode'].nunique()}\n\n")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                value_cols = [col for col in numeric_cols if col != 'pincode']
                
                f.write("Value Columns Statistics:\n")
                for col in value_cols:
                    f.write(f"\n{col}:\n")
                    f.write(f"  Total: {df[col].sum():,}\n")
                    f.write(f"  Mean: {df[col].mean():.2f}\n")
                    f.write(f"  Median: {df[col].median():.2f}\n")
                    f.write(f"  Max: {df[col].max():,}\n")
                    f.write(f"  99th percentile: {df[col].quantile(0.99):.2f}\n")
                
                f.write("\n")
        
        print(f"\n✓ Saved: eda_summary.txt")
        print(f"Report location: {report_path}")
    
    def run_full_eda(self):
        """Run complete EDA pipeline with validation"""
        print("\n" + "="*60)
        print("AADHAR ANOMALY DETECTION - EDA WITH VALIDATION")
        print("="*60)
        
        # Load data (now includes validation)
        datasets = self.load_processed_data()
        
        # Analyze individual datasets
        for name in ['biometric', 'demographic', 'enrolment']:
            if name in datasets:
                df = datasets[name]
                validation_report = datasets.get(f'{name}_validation_report')
                
                self.temporal_analysis(df, name, validation_report)
                self.geographic_analysis(df, name)
                self.distribution_analysis(df, name)
        
        # Cross-dataset analysis
        if 'combined_aggregated' in datasets:
            self.cross_dataset_analysis(datasets)
        
        # Generate summary report
        self.generate_summary_report(datasets)
        
        print("\n" + "="*60)
        print("EDA COMPLETE!")
        print("="*60)
        print(f"\nPlots saved to: {self.plots_dir}")
        print(f"Report saved to: {self.output_dir / 'eda_summary.txt'}")
        print("\nNext step: python src/anomaly_detection.py")


def main():
    """Main execution function"""
    eda = AadharEDA(config_path='config.yaml')
    eda.run_full_eda()


if __name__ == "__main__":
    main()
