"""
Data Validation and Cleaning Module
Fixes critical data quality issues before analysis:
1. State name standardization
2. Date continuity validation
3. Geographic hierarchy consistency
4. Robust baseline computation
"""

import pandas as pd
import numpy as np
from datetime import timedelta


class DataValidator:
    """Validates and cleans Aadhar datasets"""
    
    def __init__(self):
        """Initialize validator with state name mappings"""
        # Master state name mapping - canonical names
        self.state_mappings = {
            # West Bengal variations
            'west bengal': 'West Bengal',
            'WEST BENGAL': 'West Bengal',
            'westbengal': 'West Bengal',
            'West bengal': 'West Bengal',
            'west Bengal': 'West Bengal',
            'West Bangal': 'West Bengal',
            'West Bengli': 'West Bengal',
            
            # Odisha variations
            'odisha': 'Odisha',
            'ODISHA': 'Odisha',
            
            # Andhra Pradesh variations
            'andhra pradesh': 'Andhra Pradesh',
            'ANDHRA PRADESH': 'Andhra Pradesh',
            
            # Other variations
            'uttaranchal': 'Uttarakhand',
            'Uttaranchal': 'Uttarakhand',
            
            # Tamil Nadu
            'tamilnadu': 'Tamil Nadu',
            'Tamilnadu': 'Tamil Nadu',
            
            # Chhattisgarh
            'chhatisgarh': 'Chhattisgarh',
            'Chhatisgarh': 'Chhattisgarh',
        }
    
    def standardize_state_names(self, df, verbose=True):
        """
        Standardize state names to canonical form
        
        Args:
            df: DataFrame with 'state' column
            verbose: Print standardization report
            
        Returns:
            DataFrame with standardized state names
        """
        if verbose:
            print("\n" + "="*60)
            print("STANDARDIZING STATE NAMES")
            print("="*60)
            
            original_states = df['state'].nunique()
            print(f"Original unique states: {original_states}")
        
        df = df.copy()
        
        # Apply mappings
        df['state'] = df['state'].replace(self.state_mappings)
        
        # Also apply title case for consistency
        df['state'] = df['state'].str.strip().str.title()
        
        if verbose:
            standardized_states = df['state'].nunique()
            print(f"Standardized states: {standardized_states}")
            print(f"States merged: {original_states - standardized_states}")
            
            # Show most common states
            print(f"\nTop 10 states after standardization:")
            print(df['state'].value_counts().head(10))
        
        return df
    
    def validate_date_continuity(self, df, verbose=True):
        """
        Check for missing dates and fill gaps
        
        Args:
            df: DataFrame with 'date' column
            verbose: Print validation report
            
        Returns:
            DataFrame with continuous dates (filled with zeros)
        """
        if verbose:
            print("\n" + "="*60)
            print("VALIDATING DATE CONTINUITY")
            print("="*60)
        
        min_date = df['date'].min()
        max_date = df['date'].max()
        expected_range = pd.date_range(min_date, max_date, freq='D')
        actual_dates = df['date'].unique()
        
        missing_dates = expected_range.difference(actual_dates)
        
        if verbose:
            print(f"Date range: {min_date.date()} to {max_date.date()}")
            print(f"Expected days: {len(expected_range)}")
            print(f"Actual unique dates: {len(actual_dates)}")
            print(f"Missing dates: {len(missing_dates)}")
            
            if len(missing_dates) > 0:
                print(f"\n⚠️  WARNING: {len(missing_dates)} missing dates found!")
                if len(missing_dates) <= 10:
                    print("Missing dates:")
                    for date in sorted(missing_dates):
                        print(f"  - {date.date()}")
                else:
                    print(f"First 10 missing dates:")
                    for date in sorted(missing_dates[:10]):
                        print(f"  - {date.date()}")
        
        return {
            'has_gaps': len(missing_dates) > 0,
            'missing_dates': missing_dates,
            'total_gaps': len(missing_dates),
            'date_range': expected_range
        }
    
    def validate_geographic_hierarchy(self, df, verbose=True):
        """
        Validate state-district-pincode hierarchy consistency
        
        Args:
            df: DataFrame with geographic columns
            verbose: Print validation report
            
        Returns:
            Validation report dictionary
        """
        if verbose:
            print("\n" + "="*60)
            print("VALIDATING GEOGRAPHIC HIERARCHY")
            print("="*60)
        
        issues = []
        
        # Check 1: Districts appearing in multiple states
        district_state_map = df.groupby('district')['state'].nunique()
        multi_state_districts = district_state_map[district_state_map > 1]
        
        if len(multi_state_districts) > 0:
            if verbose:
                print(f"\n⚠️  WARNING: {len(multi_state_districts)} districts appear in multiple states!")
                print(f"Top 10 inconsistent districts:")
                for district in multi_state_districts.head(10).index:
                    states = df[df['district'] == district]['state'].unique()
                    print(f"  '{district}' in: {', '.join(states)}")
            
            issues.append({
                'type': 'multi_state_district',
                'count': len(multi_state_districts),
                'districts': multi_state_districts.index.tolist()
            })
        else:
            if verbose:
                print("✓ All districts belong to single state")
        
        # Check 2: Pincodes in multiple districts
        pincode_district_map = df.groupby('pincode')['district'].nunique()
        multi_district_pins = pincode_district_map[pincode_district_map > 1]
        
        if len(multi_district_pins) > 0:
            if verbose:
                print(f"\n⚠️  WARNING: {len(multi_district_pins)} pincodes appear in multiple districts!")
                print(f"Sample (first 5):")
                for pin in list(multi_district_pins.head(5).index):
                    districts = df[df['pincode'] == pin]['district'].unique()
                    print(f"  Pincode {pin} in: {', '.join(districts[:3])}")
            
            issues.append({
                'type': 'multi_district_pincode',
                'count': len(multi_district_pins),
                'pincodes': multi_district_pins.index.tolist()[:100]  # Limit to 100
            })
        else:
            if verbose:
                print("✓ All pincodes belong to single district")
        
        # Check 3: Pincodes in multiple states
        pincode_state_map = df.groupby('pincode')['state'].nunique()
        multi_state_pins = pincode_state_map[pincode_state_map > 1]
        
        if len(multi_state_pins) > 0:
            if verbose:
                print(f"\n⚠️  WARNING: {len(multi_state_pins)} pincodes appear in multiple states!")
            
            issues.append({
                'type': 'multi_state_pincode',
                'count': len(multi_state_pins),
                'pincodes': multi_state_pins.index.tolist()[:100]
            })
        
        if verbose:
            print(f"\nTotal hierarchy issues: {len(issues)}")
        
        return issues
    
    def compute_robust_baselines(self, df, columns, percentile=99, verbose=True):
        """
        Compute robust baseline statistics excluding top outliers
        
        Args:
            df: DataFrame
            columns: List of columns to compute baselines for
            percentile: Percentile threshold (default 99, excludes top 1%)
            verbose: Print baseline report
            
        Returns:
            Dictionary with robust statistics
        """
        if verbose:
            print("\n" + "="*60)
            print("COMPUTING ROBUST BASELINES")
            print("="*60)
            print(f"Excluding top {100-percentile}% for baseline computation")
        
        baselines = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Full statistics (including outliers)
            full_mean = df[col].mean()
            full_std = df[col].std()
            full_median = df[col].median()
            
            # Robust statistics (excluding top 1%)
            threshold = df[col].quantile(percentile / 100)
            baseline_data = df[df[col] <= threshold][col]
            
            robust_mean = baseline_data.mean()
            robust_std = baseline_data.std()
            robust_median = baseline_data.median()
            
            baselines[col] = {
                'full': {
                    'mean': full_mean,
                    'std': full_std,
                    'median': full_median,
                    'min': df[col].min(),
                    'max': df[col].max()
                },
                'robust': {
                    'mean': robust_mean,
                    'std': robust_std,
                    'median': robust_median,
                    'threshold': threshold,
                    'n_baseline': len(baseline_data),
                    'n_excluded': len(df) - len(baseline_data)
                },
                'corruption_factor': full_mean / robust_mean if robust_mean > 0 else np.inf
            }
            
            if verbose:
                print(f"\n{col}:")
                print(f"  Full mean: {full_mean:,.2f} (corrupted by outliers)")
                print(f"  Robust mean: {robust_mean:,.2f} (baseline for detection)")
                print(f"  Corruption factor: {baselines[col]['corruption_factor']:.2f}x")
                print(f"  Excluded {len(df) - len(baseline_data):,} outliers (>{threshold:,.0f})")
        
        return baselines
    
    def check_functional_duplicates(self, df, key_columns, verbose=True):
        """
        Check for functional duplicates (same key, multiple records)
        
        Args:
            df: DataFrame
            key_columns: Columns that define uniqueness
            verbose: Print duplicate report
            
        Returns:
            Duplicate report
        """
        if verbose:
            print("\n" + "="*60)
            print("CHECKING FUNCTIONAL DUPLICATES")
            print("="*60)
            print(f"Key columns: {', '.join(key_columns)}")
        
        # Count records per key
        key_counts = df.groupby(key_columns).size()
        duplicates = key_counts[key_counts > 1]
        
        if len(duplicates) > 0:
            total_duplicate_records = (key_counts - 1).sum()
            
            if verbose:
                print(f"\n⚠️  WARNING: {len(duplicates):,} keys have multiple records!")
                print(f"Total duplicate records: {total_duplicate_records:,}")
                print(f"Percentage: {total_duplicate_records/len(df)*100:.2f}%")
                
                print(f"\nTop 10 keys with most duplicates:")
                for key, count in duplicates.nlargest(10).items():
                    print(f"  {key}: {count} records")
            
            return {
                'has_duplicates': True,
                'duplicate_keys': len(duplicates),
                'duplicate_records': total_duplicate_records,
                'percentage': total_duplicate_records/len(df)*100
            }
        else:
            if verbose:
                print("✓ No functional duplicates found")
            return {'has_duplicates': False}
    
    def validate_dataset(self, df, dataset_name, verbose=True):
        """
        Run all validations and return cleaned dataset
        
        Args:
            df: Input DataFrame
            dataset_name: Name of dataset for reporting
            verbose: Print validation reports
            
        Returns:
            Cleaned DataFrame and validation report
        """
        if verbose:
            print("\n" + "="*70)
            print(f"VALIDATING DATASET: {dataset_name.upper()}")
            print("="*70)
        
        report = {'dataset': dataset_name}
        
        # 1. Standardize state names
        df = self.standardize_state_names(df, verbose=verbose)
        
        # 2. Validate date continuity
        date_validation = self.validate_date_continuity(df, verbose=verbose)
        report['date_validation'] = date_validation
        
        # 3. Validate geographic hierarchy
        geo_issues = self.validate_geographic_hierarchy(df, verbose=verbose)
        report['geo_issues'] = geo_issues
        
        # 4. Check for functional duplicates
        dup_report = self.check_functional_duplicates(
            df, 
            ['date', 'state', 'district', 'pincode'],
            verbose=verbose
        )
        report['duplicates'] = dup_report
        
        # 5. Compute robust baselines
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_cols = [col for col in numeric_cols if col not in ['pincode', 'week']]
        
        baselines = self.compute_robust_baselines(df, value_cols, verbose=verbose)
        report['baselines'] = baselines
        
        if verbose:
            print("\n" + "="*70)
            print(f"VALIDATION COMPLETE: {dataset_name.upper()}")
            print("="*70)
        
        return df, report


def main():
    """Test validation on processed datasets"""
    import yaml
    from pathlib import Path
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    processed_dir = Path(config['data']['processed_dir'])
    
    validator = DataValidator()
    
    # Validate each dataset
    for dataset_name in ['biometric', 'demographic', 'enrolment']:
        file_path = processed_dir / f"{dataset_name}.csv"
        
        if file_path.exists():
            print(f"\n{'#'*70}")
            print(f"# Processing: {dataset_name}")
            print(f"{'#'*70}")
            
            df = pd.read_csv(file_path, parse_dates=['date'])
            df_clean, report = validator.validate_dataset(df, dataset_name, verbose=True)
            
            # Save cleaned version
            output_path = processed_dir / f"{dataset_name}_validated.csv"
            df_clean.to_csv(output_path, index=False)
            print(f"\n✓ Saved validated dataset to: {output_path}")


if __name__ == "__main__":
    main()
