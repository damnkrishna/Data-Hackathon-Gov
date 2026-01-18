"""
Interactive Visualization Dashboard for Aadhar Anomaly Detection
Run with: streamlit run src/visualization_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml

# Page config
st.set_page_config(
    page_title="Aadhar Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Paths
processed_dir = Path(config['data']['processed_dir'])
anomaly_dir = Path(config['data']['output_dir']) / 'anomalies'

# Load data functions
@st.cache_data
def load_processed_data():
    """Load processed datasets"""
    datasets = {}
    for name in ['biometric', 'demographic', 'enrolment']:
        file_path = processed_dir / f"{name}.csv"
        if file_path.exists():
            datasets[name] = pd.read_csv(file_path, parse_dates=['date'])
    return datasets

@st.cache_data
def load_anomaly_data():
    """Load all anomaly detection results"""
    anomalies = {}
    if anomaly_dir.exists():
        for file in anomaly_dir.glob('*.csv'):
            anomalies[file.stem] = pd.read_csv(file)
    return anomalies

# Main dashboard
def main():
    st.title("🔍 Aadhar Anomaly Detection Dashboard")
    st.markdown("**Comprehensive anomaly analysis across biometric, demographic, and enrolment data**")
    
    # Load data
    with st.spinner("Loading data..."):
        datasets = load_processed_data()
        anomalies = load_anomaly_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "⚠️ Temporal Spikes", "🗺️ Geographic Analysis", 
         "🔢 Ratio Anomalies", "📈 Statistical Outliers", "🤖 ML Detection", 
         "⚙️ Data Quality"]
    )
    
    # Overview Page
    if page == "📊 Overview":
        show_overview(datasets, anomalies)
    
    # Temporal Spikes
    elif page == "⚠️ Temporal Spikes":
        show_temporal_spikes(datasets, anomalies)
    
    # Geographic Analysis
    elif page == "🗺️ Geographic Analysis":
        show_geographic_analysis(datasets, anomalies)
    
    # Ratio Anomalies
    elif page == "🔢 Ratio Anomalies":
        show_ratio_anomalies(anomalies)
    
    # Statistical Outliers
    elif page == "📈 Statistical Outliers":
        show_statistical_outliers(anomalies)
    
    # ML Detection
    elif page == "🤖 ML Detection":
        show_ml_detection(anomalies)
    
    # Data Quality
    elif page == "⚙️ Data Quality":
        show_data_quality(anomalies)


def show_overview(datasets, anomalies):
    """Overview dashboard page"""
    st.header("Overview")
    
    # Dataset stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Biometric Records", f"{len(datasets['biometric']):,}")
        st.metric("Date Range", f"{datasets['biometric']['date'].min().date()} to {datasets['biometric']['date'].max().date()}")
    
    with col2:
        st.metric("Demographic Records", f"{len(datasets['demographic']):,}")
        st.metric("States", f"{datasets['demographic']['state'].nunique()}")
    
    with col3:
        st.metric("Enrolment Records", f"{len(datasets['enrolment']):,}")
        st.metric("Districts", f"{datasets['enrolment']['district'].nunique()}")
    
    st.markdown("---")
    
    # Anomaly summary
    st.subheader("🚨 Anomaly Detection Summary")
    
    anomaly_counts = []
    for name, df in anomalies.items():
        anomaly_counts.append({
            'Type': name.replace('_', ' ').title(),
            'Count': len(df)
        })
    
    anomaly_df = pd.DataFrame(anomaly_counts).sort_values('Count', ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(anomaly_df, use_container_width=True)
    
    with col2:
        fig = px.bar(anomaly_df, x='Type', y='Count', 
                     title='Anomalies by Detection Method',
                     color='Count', color_continuous_scale='Reds')
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Critical findings
    st.markdown("---")
    st.subheader("🔴 Critical Findings")
    
    st.error("**CRITICAL: Massive first-of-month spikes detected**")
    st.write("- **March 1, 2025**: 7.5M demographic authentications (28.8x median)")
    st.write("- Pattern repeats on 1st of every month (Mar-Jul 2025)")
    
    st.warning("**HIGH: 96,718 extreme bio/demo ratio anomalies (8.17%)**")
    st.write("- Ratios up to 333:1 detected (potential fraud)")
    st.write("- Concentrated in J&K and Maharashtra")
    
    st.info("**MEDIUM: 13 data quality issues - state name inconsistencies**")
    st.write("- 'West Bengal' has 4 spelling variants corrupting analytics")


def show_temporal_spikes(datasets, anomalies):
    """Temporal spike analysis page"""
    st.header("⚠️ Temporal Spike Detection")
    
    # Dataset selector
    dataset_name = st.selectbox("Select Dataset", ['biometric', 'demographic', 'enrolment'])
    
    df = datasets[dataset_name]
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    value_cols = [col for col in numeric_cols if col not in ['pincode', 'week']]
    
    # Daily aggregation
    daily = df.groupby('date')[value_cols].sum().reset_index()
    
    # Plot daily trends
    st.subheader(f"{dataset_name.title()} Daily Trends")
    
    col = st.selectbox("Select Metric", value_cols)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily[col],
        mode='lines',
        name=col,
        line=dict(color='blue', width=2)
    ))
    
    # Add median line
    median_val = daily[col].median()
    fig.add_hline(y=median_val, line_dash="dash", line_color="green", 
                  annotation_text=f"Median: {median_val:,.0f}")
    
    # Add 10x threshold
    fig.add_hline(y=median_val * 10, line_dash="dash", line_color="red",
                  annotation_text=f"10x Threshold: {median_val*10:,.0f}")
    
    fig.update_layout(
        title=f"{col} - Daily Trend",
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show spike details
    st.markdown("---")
    st.subheader("Detected Spikes")
    
    spike_file = f"{dataset_name}_temporal_spikes"
    if spike_file in anomalies:
        spikes = anomalies[spike_file]
        st.dataframe(spikes, use_container_width=True)
        
        # Download button
        csv = spikes.to_csv(index=False)
        st.download_button(
            label="📥 Download Spikes CSV",
            data=csv,
            file_name=f"{spike_file}.csv",
            mime="text/csv"
        )
    else:
        st.info("No temporal spikes detected for this dataset")


def show_geographic_analysis(datasets, anomalies):
    """Geographic analysis page"""
    st.header("🗺️ Geographic Analysis")
    
    dataset_name = st.selectbox("Select Dataset", ['biometric', 'demographic', 'enrolment'])
    
    df = datasets[dataset_name]
    
    # State-level aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    value_cols = [col for col in numeric_cols if col not in ['pincode', 'week']]
    
    state_agg = df.groupby('state')[value_cols].sum().reset_index()
    
    col = st.selectbox("Select Metric", value_cols)
    
    state_agg_sorted = state_agg.sort_values(col, ascending=False)
    
    # Top states bar chart
    st.subheader(f"Top 20 States by {col}")
    
    fig = px.bar(
        state_agg_sorted.head(20),
        x='state', y=col,
        title=f"Top 20 States - {col}",
        color=col,
        color_continuous_scale='Viridis'
    )
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Geographic outliers
    st.markdown("---")
    st.subheader("Geographic Outliers")
    
    outlier_file = f"{dataset_name}_geographic_outliers"
    if outlier_file in anomalies:
        outliers = anomalies[outlier_file]
        
        # Filter by column
        col_outliers = outliers[outliers['column'] == col]
        
        st.write(f"**{len(col_outliers)} outliers detected for {col}**")
        
        # Show top outliers
        st.dataframe(
            col_outliers.nlargest(20, 'value')[['state', 'district', 'value', 'state_median', 'state_upper_bound']],
            use_container_width=True
        )
        
        # Outlier map (state-level heatmap)
        outlier_count = col_outliers.groupby('state').size().reset_index(name='outlier_count')
        
        fig = px.bar(
            outlier_count.nlargest(15, 'outlier_count'),
            x='state', y='outlier_count',
            title=f"States with Most {col} Outliers",
            color='outlier_count',
            color_continuous_scale='Reds'
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


def show_ratio_anomalies(anomalies):
    """Cross-dataset ratio anomalies page"""
    st.header("🔢 Biometric/Demographic Ratio Anomalies")
    
    if 'cross_dataset_ratio_anomalies' in anomalies:
        ratio_anomalies = anomalies['cross_dataset_ratio_anomalies']
        
        # Convert date if present
        if 'date' in ratio_anomalies.columns:
            ratio_anomalies['date'] = pd.to_datetime(ratio_anomalies['date'])
        
        st.write(f"**Total ratio anomalies detected: {len(ratio_anomalies):,}**")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Ratio", f"{ratio_anomalies['bio_demo_ratio'].mean():.2f}")
        with col2:
            st.metric("Median Ratio", f"{ratio_anomalies['bio_demo_ratio'].median():.2f}")
        with col3:
            st.metric("Max Ratio", f"{ratio_anomalies['bio_demo_ratio'].max():.0f}")
        
        # Distribution plot
        st.subheader("Ratio Distribution")
        
        # Clip for visualization
        ratio_clipped = ratio_anomalies['bio_demo_ratio'].clip(0, 20)
        
        fig = px.histogram(
            ratio_clipped,
            nbins=50,
            title="Bio/Demo Ratio Distribution (clipped at 20 for visibility)",
            labels={'value': 'Ratio', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top anomalies
        st.subheader("Top 50 Extreme Ratios")
        
        top_anomalies = ratio_anomalies.nlargest(50, 'bio_demo_ratio')[
            ['date', 'state', 'district', 'bio_total', 'demo_total', 'bio_demo_ratio']
        ]
        
        st.dataframe(top_anomalies, use_container_width=True)
        
        # State breakdown
        st.subheader("Anomalies by State")
        
        state_counts = ratio_anomalies['state'].value_counts().head(20).reset_index()
        state_counts.columns = ['state', 'count']
        
        fig = px.bar(
            state_counts,
            x='state', y='count',
            title="Top 20 States with Ratio Anomalies",
            color='count',
            color_continuous_scale='Oranges'
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No ratio anomaly data found")


def show_statistical_outliers(anomalies):
    """Statistical outliers page"""
    st.header("📈 Statistical Outliers (Z-Score & IQR)")
    
    method = st.radio("Select Method", ['Z-Score', 'IQR'])
    dataset = st.selectbox("Select Dataset", ['biometric', 'demographic', 'enrolment'])
    
    file_key = f"{dataset}_{method.lower().replace('-', '')}_anomalies"
    
    if file_key in anomalies:
        outliers = anomalies[file_key]
        
        st.write(f"**Total {method} outliers: {len(outliers):,}**")
        
        # Show sample
        st.subheader("Sample Outliers")
        st.dataframe(outliers.head(100), use_container_width=True)
        
        # Breakdown by column
        if 'anomaly_column' in outliers.columns:
            st.subheader("Outliers by Column")
            
            col_counts = outliers['anomaly_column'].value_counts().reset_index()
            col_counts.columns = ['column', 'count']
            
            fig = px.pie(
                col_counts,
                values='count', names='column',
                title=f"{method} Outliers by Column"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download
        csv = outliers.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {method} Outliers CSV",
            data=csv,
            file_name=f"{file_key}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No {method} data found for {dataset}")


def show_ml_detection(anomalies):
    """ML-based detection page"""
    st.header("🤖 Machine Learning Detection (Isolation Forest)")
    
    dataset = st.selectbox("Select Dataset", ['biometric', 'demographic', 'enrolment'])
    
    file_key = f"{dataset}_isolation_forest"
    
    if file_key in anomalies:
        ml_anomalies = anomalies[file_key]
        
        if 'date' in ml_anomalies.columns:
            ml_anomalies['date'] = pd.to_datetime(ml_anomalies['date'])
        
        st.write(f"**Total ML-detected anomalies: {len(ml_anomalies):,}**")
        
        # Anomaly score distribution
        if 'anomaly_score' in ml_anomalies.columns:
            st.subheader("Anomaly Score Distribution")
            
            fig = px.histogram(
                ml_anomalies,
                x='anomaly_score',
                nbins=50,
                title="Isolation Forest Anomaly Scores (lower = more anomalous)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top anomalies
            st.subheader("Most Anomalous Records (lowest scores)")
            
            top_anomalies = ml_anomalies.nsmallest(50, 'anomaly_score')
            st.dataframe(top_anomalies, use_container_width=True)
        
        else:
            st.dataframe(ml_anomalies.head(100), use_container_width=True)
        
        # Download
        csv = ml_anomalies.to_csv(index=False)
        st.download_button(
            label="📥 Download ML Anomalies CSV",
            data=csv,
            file_name=f"{file_key}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No ML detection data found for {dataset}")


def show_data_quality(anomalies):
    """Data quality issues page"""
    st.header("⚙️ Data Quality Issues")
    
    if 'data_quality_issues' in anomalies:
        issues = anomalies['data_quality_issues']
        
        st.write(f"**Total issues detected: {len(issues)}**")
        
        # Show all issues
        st.subheader("State Name Inconsistencies")
        st.dataframe(issues, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("🔧 Recommended Actions")
        
        st.error("**CRITICAL: Standardize state names immediately**")
        st.write("""
        **Impact**: All state-level analytics are corrupted due to naming inconsistencies
        
        **Action Plan**:
        1. Create master state name lookup table with canonical names
        2. Implement data validation at ingestion points
        3. Reprocess historical data with standardized names
        4. Add database constraints to prevent future inconsistencies
        
        **Example Standardization**:
        - All variations of "West Bengal" → "West Bengal"
        - All variations of "Odisha" → "Odisha"
        - All variations of "Andhra Pradesh" → "Andhra Pradesh"
        """)
        
    else:
        st.success("No data quality issues detected!")


if __name__ == "__main__":
    main()
