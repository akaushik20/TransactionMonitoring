import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def generate_eda_report(df: pd.DataFrame, title: str = "Data Report", output_file: str = "data_report.html") -> None:
    """Generate an EDA report using ydata-profiling.
    
    Args:
        df: The pandas DataFrame to analyze
        title: The title for the report
        output_file: The filename to save the report to
    """
    profile = ProfileReport(df, title=title)
    profile.to_file(output_file)
    print(f"EDA report saved as '{output_file}'")

def calculate_false_positive_rates(df: pd.DataFrame) -> None:
    """Calculate and plot false positive rates by different dimensions."""
    
    # Check if alert_outcome column exists (this contains true_positive/false_positive)
    if 'alert_outcome' not in df.columns:
        print("No 'alert_outcome' column found. Cannot calculate false positive rate.")
        return
    
    # Calculate overall FPR using alert_outcome
    total_alerts = len(df)
    false_positives = (df['alert_outcome'] == 'false_positive').sum()
    true_positives = (df['alert_outcome'] == 'true_positive').sum()
    
    # FPR = False Positives / (False Positives + True Negatives)
    # Since we only have alerts data, we calculate FPR as: False Positives / Total Alerts
    overall_fpr = false_positives / total_alerts if total_alerts > 0 else 0
    
    print(f"\n📊 FALSE POSITIVE RATE ANALYSIS")
    print("=" * 50)
    print(f"Total Alerts: {total_alerts:,}")
    print(f"False Positives: {false_positives:,}")
    print(f"True Positives: {true_positives:,}")
    print(f"False Positive Rate: {overall_fpr:.3f} ({overall_fpr*100:.1f}%)")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Overall FPR
    categories = ['False Positive', 'True Positive']
    values = [false_positives, true_positives]
    colors = ['coral', 'lightgreen']
    axes[0,0].bar(categories, values, color=colors)
    axes[0,0].set_title('Alert Outcomes Distribution')
    axes[0,0].set_ylabel('Count')
    
    # Plot 2: FPR by alert_type
    if 'alert_type' in df.columns:
        fpr_by_type = df.groupby('alert_type')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        ).sort_values(ascending=False)
        
        fpr_by_type.plot(kind='bar', ax=axes[0,1], color='skyblue')
        axes[0,1].set_title('FPR by Alert Type')
        axes[0,1].set_ylabel('FPR')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        print("\nFPR by Alert Type:")
        for alert_type, fpr in fpr_by_type.items():
            count = ((df['alert_type'] == alert_type) & (df['alert_outcome'] == 'false_positive')).sum()
            total = (df['alert_type'] == alert_type).sum()
            print(f"  {alert_type}: {fpr:.3f} ({count}/{total})")
    else:
        axes[0,1].text(0.5, 0.5, 'alert_type column\nnot found', ha='center', va='center')
        axes[0,1].set_title('FPR by Alert Type (No Data)')
    
    # Plot 3: FPR by customer_risk_tier
    if 'customer_risk_tier' in df.columns:
        fpr_by_risk = df.groupby('customer_risk_tier')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        ).sort_values(ascending=False)
        
        fpr_by_risk.plot(kind='bar', ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('FPR by Customer Risk Tier')
        axes[1,0].set_ylabel('FPR')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        print("\nFPR by Customer Risk Tier:")
        for risk_tier, fpr in fpr_by_risk.items():
            count = ((df['customer_risk_tier'] == risk_tier) & (df['alert_outcome'] == 'false_positive')).sum()
            total = (df['customer_risk_tier'] == risk_tier).sum()
            print(f"  {risk_tier}: {fpr:.3f} ({count}/{total})")
    else:
        axes[1,0].text(0.5, 0.5, 'customer_risk_tier\ncolumn not found', ha='center', va='center')
        axes[1,0].set_title('FPR by Risk Tier (No Data)')
    
    # Plot 4: FPR by country
    if 'country' in df.columns:
        fpr_by_country = df.groupby('country')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        ).sort_values(ascending=False)
        
        # Show top 10 countries by FPR
        top_countries = fpr_by_country.head(10)
        top_countries.plot(kind='bar', ax=axes[1,1], color='gold')
        axes[1,1].set_title('FPR by Country (Top 10)')
        axes[1,1].set_ylabel('FPR')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        print("\nFPR by Country (Top 10):")
        for country, fpr in top_countries.items():
            count = ((df['country'] == country) & (df['alert_outcome'] == 'false_positive')).sum()
            total = (df['country'] == country).sum()
            print(f"  {country}: {fpr:.3f} ({count}/{total})")
    else:
        axes[1,1].text(0.5, 0.5, 'country column\nnot found', ha='center', va='center')
        axes[1,1].set_title('FPR by Country (No Data)')
    
    plt.tight_layout()
    plt.savefig('false_positive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPlots saved as 'false_positive_analysis.png'")
    print("=" * 50)
    
    plt.tight_layout()
    plt.savefig('false_positive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPlots saved as 'false_positive_analysis.png'")
    print("=" * 50)

def generate_fpr_report(df: pd.DataFrame, output_file: str = "false_positive_analysis_report.txt") -> None:
    """Generate a comprehensive false positive rate analysis report."""
    
    if 'alert_outcome' not in df.columns:
        print("Cannot generate FPR report: Missing 'alert_outcome' column")
        return
    
    # Calculate metrics
    total_alerts = len(df)
    false_positives = (df['alert_outcome'] == 'false_positive').sum()
    true_positives = (df['alert_outcome'] == 'true_positive').sum()
    overall_fpr = false_positives / total_alerts if total_alerts > 0 else 0
    
    # Generate report content
    report_content = f"""
FALSE POSITIVE RATE ANALYSIS REPORT
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
================================

EXECUTIVE SUMMARY
-----------------
Overall False Positive Rate: {overall_fpr:.3f} ({overall_fpr*100:.1f}%)
Total False Positives: {false_positives:,}
Total True Positives: {true_positives:,}
Total Alerts: {total_alerts:,}

DETAILED ANALYSIS
-----------------

1. OVERALL PERFORMANCE
   - Dataset Size: {len(df):,} alerts
   - False Positive Rate: {overall_fpr*100:.1f}%
   - True Positive Rate: {(true_positives/total_alerts)*100:.1f}%
   - Alert Effectiveness: {((true_positives/total_alerts)*100):.1f}% of alerts are valid

"""

    # Add breakdown by alert_type if available
    if 'alert_type' in df.columns:
        report_content += "2. FALSE POSITIVE RATE BY ALERT TYPE\n"
        fpr_by_type = df.groupby('alert_type')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        )
        for alert_type, fpr in fpr_by_type.items():
            fp_count = ((df['alert_type'] == alert_type) & (df['alert_outcome'] == 'false_positive')).sum()
            total_count = (df['alert_type'] == alert_type).sum()
            report_content += f"   - {alert_type}: {fpr:.3f} ({fpr*100:.1f}%) - {fp_count:,}/{total_count:,} alerts\n"
        report_content += "\n"

    # Add breakdown by customer_risk_tier if available
    if 'customer_risk_tier' in df.columns:
        report_content += "3. FALSE POSITIVE RATE BY CUSTOMER RISK TIER\n"
        fpr_by_risk = df.groupby('customer_risk_tier')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        )
        for risk_tier, fpr in fpr_by_risk.items():
            fp_count = ((df['customer_risk_tier'] == risk_tier) & (df['alert_outcome'] == 'false_positive')).sum()
            total_count = (df['customer_risk_tier'] == risk_tier).sum()
            report_content += f"   - {risk_tier}: {fpr:.3f} ({fpr*100:.1f}%) - {fp_count:,}/{total_count:,} alerts\n"
        report_content += "\n"

    # Add breakdown by country if available
    if 'country' in df.columns:
        report_content += "4. FALSE POSITIVE RATE BY COUNTRY (TOP 10)\n"
        fpr_by_country = df.groupby('country')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        ).sort_values(ascending=False)
        
        for country, fpr in fpr_by_country.head(10).items():
            fp_count = ((df['country'] == country) & (df['alert_outcome'] == 'false_positive')).sum()
            total_count = (df['country'] == country).sum()
            report_content += f"   - {country}: {fpr:.3f} ({fpr*100:.1f}%) - {fp_count:,}/{total_count:,} alerts\n"
        report_content += "\n"

    # Add recommendations
    report_content += """
RECOMMENDATIONS
---------------
"""
    if overall_fpr > 0.8:
        report_content += "- VERY HIGH FPR: Urgent model review needed - most alerts are false positives\n"
    elif overall_fpr > 0.6:
        report_content += "- HIGH FPR: Consider raising alert thresholds to reduce false positives\n"
    elif overall_fpr < 0.3:
        report_content += "- GOOD FPR: Alert system performing well\n"
    else:
        report_content += "- MODERATE FPR: Room for improvement in alert precision\n"
    
    report_content += "- Review alert types with highest FPR for threshold adjustments\n"
    report_content += "- Monitor performance by customer risk tiers and geographic regions\n"
    report_content += "- Consider model retraining if FPR is consistently high\n\n"
    
    report_content += "VISUAL ANALYSIS\n---------------\n"
    report_content += "- Charts saved as: false_positive_analysis.png\n"
    report_content += "- EDA Report: transaction_monitoring_report.html\n"
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report_content)
    
    print(f"\n📄 Comprehensive FPR report saved as: {output_file}")
    return report_content

def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Comprehensive data quality analysis.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary containing data quality metrics and issues
    """
    quality_report = {
        'dataset_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum()
        },
        'missing_data': {},
        'duplicates': {},
        'data_types': {},
        'outliers': {},
        'inconsistencies': {},
        'summary_flags': []
    }
    
    # 1. Missing Data Analysis
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    quality_report['missing_data'] = {
        'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
        'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
        'total_missing_values': missing_counts.sum(),
        'rows_with_any_missing': df.isnull().any(axis=1).sum()
    }
    
    # 2. Duplicate Analysis
    quality_report['duplicates'] = {
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
        'unique_rows': len(df.drop_duplicates())
    }
    
    # 3. Data Type Analysis
    quality_report['data_types'] = {
        'column_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
    }
    
    # 4. Outlier Detection (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    
    for col in numeric_cols:
        if df[col].notna().sum() > 0:  # Only analyze if column has non-null values
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_value': df[col].min(),
                'max_value': df[col].max()
            }
    
    quality_report['outliers'] = outlier_info
    
    # 5. Inconsistency Detection
    inconsistencies = {}
    
    # Check for negative values in columns that shouldn't have them
    amount_columns = [col for col in df.columns if 'amount' in col.lower() or 'value' in col.lower()]
    for col in amount_columns:
        if col in numeric_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                inconsistencies[f'{col}_negative_values'] = negative_count
    
    # Check for empty strings in text columns
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        empty_strings = (df[col] == '').sum()
        whitespace_only = df[col].str.strip().eq('').sum() if df[col].dtype == 'object' else 0
        if empty_strings > 0:
            inconsistencies[f'{col}_empty_strings'] = empty_strings
        if whitespace_only > empty_strings:
            inconsistencies[f'{col}_whitespace_only'] = whitespace_only - empty_strings
    
    quality_report['inconsistencies'] = inconsistencies
    
    # 6. Generate Summary Flags
    flags = []
    
    # Missing data flags
    high_missing_threshold = 50
    for col, pct in missing_percentages.items():
        if pct > high_missing_threshold:
            flags.append(f"HIGH MISSING: {col} has {pct:.1f}% missing values")
        elif pct > 20:
            flags.append(f"MODERATE MISSING: {col} has {pct:.1f}% missing values")
    
    # Duplicate flags
    if quality_report['duplicates']['duplicate_percentage'] > 10:
        flags.append(f"HIGH DUPLICATES: {quality_report['duplicates']['duplicate_percentage']:.1f}% duplicate rows")
    elif quality_report['duplicates']['duplicate_percentage'] > 5:
        flags.append(f"MODERATE DUPLICATES: {quality_report['duplicates']['duplicate_percentage']:.1f}% duplicate rows")
    
    # Outlier flags
    for col, info in outlier_info.items():
        if info['outlier_percentage'] > 10:
            flags.append(f"HIGH OUTLIERS: {col} has {info['outlier_percentage']:.1f}% outliers")
    
    # Inconsistency flags
    for issue, count in inconsistencies.items():
        flags.append(f"INCONSISTENCY: {issue} - {count} occurrences")
    
    quality_report['summary_flags'] = flags
    
    return quality_report

def print_data_quality_summary(quality_report: dict) -> None:
    """Print a formatted summary of data quality issues."""
    print("=" * 60)
    print("DATA QUALITY ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Dataset Overview
    info = quality_report['dataset_info']
    print(f"\n DATASET OVERVIEW:")
    print(f"   Rows: {info['total_rows']:,}")
    print(f"   Columns: {info['total_columns']}")
    print(f"   Memory Usage: {info['memory_usage']/1024/1024:.2f} MB")
    
    # Missing Data
    missing = quality_report['missing_data']
    print(f"\n MISSING DATA:")
    print(f"   Total Missing Values: {missing['total_missing_values']:,}")
    print(f"   Rows with Missing Data: {missing['rows_with_any_missing']:,}")
    
    if missing['missing_percentages']:
        print("   Columns with Missing Values:")
        for col, pct in missing['missing_percentages'].items():
            print(f"     • {col}: {pct:.1f}%")
    
    # Duplicates
    dupes = quality_report['duplicates']
    print(f"\n DUPLICATE DATA:")
    print(f"   Duplicate Rows: {dupes['duplicate_rows']:,} ({dupes['duplicate_percentage']:.1f}%)")
    
    # Outliers
    print(f"\n OUTLIERS:")
    outliers = quality_report['outliers']
    if outliers:
        for col, info in outliers.items():
            if info['outlier_count'] > 0:
                print(f"   • {col}: {info['outlier_count']} outliers ({info['outlier_percentage']:.1f}%)")
    else:
        print("   No outliers detected")
    
    # Inconsistencies
    print(f"\n INCONSISTENCIES:")
    inconsistencies = quality_report['inconsistencies']
    if inconsistencies:
        for issue, count in inconsistencies.items():
            print(f"   • {issue}: {count} occurrences")
    else:
        print("   No inconsistencies detected")
    
    # Summary Flags
    print(f"\n CRITICAL FLAGS:")
    flags = quality_report['summary_flags']
    if flags:
        for flag in flags:
            print(f"   • {flag}")
    else:
        print("   No critical data quality issues detected")
    
    print("=" * 60)

