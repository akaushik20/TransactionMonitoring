import pandas as pd
import numpy as np
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
    """Calculate false positive rates by different dimensions."""
    
    # Check if alert_outcome column exists (this contains true_positive/false_positive)
    if 'alert_outcome' not in df.columns:
        print("No 'alert_outcome' column found. Cannot calculate false positive rate.")
        return
    
    # Calculate overall FPR using alert_outcome
    total_alerts = len(df)
    false_positives = (df['alert_outcome'] == 'false_positive').sum()
    true_positives = (df['alert_outcome'] == 'true_positive').sum()
    
    # FPR = False Positives / Total Alerts
    overall_fpr = false_positives / total_alerts if total_alerts > 0 else 0
    
    print(f"\n📊 FALSE POSITIVE RATE ANALYSIS")
    print("=" * 50)
    print(f"Total Alerts: {total_alerts:,}")
    print(f"False Positives: {false_positives:,}")
    print(f"True Positives: {true_positives:,}")
    print(f"False Positive Rate: {overall_fpr:.3f} ({overall_fpr*100:.1f}%)")
    
    # FPR by alert_type
    if 'alert_type' in df.columns:
        fpr_by_type = df.groupby('alert_type')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        ).sort_values(ascending=False)
        
        print("\nFPR by Alert Type:")
        for alert_type, fpr in fpr_by_type.items():
            count = ((df['alert_type'] == alert_type) & (df['alert_outcome'] == 'false_positive')).sum()
            total = (df['alert_type'] == alert_type).sum()
            print(f"  {alert_type}: {fpr:.3f} ({count}/{total})")
    
    # FPR by customer_risk_tier
    if 'customer_risk_tier' in df.columns:
        fpr_by_risk = df.groupby('customer_risk_tier')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        ).sort_values(ascending=False)
        
        print("\nFPR by Customer Risk Tier:")
        for risk_tier, fpr in fpr_by_risk.items():
            count = ((df['customer_risk_tier'] == risk_tier) & (df['alert_outcome'] == 'false_positive')).sum()
            total = (df['customer_risk_tier'] == risk_tier).sum()
            print(f"  {risk_tier}: {fpr:.3f} ({count}/{total})")
    
    # FPR by country
    if 'country' in df.columns:
        fpr_by_country = df.groupby('country')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        ).sort_values(ascending=False)
        
        # Show top 10 countries by FPR
        top_countries = fpr_by_country.head(10)
        
        print("\nFPR by Country (Top 10):")
        for country, fpr in top_countries.items():
            count = ((df['country'] == country) & (df['alert_outcome'] == 'false_positive')).sum()
            total = (df['country'] == country).sum()
            print(f"  {country}: {fpr:.3f} ({count}/{total})")
    
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
    
    report_content += "ADDITIONAL REPORTS\n------------------\n"
    report_content += "- EDA Report: transaction_monitoring_report.html\n"
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report_content)
    
    print(f"\n📄 Comprehensive FPR report saved as: {output_file}")
    return report_content

