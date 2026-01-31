import pandas as pd
import yaml

from helper.function import load_data, generate_eda_report, analyze_data_quality, print_data_quality_summary, calculate_false_positive_rates, generate_fpr_report

if __name__=="__main__":
    ## REad config
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    ## Load data
    df = pd.read_csv(config["INPUT_DATA_PATH"])

    ## Print data shape and statistics
    print("Data Shape:", df.shape)
    print("Data Statistics:\n", df.describe())
    print("Data Head:\n", df.head())

    ## Data Quality Analysis
    print("\n" + "="*50)
    print("PERFORMING DATA QUALITY ANALYSIS...")
    print("="*50)
    quality_report = analyze_data_quality(df)
    print_data_quality_summary(quality_report)

    ## False Positive Rate Analysis
    print("\n" + "="*60)
    print("GENERATING FALSE POSITIVE RATE ANALYSIS...")
    print("="*60)
    calculate_false_positive_rates(df)
    generate_fpr_report(df, "false_positive_analysis_report.txt")

    ## Initial data exploration
    ## Using ydata-profiling for EDA
    generate_eda_report(df, "Transaction Monitoring Data Report", "transaction_monitoring_report.html")
    