import pandas as pd
import yaml

from helper.function import load_data, generate_eda_report, calculate_false_positive_rates, generate_fpr_report
from generate_plotly_report import create_interactive_report

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

    ## False Positive Rate Analysis
    print("\n" + "="*60)
    print("GENERATING FALSE POSITIVE RATE ANALYSIS...")
    print("="*60)
    calculate_false_positive_rates(df)
    generate_fpr_report(df, "false_positive_analysis_report.txt")

    ## Interactive Plotly Report
    print("\n" + "="*60)
    print("GENERATING INTERACTIVE PLOTLY REPORT...")
    print("="*60)
    create_interactive_report(df, "transaction_monitoring_analysis.html")

    ## Initial data exploration
    ## Using ydata-profiling for EDA
    generate_eda_report(df, "Transaction Monitoring Data Report", "EDA_report.html")
    