import pandas as pd
import yaml

from helper.function import load_data, generate_eda_report

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

    ## Initial data exploration
    ## Using ydata-profiling for EDA
    generate_eda_report(df, "Transaction Monitoring Data Report", "transaction_monitoring_report.html")
    