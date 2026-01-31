import pandas as pd
import yaml
from ydata_profiling import ProfileReport

from helper.function import load_data

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
    profile = ProfileReport(df, title="Transaction Monitoring Data Report")
    profile.to_file("transaction_monitoring_report.html")
    print("EDA report saved as 'transaction_monitoring_report.html'")
    