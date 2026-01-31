import pandas as pd
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

