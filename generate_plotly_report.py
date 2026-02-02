import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

def create_interactive_report(df: pd.DataFrame, output_file: str = "transaction_monitoring_analysis.html") -> None:
    """Generate simple interactive plotly report for transaction monitoring analysis."""
    
    if 'alert_outcome' not in df.columns:
        print("Cannot create report: Missing 'alert_outcome' column")
        return
    
    # Calculate FPR data
    total_alerts = len(df)
    false_positives = (df['alert_outcome'] == 'false_positive').sum()
    true_positives = (df['alert_outcome'] == 'true_positive').sum()
    overall_fpr = false_positives / total_alerts
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Alert Distribution', 'FPR by Alert Type', 
                       'FPR by Risk Tier', 'FPR by Country'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. Alert Distribution Pie Chart
    fig.add_trace(
        go.Pie(
            labels=['False Positive', 'True Positive'],
            values=[false_positives, true_positives],
            marker=dict(colors=['#ff7f7f', '#7fbf7f'])
        ),
        row=1, col=1
    )
    
    # 2. FPR by Alert Type
    if 'alert_type' in df.columns:
        fpr_by_type = df.groupby('alert_type')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        ).sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=list(fpr_by_type.index),
                y=list(fpr_by_type.values),
                marker_color='lightblue',
                name='FPR by Alert Type',
                text=[f"{val:.1%}" for val in fpr_by_type.values],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. FPR by Customer Risk Tier
    if 'customer_risk_tier' in df.columns:
        fpr_by_risk = df.groupby('customer_risk_tier')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        ).sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=list(fpr_by_risk.index),
                y=list(fpr_by_risk.values),
                marker_color='lightgreen',
                name='FPR by Risk Tier',
                text=[f"{val:.1%}" for val in fpr_by_risk.values],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. FPR by Country (Top 10)
    if 'country' in df.columns:
        fpr_by_country = df.groupby('country')['alert_outcome'].apply(
            lambda x: (x == 'false_positive').sum() / len(x) if len(x) > 0 else 0
        ).sort_values(ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(
                x=list(fpr_by_country.index),
                y=list(fpr_by_country.values),
                marker_color='gold',
                name='FPR by Country',
                text=[f"{val:.1%}" for val in fpr_by_country.values],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text=f"Transaction Monitoring Model Analysis<br>Overall FPR: {overall_fpr:.1%}",
        title_x=0.5,
        height=800,
        showlegend=False
    )
    
    # Update y-axes to show percentages
    fig.update_yaxes(tickformat='.1%', row=1, col=2)
    fig.update_yaxes(tickformat='.1%', row=2, col=1)
    fig.update_yaxes(tickformat='.1%', row=2, col=2)
    
    # Ensure y-axes start at 0 and have reasonable range
    fig.update_yaxes(range=[0, 1], row=1, col=2)
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=2)
    
    # Section 2: Time to Disposition Analysis
    fig2 = go.Figure()
    
    # Simple bar chart showing average time by alert type
    if 'time_to_disposition_days' in df.columns and 'alert_type' in df.columns:
        avg_time_by_type = df.groupby('alert_type')['time_to_disposition_days'].mean().sort_values(ascending=False)
        
        fig2.add_trace(
            go.Bar(
                x=list(avg_time_by_type.index),
                y=list(avg_time_by_type.values),
                marker_color='lightcoral',
                text=[f"{val:.1f}" for val in avg_time_by_type.values],
                textposition='outside'
            )
        )
    
    fig2.update_layout(
        title="Section 2: Average Time to Disposition by Alert Type",
        title_x=0.5,
        height=400,
        xaxis_title="Alert Type",
        yaxis_title="Average Days",
        showlegend=False
    )
    
    # Section 3: Time Distribution by Alert Type (Box Plot)
    fig3 = go.Figure()
    
    if 'time_to_disposition_days' in df.columns and 'alert_type' in df.columns:
        for alert_type in df['alert_type'].unique():
            type_data = df[df['alert_type'] == alert_type]['time_to_disposition_days'].tolist()
            fig3.add_trace(
                go.Box(
                    y=type_data,
                    name=str(alert_type),
                    boxmean=True
                )
            )
    
    fig3.update_layout(
        title="Section 3: Time to Disposition Distribution by Alert Type",
        title_x=0.5,
        height=400,
        xaxis_title="Alert Type",
        yaxis_title="Days",
        showlegend=False
    )
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Transaction Monitoring Model Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .section {{ margin: 20px 0; }}
        .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 15px; background: #f0f0f0; border-radius: 5px; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #333; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Transaction Monitoring Model Analysis</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{overall_fpr:.1%}</div>
            <div>False Positive Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{false_positives:,}</div>
            <div>False Positives</div>
        </div>
        <div class="metric">
            <div class="metric-value">{true_positives:,}</div>
            <div>True Positives</div>
        </div>
        <div class="metric">
            <div class="metric-value">{total_alerts:,}</div>
            <div>Total Alerts</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Section 1: False Positive Rate Analysis by Different Dimensions</h2>
        <div id="charts">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    
    <div class="section">
        <h2>Section 2: Time to Disposition Analysis</h2>
        <div id="charts2">{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    
    <div class="section">
        <h2>Section 3: Time to Disposition Distribution by Alert Type</h2>
        <div id="charts3">{fig3.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
</body>
</html>
"""
    
    # Save HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📊 Interactive report saved as: {output_file}")

if __name__ == "__main__":
    # This allows the script to be run independently for testing
    
    
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    df = pd.read_csv(config["INPUT_DATA_PATH"])
    print(df.info())
    create_interactive_report(df)