import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_age_distribution(df):
    """Plots age distribution vs cardiovascular disease."""
    df_plot = df.copy()
    df_plot['age_years'] = (df_plot['age'] / 365.25).astype(int)
    fig = px.histogram(
        df_plot, x="age_years", color="cardio",
        labels={'age_years': 'Age (Years)', 'cardio': 'Cardiovascular Disease'},
        title="Age Distribution vs Cardiovascular Disease",
        barmode='group',
        template="plotly_dark",
        color_discrete_sequence=["#00D1FF", "#FF2E63"]
    )
    return fig

def plot_bmi_kde(df):
    """Plots BMI distribution."""
    fig = px.histogram(
        df, x="bmi", color="cardio", marginal="box",
        labels={'bmi': 'BMI', 'cardio': 'Cardiovascular Disease'},
        title="BMI Distribution vs Cardiovascular Disease",
        template="plotly_dark",
        color_discrete_sequence=["#00D1FF", "#FF2E63"]
    )
    return fig

def plot_correlation_heatmap(df):
    """Plots correlation heatmap of health metrics."""
    health_cols = [
        'age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 
        'cholesterol', 'gluc', 'bmi', 'pulse_pressure', 'cardio'
    ]
    # Filter only available health columns
    valid_cols = [col for col in health_cols if col in df.columns]
    corr = df[valid_cols].corr()
    
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Correlation Heatmap: Health Factors vs Risk",
        template="plotly_dark",
        color_continuous_scale='RdBu_r' # Red-Blue scale for correlation
    )
    return fig

def plot_bp_summary(df):
    """Plots blood pressure category summary."""
    df_plot = df.copy()
    df_plot['bp_label'] = df_plot['bp_category'].map({
        0: 'Normal', 1: 'Elevated', 2: 'Stage 1', 3: 'Stage 2'
    })
    counts = df_plot.groupby(['bp_label', 'cardio']).size().reset_index(name='count')
    fig = px.bar(
        counts, x="bp_label", y="count", color="cardio",
        labels={'bp_label': 'Blood Pressure Category', 'count': 'Number of Patients'},
        title="Blood Pressure Category vs Cardiovascular Disease",
        barmode='group',
        template="plotly_dark",
        color_discrete_sequence=["#00D1FF", "#FF2E63"]
    )
    return fig
