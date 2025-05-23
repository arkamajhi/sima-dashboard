import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Title
st.title("SIMA Model Output Dashboard")

# Load or simulate data
def load_data():
    np.random.seed(42)
    stands = [f"Stand_{i}" for i in range(1, 11)]
    scenarios = ['REST', 'THIN', 'CLEAR']
    years = [2025, 2035, 2045]

    data = []
    for stand in stands:
        for scenario in scenarios:
            volume = np.random.uniform(150, 300)
            deadwood = np.random.uniform(5, 25)
            income = np.random.uniform(5000, 15000)
            carbon = np.random.uniform(10, 50)
            for year in years:
                data.append({
                    "stand_id": stand,
                    "scenario": scenario,
                    "year": year,
                    "volume": volume + np.random.normal(0, 10),
                    "deadwood": deadwood + np.random.normal(0, 2),
                    "income": income + np.random.normal(0, 500),
                    "carbon": carbon + np.random.normal(0, 5)
                })
    return pd.DataFrame(data)

# Load data
data = load_data()

# Sidebar for filtering
st.sidebar.header("Filter Options")
scenario_filter = st.sidebar.multiselect("Select Scenario(s)", options=data['scenario'].unique(), default=data['scenario'].unique())
year_filter = st.sidebar.selectbox("Select Year", options=sorted(data['year'].unique()))

# Filter data
filtered_data = data[(data['scenario'].isin(scenario_filter)) & (data['year'] == year_filter)]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Table", "Heatmap", "Motion Chart", "Export Data"])

# Tab 1: Data Table
with tab1:
    st.subheader("Filtered SIMA Data")
    st.dataframe(filtered_data)

# Tab 2: Heatmap
with tab2:
    st.subheader("Normalized Mean Outputs by Scenario")
    heatmap_data = filtered_data.groupby('scenario')[['volume', 'deadwood', 'income', 'carbon']].mean()
    normalized_heatmap = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(normalized_heatmap, annot=True, cmap='YlGnBu', cbar=True, ax=ax)
    st.pyplot(fig)

# Tab 3: Motion Chart
with tab3:
    st.subheader("Time Series of Outputs")
    chart_metric = st.selectbox("Select Metric", ['volume', 'deadwood', 'income', 'carbon'])
    line_data = data[data['scenario'].isin(scenario_filter)]
    fig = px.line(line_data, x='year', y=chart_metric, color='scenario', line_group='stand_id', hover_name='stand_id')
    fig.update_layout(title=f"{chart_metric.capitalize()} Over Time by Scenario")
    st.plotly_chart(fig)

# Tab 4: Export Data
with tab4:
    st.subheader("Export Filtered Data")
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='filtered_sima_data.csv',
        mime='text/csv',
    )

st.markdown("""
This dashboard allows exploration of simulated SIMA outputs by scenario and year.
Use the sidebar to filter and explore trade-offs between volume, deadwood, income, and carbon.
Navigate between tabs to see a data table, heatmap, motion chart, and download options.
""")
