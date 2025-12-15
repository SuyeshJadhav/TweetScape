import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

# === CONFIGURATION ===
st.set_page_config(
    page_title="Twitter Narrative Mapper",
    page_icon="🗺️",
    layout="wide"
)

# === LOAD DATA ===
def load_data():
    """
    Scans the 'data/' folder for clustered JSON files.
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    files = [f for f in os.listdir(data_dir) if f.startswith("clustered_") and f.endswith(".json")]
    
    if not files:
        return None, []
    
    return data_dir, files

# === SIDEBAR ===
st.sidebar.title("🛸 Narrative Radar")
data_dir, files = load_data()

if not files:
    st.error("No data found! Run the scraper and engine first.")
    st.stop()

# Dropdown to select topic
selected_file = st.sidebar.selectbox("Select Topic:", files)
topic_name = selected_file.replace("clustered_", "").replace(".json", "")

# === MAIN LOGIC ===
file_path = os.path.join(data_dir, selected_file)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# === THE VISUALIZATION ===
st.title(f"Topic Analysis: {topic_name}")

# Create columns for metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Tweets", len(df))
col2.metric("Distinct Tribes", df['cluster'].nunique())
col3.metric("Data Freshness", "Just Now")

st.markdown("---")

# THE MAP (Plotly)
# We map x/y to position, color to cluster, and hover_data to text
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster",
    hover_data={"handle": True, "text": True, "x": False, "y": False, "cluster": False},
    title=f"Semantic Landscape of '{topic_name}'",
    template="plotly_dark",
    height=700,
    size_max=20
)

# Customize the look (Remove grid lines for a 'Space' look)
fig.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    legend_title_text='Opinion Tribes'
)

# Make bubbles slightly larger for visibility
fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))

st.plotly_chart(fig, use_container_width=True)

# === RAW DATA TABLE ===
with st.expander("🔍 Drill Down into Tweets"):
    st.dataframe(
        df[["handle", "text", "cluster", "timestamp"]],
        use_container_width=True
    )