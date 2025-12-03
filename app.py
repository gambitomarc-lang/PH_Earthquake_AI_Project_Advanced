import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson
from datetime import datetime
import html

st.set_page_config(page_title="PH Earthquake AI — Advanced", layout="wide")

PHIVOLCS_TEXT_API = "https://earthquake.phivolcs.dost.gov.ph/fdsnws/event/1/query?format=text&limit=200"
MAP_CENTER = [12.8797, 121.7740]

@st.cache_data(ttl=60)
def load_cleaned_dataset(path: str = "phivolcs_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date_Time_PH"])
    df = df.rename(columns={
        "Depth_In_Km": "depth_km",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Magnitude": "magnitude"
    }, errors="ignore")
    return df

@st.cache_data(ttl=30)
def fetch_phivolcs_live(limit: int = 200) -> pd.DataFrame:
    try:
        resp = requests.get(PHIVOLCS_TEXT_API.replace("limit=200", f"limit={limit}"), timeout=8)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        if not lines:
            return pd.DataFrame()
        header = lines[0].split("|")
        records = []
        for ln in lines[1:]:
            parts = ln.split("|")
            if len(parts) < 5:
                continue
            try:
                records.append({
                    "datetime": pd.to_datetime(parts[0]),
                    "latitude": float(parts[1]),
                    "longitude": float(parts[2]),
                    "depth_km": float(parts[3]),
                    "magnitude": float(parts[4]),
                })
            except Exception:
                continue
        df = pd.DataFrame(records)
        return df.sort_values("datetime", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def build_folium_map(events, show_heatmap=True, heat_radius=12, heat_blur=8, cluster=True, min_mag=0.0):
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    m = folium.Map(location=MAP_CENTER, zoom_start=5, tiles="CartoDB positron")
    if events is None or events.empty:
        return m
    events = events[events["magnitude"] >= min_mag]

    if show_heatmap and not events.empty:
        HeatMap(events[["latitude","longitude","magnitude"]].values.tolist(),
                radius=heat_radius, blur=heat_blur).add_to(m)

    if cluster:
        mc = MarkerCluster().add_to(m)
        for _, r in events.iterrows():
            color = "red" if r["magnitude"]>=5 else "orange" if r["magnitude"]>=4 else "blue"
            popup = f"<b>{r['datetime']}</b><br>Mag: {r['magnitude']}<br>Depth: {r['depth_km']} km"
            folium.CircleMarker([r["latitude"], r["longitude"]],
                                radius=max(3, r["magnitude"]*1.3),
                                color=color, fill=True,
                                fill_opacity=0.7, popup=popup).add_to(mc)
    else:
        for _, r in events.iterrows():
            color = "red" if r["magnitude"]>=5 else "orange" if r["magnitude"]>=4 else "blue"
            folium.CircleMarker([r["latitude"], r["longitude"]],
                                radius=max(3, r["magnitude"]*1.3),
                                color=color, fill=True,
                                fill_opacity=0.7).add_to(m)
    # legend
    legend_html = """
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 160px; height: 110px;
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white; opacity:0.9; padding:8px;">
     <b>Magnitude</b><br>
     <i style="color:red">●</i> >= 5.0<br>
     <i style="color:orange">●</i> 4.0 - 4.9<br>
     <i style="color:blue">●</i> &lt; 4.0
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

hist_df = load_cleaned_dataset()
try:
    model = joblib.load("earthquake_rf_model.pkl")
except Exception:
    model = None

page = st.sidebar.radio("Navigation", ["Map", "Predict", "Advanced Live Feed"])

if page == "Map":
    st.header("📍 Earthquake Map — Heatmap + Clustering")
    show_heat = st.checkbox("Heatmap", True)
    use_cluster = st.checkbox("Cluster markers", True)
    min_mag = st.slider("Minimum magnitude", 0.0, 8.0, 0.0, 0.1)

    df = hist_df.rename(columns={"Date_Time_PH":"datetime"}) if "Date_Time_PH" in hist_df.columns else hist_df
    st.write(f"Showing {len(df)} events")

    m = build_folium_map(df, show_heat, 12, 8, use_cluster, min_mag)
    st_folium(m, width=1150, height=650)

elif page == "Predict":
    st.header("🤖 Earthquake Magnitude Prediction")
    lat = st.number_input("Latitude", value=12.0)
    lon = st.number_input("Longitude", value=121.0)
    depth = st.number_input("Depth (km)", value=10.0)
    now = datetime.now()
    feat = np.array([[lat,lon,depth,now.year,now.month,now.day,now.hour]])
    if st.button("Predict"):
        if model is None:
            st.warning("Model not found")
        else:
            pred = model.predict(feat)[0]
            st.metric("Predicted Magnitude", round(float(pred),2))

elif page == "Advanced Live Feed":
    st.header("📡 Advanced PHIVOLCS Live Feed")
    limit = st.selectbox("Max events", [50,100,200,500], index=2)
    enable_heat = st.checkbox("Heatmap", True)
    enable_cluster = st.checkbox("Cluster markers", True)
    min_mag = st.slider("Minimum magnitude", 0.0, 8.0, 0.0, 0.1)
    auto_refresh = st.checkbox("Auto-refresh", False)
    interval = st.selectbox("Refresh interval (sec)", [10,20,30,60], index=2)

    if auto_refresh:
        st.components.v1.html(f"<script>setTimeout(()=>window.location.reload(), {interval*1000});</script>", height=0)

    live_df = fetch_phivolcs_live(limit)
    if live_df.empty:
        st.error("No live data available.")
    else:
        st.dataframe(live_df.head(200))
        m = build_folium_map(live_df, enable_heat, 12, 8, enable_cluster, min_mag)
        st_folium(m, width=1150, height=650)
