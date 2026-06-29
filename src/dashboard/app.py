"""
app.py  —  Streamlit Market Intelligence Dashboard
Run: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

st.set_page_config(page_title="IEX · HPX Market Intelligence", page_icon="⚡", layout="wide")

PROCESSED  = Path("data/processed/master_features.csv")
MODELS_DIR = Path("src/models")
FEATURES   = ["hour","day_of_week","month","is_weekend","is_peak",
               "mcp_lag1","mcp_roll7_mean","mcp_roll7_std",
               "mcv","buy_volume","sell_volume"]


@st.cache_data
def load_data():
    if not PROCESSED.exists():
        return None
    return pd.read_csv(PROCESSED, parse_dates=["date"])


df = load_data()

st.sidebar.title("⚡ Market Intelligence")
st.sidebar.markdown("IEX · HPX · India Power Exchanges")

if df is not None:
    exchanges  = st.sidebar.multiselect("Exchange", df["exchange"].unique(), default=df["exchange"].unique())
    date_range = st.sidebar.date_input("Date range", [df["date"].min(), df["date"].max()])
    df_f = df[
        df["exchange"].isin(exchanges) &
        df["date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    ]
else:
    df_f = None

st.title("⚡ Electricity Market Intelligence Dashboard")
st.caption("Short-term MCP forecasting · IEX India & HPX India · Internship Project 2026")

if df_f is None or df_f.empty:
    st.warning("No data yet. Run the pipeline first:")
    st.code("python src/ingestion/load_data.py\npython src/features/engineer.py\npython src/models/train.py")
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg MCP (₹/kWh)", f"{df_f['mcp'].mean():.2f}")
c2.metric("Volatility (std)", f"{df_f['mcp'].std():.2f}")
c3.metric("Avg MCV (MU)",     f"{df_f['mcv'].mean():.2f}")
c4.metric("Total Records",    f"{len(df_f):,}")

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 EDA", "🔀 IEX vs HPX", "🤖 Forecast"])

with tab1:
    daily = df_f.groupby(["date","exchange"])["mcp"].mean().reset_index()
    st.plotly_chart(px.line(daily, x="date", y="mcp", color="exchange",
                            title="Daily Average MCP", labels={"mcp":"MCP (₹/kWh)"}),
                    use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df_f, x="mcp", color="exchange",
                                     title="MCP Distribution", nbins=50),
                        use_container_width=True)
    with col2:
        day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
        wd = df_f.groupby(["day_of_week","exchange"])["mcp"].mean().reset_index()
        wd["day"] = wd["day_of_week"].map(day_map)
        st.plotly_chart(px.bar(wd, x="day", y="mcp", color="exchange",
                                barmode="group", title="Avg MCP by Weekday"),
                        use_container_width=True)

with tab3:
    spread = df_f.groupby(["date","exchange"])["mcp"].mean().unstack("exchange")
    if "IEX" in spread.columns and "HPX" in spread.columns:
        spread["spread"] = spread["IEX"] - spread["HPX"]
        fig = px.line(spread.reset_index(), x="date", y="spread",
                      title="IEX – HPX Price Spread", labels={"spread":"Spread (₹/kWh)"})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need both IEX and HPX data loaded for spread chart.")

with tab4:
    st.subheader("Next-Day MCP Forecast")
    model_path = MODELS_DIR / "randomforest.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        st.success("Random Forest model loaded.")
        last = df_f[FEATURES].dropna().tail(24)
        if not last.empty:
            preds = model.predict(last)
            st.metric("Forecast avg MCP (next day)", f"₹ {preds.mean():.2f} / kWh")
            st.plotly_chart(px.line(y=preds, labels={"y":"Forecast MCP (₹/kWh)","index":"Hour"},
                                     title="Forecasted Hourly MCP — Next Day"),
                            use_container_width=True)
    else:
        st.info("Run `python src/models/train.py` first to generate a model.")
