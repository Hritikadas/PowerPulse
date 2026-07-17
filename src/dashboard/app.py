"""
app.py  —  Streamlit Market Intelligence Dashboard
Run: streamlit run src/dashboard/app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="IEX · HPX Market Intelligence", page_icon="⚡", layout="wide")

PROCESSED  = Path("data/processed/master_features.csv")
MODELS_DIR = Path("src/models")
FEATURES   = [
    "hour", "day_of_week", "month", "is_weekend", "is_peak",
    "mcp_lag1", "mcp_roll7_mean", "mcp_roll7_std",
    "mcv", "buy_volume", "sell_volume",
]


@st.cache_data
def load_data():
    if not PROCESSED.exists():
        return None
    return pd.read_csv(PROCESSED, parse_dates=["date"])


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Market Intelligence")
st.sidebar.markdown("IEX · HPX · India Power Exchanges")

if df is not None:
    exchanges  = st.sidebar.multiselect(
        "Exchange", df["exchange"].unique(), default=df["exchange"].unique()
    )
    date_range = st.sidebar.date_input(
        "Date range", [df["date"].min(), df["date"].max()]
    )
    df_f = df[
        df["exchange"].isin(exchanges) &
        df["date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    ]
else:
    df_f = None

# ── Title ──────────────────────────────────────────────────────────────────
st.title("⚡ Electricity Market Intelligence Dashboard")
st.caption("Short-term MCP forecasting · IEX India & HPX India · Internship Project 2026")

if df_f is None or df_f.empty:
    st.warning("No data yet. Run the pipeline first:")
    st.code(
        "python src/ingestion/load_data.py\n"
        "python src/features/engineer.py\n"
        "python src/models/train.py"
    )
    st.stop()

# ── KPI Cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg MCP (₹/kWh)", f"{df_f['mcp'].mean():.2f}")
c2.metric("Volatility (std)", f"{df_f['mcp'].std():.2f}")
c3.metric("Avg MCV (MU)",     f"{df_f['mcv'].mean():.2f}")
c4.metric("Total Records",    f"{len(df_f):,}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 EDA", "🔀 IEX vs HPX", "🤖 Forecast"])

# ── Tab 1: Trends ──────────────────────────────────────────────────────────
with tab1:
    daily = df_f.groupby(["date", "exchange"])["mcp"].mean().reset_index()
    st.plotly_chart(
        px.line(
            daily, x="date", y="mcp", color="exchange",
            title="Daily Average MCP Over Time",
            labels={"mcp": "MCP (₹/kWh)", "date": "Date"},
        ),
        use_container_width=True,
    )

    # Monthly average trend
    df_f["month_period"] = df_f["date"].dt.to_period("M").astype(str)
    monthly = df_f.groupby(["month_period", "exchange"])["mcp"].mean().reset_index()
    st.plotly_chart(
        px.line(
            monthly, x="month_period", y="mcp", color="exchange",
            title="Monthly Average MCP Trend",
            labels={"mcp": "MCP (₹/kWh)", "month_period": "Month"},
            markers=True,
        ),
        use_container_width=True,
    )

# ── Tab 2: EDA ─────────────────────────────────────────────────────────────
with tab2:
    # Row 1: Distribution + Weekday breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.histogram(
                df_f, x="mcp", color="exchange",
                title="MCP Price Distribution",
                nbins=50, barmode="overlay", opacity=0.7,
                labels={"mcp": "MCP (₹/kWh)"},
            ),
            use_container_width=True,
        )

    with col2:
        day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        wd = df_f.groupby(["day_of_week", "exchange"])["mcp"].mean().reset_index()
        wd["day"] = wd["day_of_week"].map(day_map)
        wd["day"] = pd.Categorical(wd["day"], categories=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        wd = wd.sort_values("day")
        st.plotly_chart(
            px.bar(
                wd, x="day", y="mcp", color="exchange",
                barmode="group", title="Avg MCP by Day of Week",
                labels={"mcp": "Avg MCP (₹/kWh)", "day": "Day"},
            ),
            use_container_width=True,
        )

    # Row 2: Weekday vs Weekend + Hour-of-day profile
    col3, col4 = st.columns(2)

    with col3:
        wk = df_f.copy()
        wk["Day Type"] = wk["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        wk_avg = wk.groupby(["Day Type", "exchange"])["mcp"].mean().reset_index()
        st.plotly_chart(
            px.bar(
                wk_avg, x="exchange", y="mcp", color="Day Type",
                barmode="group", title="Weekday vs Weekend Average MCP",
                labels={"mcp": "Avg MCP (₹/kWh)"},
            ),
            use_container_width=True,
        )

    with col4:
        hourly = df_f.groupby(["hour", "exchange"])["mcp"].mean().reset_index()
        st.plotly_chart(
            px.line(
                hourly, x="hour", y="mcp", color="exchange",
                title="Avg MCP by Hour of Day",
                labels={"mcp": "Avg MCP (₹/kWh)", "hour": "Hour"},
                markers=True,
            ),
            use_container_width=True,
        )

    # Row 3: Buy vs Sell Volume over time
    st.subheader("Buy vs Sell Volume Over Time")
    if "buy_volume" in df_f.columns and "sell_volume" in df_f.columns:
        vol = df_f.groupby(["date", "exchange"])[["buy_volume", "sell_volume"]].sum().reset_index()
        vol_melted = vol.melt(
            id_vars=["date", "exchange"],
            value_vars=["buy_volume", "sell_volume"],
            var_name="Type", value_name="Volume (MWh)",
        )
        vol_melted["Type"] = vol_melted["Type"].map(
            {"buy_volume": "Buy Volume", "sell_volume": "Sell Volume"}
        )
        st.plotly_chart(
            px.line(
                vol_melted, x="date", y="Volume (MWh)", color="Type",
                facet_col="exchange", title="Buy vs Sell Volume Over Time",
            ),
            use_container_width=True,
        )
    else:
        st.info("Buy/sell volume columns not found in dataset.")

    # Row 4: Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    num_cols = ["mcp", "mcv", "buy_volume", "sell_volume",
                "mcp_lag1", "mcp_roll7_mean", "mcp_roll7_std"]
    available = [c for c in num_cols if c in df_f.columns]
    corr = df_f[available].corr().round(2)
    fig_corr = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        title="Correlation Heatmap", aspect="auto",
        zmin=-1, zmax=1,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 3: IEX vs HPX ──────────────────────────────────────────────────────
with tab3:
    spread_base = df_f.groupby(["date", "exchange"])["mcp"].mean().unstack("exchange")

    if "IEX" in spread_base.columns and "HPX" in spread_base.columns:
        spread_base["spread"] = spread_base["IEX"] - spread_base["HPX"]
        spread_reset = spread_base.reset_index()

        fig_spread = px.line(
            spread_reset, x="date", y="spread",
            title="IEX − HPX Price Spread Over Time",
            labels={"spread": "Spread (₹/kWh)", "date": "Date"},
        )
        fig_spread.add_hline(y=0, line_dash="dash", line_color="gray",
                             annotation_text="Zero line")
        st.plotly_chart(fig_spread, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                px.histogram(
                    spread_reset, x="spread",
                    title="Distribution of IEX–HPX Spread",
                    labels={"spread": "Spread (₹/kWh)"},
                    nbins=40,
                ),
                use_container_width=True,
            )
        with col_b:
            both = spread_reset[["IEX", "HPX"]].dropna()
            both_melted = both.reset_index().melt(
                id_vars="index", value_vars=["IEX", "HPX"],
                var_name="Exchange", value_name="MCP (₹/kWh)",
            )
            st.plotly_chart(
                px.box(
                    both_melted, x="Exchange", y="MCP (₹/kWh)",
                    color="Exchange", title="MCP Box Plot: IEX vs HPX",
                ),
                use_container_width=True,
            )
    else:
        st.info("Need both IEX and HPX data loaded for spread analysis.")
        # Still show individual exchange stats
        st.subheader("Available Exchange Data")
        st.dataframe(
            df_f.groupby("exchange")["mcp"].agg(["mean", "std", "min", "max"]).round(3),
            use_container_width=True,
        )

# ── Tab 4: Forecast ────────────────────────────────────────────────────────
with tab4:
    model_path = MODELS_DIR / "randomforest.pkl"

    if not model_path.exists():
        st.info("Run `python src/models/train.py` first to generate a model.")
        st.stop()

    model = load_model(model_path)
    st.success("✅ Random Forest model loaded.")

    # ── Section A: Next-day forecast
    st.subheader("Next-Day Hourly MCP Forecast")
    last = df_f[FEATURES].dropna().tail(24)

    if not last.empty:
        preds = model.predict(last)
        forecast_df = pd.DataFrame({
            "Hour": range(1, len(preds) + 1),
            "Forecast MCP (₹/kWh)": preds,
        })

        fa, fb = st.columns([2, 1])
        with fa:
            st.plotly_chart(
                px.line(
                    forecast_df, x="Hour", y="Forecast MCP (₹/kWh)",
                    title="Forecasted Hourly MCP — Next Day",
                    markers=True,
                ),
                use_container_width=True,
            )
        with fb:
            st.metric("Forecast Avg MCP", f"₹ {preds.mean():.2f} / kWh")
            st.metric("Forecast Peak MCP", f"₹ {preds.max():.2f} / kWh")
            st.metric("Forecast Min MCP",  f"₹ {preds.min():.2f} / kWh")
            st.dataframe(forecast_df.set_index("Hour").round(3), use_container_width=True)
    else:
        st.warning("Not enough feature data to generate forecast.")

    st.divider()

    # ── Section B: Model validation — Predicted vs Actual
    st.subheader("Model Validation — Predicted vs Actual (Test Set)")

    df_model = df[FEATURES + ["mcp"]].dropna()
    X = df_model[FEATURES]
    y = df_model["mcp"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    y_pred = model.predict(X_test)

    # Metrics row
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE  (₹/kWh)", f"{mae:.4f}")
    m2.metric("RMSE (₹/kWh)", f"{rmse:.4f}")
    m3.metric("R² Score",      f"{r2:.4f}")

    # Predicted vs Actual chart
    n_plot = st.slider("Points to display", min_value=50, max_value=min(500, len(y_test)),
                       value=200, step=50)
    val_df = pd.DataFrame({
        "Actual":    y_test.values[-n_plot:],
        "Predicted": y_pred[-n_plot:],
    }).reset_index(drop=True)
    val_df.index.name = "Time Step"

    st.plotly_chart(
        px.line(
            val_df.reset_index(), x="Time Step", y=["Actual", "Predicted"],
            title=f"Actual vs Predicted MCP — last {n_plot} test points",
            labels={"value": "MCP (₹/kWh)"},
        ),
        use_container_width=True,
    )

    # Error distribution
    val_df["Error"] = val_df["Actual"] - val_df["Predicted"]
    st.plotly_chart(
        px.histogram(
            val_df, x="Error", nbins=40,
            title="Prediction Error Distribution",
            labels={"Error": "Actual − Predicted (₹/kWh)"},
        ),
        use_container_width=True,
    )

    st.divider()

    # ── Section C: Feature importances
    st.subheader("Feature Importances (Random Forest)")
    if hasattr(model, "feature_importances_"):
        imp = (
            pd.Series(model.feature_importances_, index=FEATURES)
            .sort_values(ascending=True)
            .reset_index()
        )
        imp.columns = ["Feature", "Importance"]
        st.plotly_chart(
            px.bar(
                imp, x="Importance", y="Feature", orientation="h",
                title="Random Forest — Feature Importances",
                color="Importance", color_continuous_scale="Blues",
            ),
            use_container_width=True,
        )