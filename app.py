"""
Investigate Hotel Business using Data Visualization — Interactive Streamlit Dashboard
Author: P Suman
Answers the three brief business questions, then extends into a portfolio-level BI dashboard:
  1. Which hotel type do customers book most often, and how does demand move through the year?
  2. Does length of stay affect the cancellation rate?
  3. Does lead time (booking-to-arrival gap) affect the cancellation rate?
Plus: revenue impact, geography, customer segments, a simple trend forecast, a cancellation
what-if simulator, a hotel-vs-hotel comparison, and a data-quality summary.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Data: place `hotel_bookings_data.csv` (or `hotel_bookings.csv`) next to this file,
or upload it from the sidebar when the app starts.
"""

import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------------------
# Page config & constants
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Booking & Cancellation Insights",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_MAP = {"City Hotel": "#4C72B0", "Resort Hotel": "#DD8452"}
MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
CANDIDATE_FILES = ["hotel_bookings_data.csv", "hotel_bookings.csv", "hotel_bookings_clean.csv"]

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.5rem; }
.block-container { padding-top: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Data loading & cleaning — mirrors the notebook's Stage 1 preprocessing
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning booking data...")
def load_and_clean(file_bytes_or_path):
    raw = pd.read_csv(file_bytes_or_path)
    df = raw.copy()

    # --- data-quality snapshot (measured BEFORE cleaning) --------------------
    quality = {"rows_before": len(raw), "cols": raw.shape[1]}
    missing = raw.isnull().sum()
    quality["missing"] = missing[missing > 0].sort_values(ascending=False)
    quality["duplicates"] = int(raw.duplicated().sum())
    quality["negative_adr"] = int((raw["adr"] < 0).sum()) if "adr" in raw.columns else 0
    guest_cols0 = [c for c in ["adults", "children", "babies"] if c in raw.columns]
    quality["zero_guest_rows"] = int((raw[guest_cols0].sum(axis=1) == 0).sum()) if guest_cols0 else 0

    # --- missing values -------------------------------------------------
    for col, fill in [("company", 0), ("agent", 0), ("children", 0)]:
        if col in df.columns:
            df[col] = df[col].fillna(fill)
    if "city" in df.columns:
        df["city"] = df["city"].fillna("Unknown")
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")

    # --- duplicates -------------------------------------------------------
    df = df.drop_duplicates()

    # --- inconsistent categories ------------------------------------------
    if "meal" in df.columns:
        df["meal"] = df["meal"].replace("Undefined", "No Meal")

    # --- anomalies ----------------------------------------------------------
    if "adr" in df.columns:
        df = df[df["adr"] >= 0]
    guest_cols = [c for c in ["adults", "children", "babies"] if c in df.columns]
    if guest_cols:
        df = df[df[guest_cols].sum(axis=1) > 0]

    # --- derived fields -----------------------------------------------------
    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(df.columns):
        df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    elif {"stays_in_weekend_nights", "stays_in_weekdays_nights"}.issubset(df.columns):
        df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_weekdays_nights"]

    if "arrival_date_month" in df.columns:
        df["arrival_date_month"] = pd.Categorical(
            df["arrival_date_month"], categories=MONTH_ORDER, ordered=True
        )
        df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_NUM)

    if "arrival_date_year" in df.columns and "arrival_month_num" in df.columns:
        df["period"] = df["arrival_date_year"].astype(int) * 12 + df["arrival_month_num"].astype(int)
        df["period_label"] = (df["arrival_date_month"].astype(str).str.slice(0, 3) + " "
                               + df["arrival_date_year"].astype(str))

    if "lead_time" in df.columns:
        bins = [-1, 7, 30, 90, 180, 365, 10000]
        labels = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days"]
        df["lead_bin"] = pd.cut(df["lead_time"], bins=bins, labels=labels)

    if "total_stay" in df.columns:
        def stay_group(n):
            if n <= 7:
                return "<= 1 Week"
            elif n <= 14:
                return "2 Weeks"
            elif n <= 21:
                return "3 Weeks"
            return ">= 4 Weeks"
        df["stay_duration_group"] = df["total_stay"].apply(stay_group)

    if {"adr", "total_stay"}.issubset(df.columns):
        df["estimated_revenue"] = df["adr"] * df["total_stay"]

    if "children" in df.columns and "adults" in df.columns:
        def guest_type(row):
            if row["children"] > 0:
                return "Family"
            if row["adults"] == 2:
                return "Couple"
            return "Solo / Other"
        df["guest_type"] = df.apply(guest_type, axis=1)

    quality["rows_after"] = len(df)
    return df, quality


def find_default_file():
    for name in CANDIDATE_FILES:
        if os.path.exists(name):
            return name
    return None


# --------------------------------------------------------------------------------------
# ML cancellation-risk model — trained once on the full cleaned dataset, cached as a
# resource so filtering the dashboard never retriggers a retrain.
# --------------------------------------------------------------------------------------
ML_NUMERIC_CANDIDATES = [
    "lead_time", "total_stay", "adr", "adults", "children", "babies",
    "previous_cancellations", "previous_bookings_not_canceled", "booking_changes",
    "total_of_special_requests", "required_car_parking_spaces", "is_repeated_guest",
    "days_in_waiting_list",
]
ML_CATEGORICAL_CANDIDATES = [
    "hotel", "meal", "market_segment", "distribution_channel", "deposit_type",
    "customer_type", "reserved_room_type",
]


@st.cache_resource(show_spinner="Training cancellation-risk model...")
def train_cancellation_model(df: pd.DataFrame):
    """Fit a Random Forest cancellation classifier on whatever relevant columns
    exist in this dataset. Returns None if `is_canceled` isn't present."""
    if "is_canceled" not in df.columns:
        return None

    num_cols = [c for c in ML_NUMERIC_CANDIDATES if c in df.columns]
    cat_cols = [c for c in ML_CATEGORICAL_CANDIDATES if c in df.columns]
    if len(num_cols) + len(cat_cols) < 3:
        return None

    model_df = df[num_cols + cat_cols + ["is_canceled"]].dropna()
    if model_df["is_canceled"].nunique() < 2 or len(model_df) < 200:
        return None

    X = model_df[num_cols + cat_cols]
    y = model_df["is_canceled"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1, class_weight="balanced",
        )),
    ])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # feature importance mapped back to readable names
    ohe = clf.named_steps["preprocess"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if cat_cols else []
    all_feature_names = num_cols + cat_feature_names
    importances = clf.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    return {
        "pipeline": clf,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "metrics": metrics,
        "importance": imp_df,
    }


# --------------------------------------------------------------------------------------
# Sidebar — data source + filters
# --------------------------------------------------------------------------------------
st.sidebar.title("🏨 Hotel Insights")
st.sidebar.caption("Investigate Hotel Business — Data Visualization Capstone")

default_path = find_default_file()
uploaded = st.sidebar.file_uploader("Upload hotel_bookings CSV", type=["csv"])

data_source = uploaded if uploaded is not None else default_path

if data_source is None:
    st.title("🏨 Hotel Booking & Cancellation Insights")
    st.info(
        "No dataset found. Upload the **hotel_bookings** CSV (2017-2019, ~119K rows) "
        "from the sidebar to load the dashboard, or place it next to `app.py` as "
        "`hotel_bookings_data.csv` before deploying."
    )
    st.stop()

df, quality = load_and_clean(data_source)
ml_bundle = train_cancellation_model(df)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

hotels = sorted(df["hotel"].dropna().unique().tolist()) if "hotel" in df.columns else []
sel_hotels = st.sidebar.multiselect("Hotel type", hotels, default=hotels)

years = sorted(df["arrival_date_year"].dropna().unique().tolist()) if "arrival_date_year" in df.columns else []
sel_years = st.sidebar.multiselect("Arrival year", years, default=years)

months_present = [m for m in MONTH_ORDER if m in df["arrival_date_month"].astype(str).unique()] \
    if "arrival_date_month" in df.columns else []
sel_months = st.sidebar.multiselect("Arrival month", months_present, default=months_present)

booking_status = st.sidebar.radio(
    "Booking status", ["All bookings", "Cancelled only", "Honoured only"], index=0
)

# Apply filters
fdf = df.copy()
if sel_hotels:
    fdf = fdf[fdf["hotel"].isin(sel_hotels)]
if sel_years:
    fdf = fdf[fdf["arrival_date_year"].isin(sel_years)]
if sel_months:
    fdf = fdf[fdf["arrival_date_month"].astype(str).isin(sel_months)]
if booking_status == "Cancelled only" and "is_canceled" in fdf.columns:
    fdf = fdf[fdf["is_canceled"] == 1]
elif booking_status == "Honoured only" and "is_canceled" in fdf.columns:
    fdf = fdf[fdf["is_canceled"] == 0]

st.sidebar.markdown("---")
st.sidebar.download_button(
    "⬇ Download filtered data (CSV)",
    data=fdf.to_csv(index=False).encode("utf-8"),
    file_name="hotel_bookings_filtered.csv",
    mime="text/csv",
)
st.sidebar.caption(f"{len(fdf):,} of {len(df):,} bookings shown")

if fdf.empty:
    st.warning("No bookings match the current filters — widen your selection in the sidebar.")
    st.stop()

has_revenue = "estimated_revenue" in fdf.columns
has_country = "country" in fdf.columns
has_segment = "guest_type" in fdf.columns
has_period = "period" in fdf.columns

# --------------------------------------------------------------------------------------
# Header + top-level KPIs
# --------------------------------------------------------------------------------------
st.title("🏨 Hotel Booking & Cancellation Insights")
st.caption("Interactive companion to the *Investigate Hotel Business using Data Visualization* notebook — 2017-2019 bookings")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total bookings", f"{len(fdf):,}")
if "is_canceled" in fdf.columns:
    k2.metric("Cancellation rate", f"{fdf['is_canceled'].mean()*100:.1f}%")
if "lead_time" in fdf.columns:
    k3.metric("Avg. lead time", f"{fdf['lead_time'].mean():.0f} days")
if "adr" in fdf.columns:
    k4.metric("Avg. daily rate (ADR)", f"{fdf['adr'].mean():.0f}")
if "total_stay" in fdf.columns:
    k5.metric("Avg. stay length", f"{fdf['total_stay'].mean():.1f} nights")
if has_revenue:
    k6.metric("Est. total revenue", f"{fdf['estimated_revenue'].sum():,.0f}")

st.markdown("---")

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
has_market = "market_segment" in fdf.columns
has_ml = ml_bundle is not None

tab_names = ["📊 Executive Overview", "① Hotel Type & Seasonality", "② Stay Duration → Cancellation",
             "③ Lead Time → Cancellation", "💰 Revenue"]
if has_country:
    tab_names.append("🌍 Country Map")
if has_segment:
    tab_names.append("👥 Customer Segments")
if has_market:
    tab_names.append("📊 Market Intelligence")
tab_names.append("🔗 Correlation Analysis")
if has_period:
    tab_names.append("🔮 Forecast")
tab_names += ["🧪 What-If Simulator", "⚖ Hotel Comparison"]
if has_ml:
    tab_names += ["🤖 ML Cancellation Prediction", "🔥 Risk Analysis"]
tab_names += ["🤖 AI Business Assistant", "🧹 Data Quality", "📋 Summary & Recommendations", "📥 Export Reports"]

tabs = st.tabs(tab_names)
tab_map = dict(zip(tab_names, tabs))

# ---- Executive Overview -----------------------------------------------------------------
with tab_map["📊 Executive Overview"]:
    st.subheader("Executive snapshot")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Bookings", f"{len(fdf):,}")
    e2.metric("Cancellation rate", f"{fdf['is_canceled'].mean()*100:.1f}%" if "is_canceled" in fdf.columns else "—")
    if has_revenue:
        lost = fdf.loc[fdf["is_canceled"] == 1, "estimated_revenue"].sum() if "is_canceled" in fdf.columns else 0
        e3.metric("Est. revenue", f"{fdf['estimated_revenue'].sum():,.0f}")
        e4.metric("Revenue lost to cancellations", f"{lost:,.0f}")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        if has_period:
            trend = (fdf.groupby(["period", "period_label"], observed=True)
                     .size().reset_index(name="bookings").sort_values("period"))
            fig = px.area(trend, x="period_label", y="bookings", title="Booking Trend Over Time")
            fig.update_layout(template="plotly_white", title_x=0.5, xaxis_title="", yaxis_title="Bookings")
            st.plotly_chart(fig, use_container_width=True)
        else:
            monthly_bookings = fdf.groupby("arrival_date_month", observed=True).size().reset_index(name="bookings")
            fig = px.area(monthly_bookings, x="arrival_date_month", y="bookings", title="Booking Trend Over Time")
            fig.update_layout(template="plotly_white", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if has_country:
            top_countries = fdf["country"].value_counts().head(8).reset_index()
            top_countries.columns = ["country", "bookings"]
            fig = px.bar(top_countries, x="bookings", y="country", orientation="h",
                         title="Top Origin Countries")
            fig.update_layout(template="plotly_white", title_x=0.5, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        elif has_segment:
            seg = fdf["guest_type"].value_counts().reset_index()
            seg.columns = ["guest_type", "bookings"]
            fig = px.pie(seg, names="guest_type", values="bookings", title="Bookings by Guest Type", hole=0.4)
            fig.update_layout(template="plotly_white", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🤖 Auto-generated insights")
    if st.button("Generate insights", key="exec_insights_btn"):
        share = fdf["hotel"].value_counts(normalize=True).mul(100) if "hotel" in fdf.columns else pd.Series(dtype=float)
        top_hotel = share.idxmax() if not share.empty else "the leading hotel type"
        top_pct = share.max() if not share.empty else 0
        lines = [f"**{top_hotel}** contributes **{top_pct:.1f}%** of bookings in the current selection."]
        if "lead_bin" in fdf.columns and "is_canceled" in fdf.columns:
            by_lead = fdf.groupby("lead_bin", observed=True)["is_canceled"].mean().mul(100)
            if not by_lead.empty:
                riskiest = by_lead.idxmax()
                lines.append(f"Cancellation risk is highest for bookings made **{riskiest}** before arrival "
                              f"(**{by_lead.max():.1f}%** cancelled).")
        if has_revenue and "is_canceled" in fdf.columns:
            lost = fdf.loc[fdf["is_canceled"] == 1, "estimated_revenue"].sum()
            lines.append(f"An estimated **{lost:,.0f}** in revenue was lost to cancellations in this selection.")
        st.success("\n\n".join(lines))
    else:
        st.caption("Click the button for a plain-language summary of the current filtered view.")

# ---- Hotel type & seasonality -----------------------------------------------------------
with tab_map["① Hotel Type & Seasonality"]:
    st.subheader("Which hotel type do customers book most often?")
    c1, c2 = st.columns([1, 1.4])

    with c1:
        share = fdf["hotel"].value_counts().reset_index()
        share.columns = ["hotel", "bookings"]
        fig = px.pie(share, names="hotel", values="bookings", color="hotel",
                     color_discrete_map=COLOR_MAP, hole=0.45,
                     title="Share of Bookings by Hotel Type")
        fig.update_traces(textinfo="percent+label",
                           hovertemplate="%{label}: %{value:,} bookings (%{percent})")
        fig.update_layout(template="plotly_white", title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        monthly = (fdf.groupby(["arrival_date_month", "hotel"], observed=True)
                   .size().reset_index(name="bookings"))
        fig = px.line(monthly, x="arrival_date_month", y="bookings", color="hotel",
                      color_discrete_map=COLOR_MAP, markers=True,
                      title="Bookings per Month by Hotel Type")
        fig.update_layout(template="plotly_white", title_x=0.5, hovermode="x unified",
                           xaxis_title="Arrival Month", yaxis_title="Number of Bookings",
                           legend_title="Hotel Type")
        st.plotly_chart(fig, use_container_width=True)

    if "arrival_date_year" in fdf.columns:
        heat = (fdf.groupby(["arrival_date_year", "arrival_date_month"], observed=True)
                .size().reset_index(name="bookings"))
        fig = px.density_heatmap(heat, x="arrival_date_month", y="arrival_date_year",
                                  z="bookings", histfunc="sum", color_continuous_scale="Blues",
                                  title="Booking Volume Heatmap: Month x Year")
        fig.update_layout(template="plotly_white", title_x=0.5, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    top_hotel = share.iloc[0]["hotel"] if not share.empty else "—"
    top_pct = share.iloc[0]["bookings"] / share["bookings"].sum() * 100 if not share.empty else 0
    st.info(f"**Insight:** {top_hotel} accounts for **{top_pct:.0f}%** of bookings in the current "
            "selection. Demand for both hotel types follows the same broad seasonal shape, "
            "peaking in the autumn months and dipping January-March.")

# ---- Stay duration vs cancellation -------------------------------------------------------
with tab_map["② Stay Duration → Cancellation"]:
    st.subheader("Does length of stay affect the cancellation rate?")
    c1, c2 = st.columns([1, 1.4])

    with c1:
        cancel_by_hotel = (fdf.groupby("hotel")["is_canceled"].mean() * 100).reset_index()
        cancel_by_hotel.columns = ["hotel", "cancellation_rate"]
        fig = px.bar(cancel_by_hotel, x="hotel", y="cancellation_rate", color="hotel",
                     color_discrete_map=COLOR_MAP, text_auto=".1f",
                     title="Cancellation Rate by Hotel Type")
        fig.update_traces(texttemplate="%{y:.1f}%")
        fig.update_layout(template="plotly_white", title_x=0.5, showlegend=False,
                           yaxis_title="Cancellation Rate (%)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        max_stay = st.slider("Max nights to include", 5, 30, 14, key="stay_slider")
        stay_range = fdf[fdf["total_stay"].between(1, max_stay)]
        stay_cancel = (stay_range.groupby(["total_stay", "hotel"])["is_canceled"]
                       .mean().mul(100).reset_index(name="cancellation_rate"))
        fig = px.line(stay_cancel, x="total_stay", y="cancellation_rate", color="hotel",
                      color_discrete_map=COLOR_MAP, markers=True,
                      title=f"Cancellation Rate by Length of Stay (1-{max_stay} nights)")
        fig.update_layout(template="plotly_white", title_x=0.5, hovermode="x unified",
                           xaxis_title="Total Stay (nights)", yaxis_title="Cancellation Rate (%)",
                           legend_title="Hotel Type")
        st.plotly_chart(fig, use_container_width=True)

    group_cancel = (fdf.groupby(["stay_duration_group", "hotel"])["is_canceled"]
                     .mean().mul(100).reset_index(name="cancellation_rate"))
    order = ["<= 1 Week", "2 Weeks", "3 Weeks", ">= 4 Weeks"]
    group_cancel["stay_duration_group"] = pd.Categorical(group_cancel["stay_duration_group"],
                                                           categories=order, ordered=True)
    group_cancel = group_cancel.sort_values("stay_duration_group")
    fig = px.bar(group_cancel, x="stay_duration_group", y="cancellation_rate", color="hotel",
                 color_discrete_map=COLOR_MAP, barmode="group", text_auto=".1f",
                 title="Cancellation Rate by Stay-Duration Group")
    fig.update_traces(texttemplate="%{y:.1f}%")
    fig.update_layout(template="plotly_white", title_x=0.5,
                       xaxis_title="", yaxis_title="Cancellation Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.info("**Insight:** Cancellation risk generally rises with stay length, and the effect is "
            "steeper for City Hotel than for Resort Hotel — longer stays leave more time for a "
            "guest's plans to change before check-in.")

# ---- Lead time vs cancellation ------------------------------------------------------------
with tab_map["③ Lead Time → Cancellation"]:
    st.subheader("Does lead time affect the cancellation rate?")
    st.caption("Lead time = number of days between booking and arrival date.")

    lead_cancel = (fdf.groupby(["lead_bin", "hotel"], observed=True)["is_canceled"]
                   .mean().mul(100).reset_index(name="cancellation_rate"))
    fig = px.bar(lead_cancel, x="lead_bin", y="cancellation_rate", color="hotel",
                 color_discrete_map=COLOR_MAP, barmode="group", text_auto=".1f",
                 title="Cancellation Rate by Lead Time Bucket")
    fig.update_traces(texttemplate="%{y:.1f}%")
    fig.update_layout(template="plotly_white", title_x=0.5,
                       xaxis_title="Lead Time (days before arrival)",
                       yaxis_title="Cancellation Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.density_heatmap(fdf, x="lead_bin", y="hotel", z="is_canceled", histfunc="avg",
                              color_continuous_scale="OrRd",
                              title="Cancellation Rate Heatmap: Hotel Type x Lead Time")
    fig.update_layout(template="plotly_white", title_x=0.5, xaxis_title="Lead Time Bucket",
                       yaxis_title="", coloraxis_colorbar_title="Cancel Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.info("**Insight:** Cancellation rate climbs steadily with lead time for both hotel types. "
            "Bookings made in the final week before arrival cancel far less often than bookings "
            "made 365+ days out — and City Hotel rises furthest at long lead times, making it the "
            "highest-risk combination in the dataset.")

# ---- Revenue --------------------------------------------------------------------------------
with tab_map["💰 Revenue"]:
    st.subheader("Revenue impact")
    if not has_revenue:
        st.info("Revenue needs both `adr` and a stay-length column, which weren't both found in this dataset.")
    else:
        r1, r2, r3 = st.columns(3)
        r1.metric("Estimated total revenue", f"{fdf['estimated_revenue'].sum():,.0f}")
        lost = fdf.loc[fdf["is_canceled"] == 1, "estimated_revenue"].sum() if "is_canceled" in fdf.columns else 0
        r2.metric("Revenue lost to cancellations", f"{lost:,.0f}")
        kept = fdf.loc[fdf["is_canceled"] == 0, "estimated_revenue"].sum() if "is_canceled" in fdf.columns else fdf["estimated_revenue"].sum()
        r3.metric("Revenue from honoured bookings", f"{kept:,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            rev_hotel = fdf.groupby("hotel")["estimated_revenue"].sum().reset_index()
            fig = px.bar(rev_hotel, x="hotel", y="estimated_revenue", color="hotel",
                         color_discrete_map=COLOR_MAP, text_auto=".2s", title="Revenue by Hotel Type")
            fig.update_layout(template="plotly_white", title_x=0.5, showlegend=False,
                               xaxis_title="", yaxis_title="Estimated Revenue")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if has_period:
                rev_month = (fdf.groupby(["period", "period_label"], observed=True)["estimated_revenue"]
                             .sum().reset_index().sort_values("period"))
                fig = px.bar(rev_month, x="period_label", y="estimated_revenue", title="Revenue by Month")
            else:
                rev_month = fdf.groupby("arrival_date_month", observed=True)["estimated_revenue"].sum().reset_index()
                fig = px.bar(rev_month, x="arrival_date_month", y="estimated_revenue", title="Revenue by Month")
            fig.update_layout(template="plotly_white", title_x=0.5, xaxis_title="", yaxis_title="Estimated Revenue")
            st.plotly_chart(fig, use_container_width=True)

        if "is_canceled" in fdf.columns:
            loss_by_hotel = fdf[fdf["is_canceled"] == 1].groupby("hotel")["estimated_revenue"].sum().reset_index()
            fig = px.bar(loss_by_hotel, x="hotel", y="estimated_revenue", color="hotel",
                         color_discrete_map=COLOR_MAP, text_auto=".2s",
                         title="Revenue Lost to Cancellations, by Hotel Type")
            fig.update_layout(template="plotly_white", title_x=0.5, showlegend=False,
                               xaxis_title="", yaxis_title="Lost Revenue")
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Estimated revenue = ADR × total stay length. Treat as a directional proxy, "
                   "not confirmed transaction revenue.")

# ---- Country map (optional) ------------------------------------------------------------------
if has_country:
    with tab_map["🌍 Country Map"]:
        st.subheader("Where do guests book from?")
        country_bookings = fdf["country"].value_counts().reset_index()
        country_bookings.columns = ["country", "bookings"]
        country_bookings = country_bookings[~country_bookings["country"].isin(["Unknown"])]
        try:
            fig = px.choropleth(country_bookings, locations="country", color="bookings",
                                 color_continuous_scale="Blues", locationmode="ISO-3",
                                 title="Bookings by Country of Origin")
            fig.update_layout(template="plotly_white", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Could not render the map — the `country` column may not use ISO-3 codes.")
        st.dataframe(country_bookings.head(20), use_container_width=True)

# ---- Customer segments (optional) --------------------------------------------------------------
if has_segment:
    with tab_map["👥 Customer Segments"]:
        st.subheader("Which customer segment generates the most bookings?")
        c1, c2 = st.columns(2)
        with c1:
            seg = fdf["guest_type"].value_counts().reset_index()
            seg.columns = ["guest_type", "bookings"]
            fig = px.pie(seg, names="guest_type", values="bookings", hole=0.4,
                         title="Bookings by Guest Type")
            fig.update_layout(template="plotly_white", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True, key="guest_type_pie_chart")
        with c2:
            if "is_canceled" in fdf.columns:
                seg_cancel = (fdf.groupby("guest_type")["is_canceled"].mean().mul(100)
                              .reset_index(name="cancellation_rate"))
                fig = px.bar(seg_cancel, x="guest_type", y="cancellation_rate", text_auto=".1f",
                             title="Cancellation Rate by Guest Type")
                fig.update_traces(texttemplate="%{y:.1f}%")
                fig.update_layout(template="plotly_white", title_x=0.5,
                                   xaxis_title="", yaxis_title="Cancellation Rate (%)")
                st.plotly_chart(fig, use_container_width=True)
        st.caption("Guest type is derived: Family = children present, Couple = 2 adults / no children, "
                   "Solo / Other = everything else.")

# ---- Market intelligence (optional) ----------------------------------------------------------
if has_market:
    with tab_map["📊 Market Intelligence"]:
        st.subheader("Which channels and segments drive the business?")
        dims = [c for c in ["market_segment", "distribution_channel", "customer_type", "deposit_type"]
                if c in fdf.columns]
        dim = st.selectbox("Break down by", dims, key="market_dim")

        agg_dict = {"bookings": ("hotel", "count")}
        if "is_canceled" in fdf.columns:
            agg_dict["cancel_rate"] = ("is_canceled", "mean")
        if has_revenue:
            agg_dict["revenue"] = ("estimated_revenue", "sum")
        market = fdf.groupby(dim).agg(**agg_dict).reset_index().sort_values("bookings", ascending=False)
        if "cancel_rate" in market.columns:
            market["cancel_rate"] = (market["cancel_rate"] * 100).round(1)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(market, x=dim, y="bookings", text_auto=",.0f",
                         title=f"Bookings by {dim.replace('_', ' ').title()}")
            fig.update_layout(template="plotly_white", title_x=0.5, xaxis_title="", yaxis_title="Bookings")
            st.plotly_chart(fig, use_container_width=True, key="market_bookings_bar")
        with c2:
            if "cancel_rate" in market.columns:
                fig = px.bar(market, x=dim, y="cancel_rate", text_auto=".1f",
                             color="cancel_rate", color_continuous_scale="OrRd",
                             title=f"Cancellation Rate by {dim.replace('_', ' ').title()} (%)")
                fig.update_layout(template="plotly_white", title_x=0.5, xaxis_title="",
                                   yaxis_title="Cancellation Rate (%)")
                st.plotly_chart(fig, use_container_width=True, key="market_cancel_bar")

        if has_revenue:
            fig = px.bar(market.sort_values("revenue", ascending=False), x=dim, y="revenue",
                         text_auto=".2s", title=f"Revenue by {dim.replace('_', ' ').title()}")
            fig.update_layout(template="plotly_white", title_x=0.5, xaxis_title="", yaxis_title="Revenue")
            st.plotly_chart(fig, use_container_width=True, key="market_revenue_bar")

        st.dataframe(market.round(1), use_container_width=True)

        if not market.empty:
            biggest = market.iloc[0][dim]
            riskiest_row = market.sort_values("cancel_rate", ascending=False).iloc[0] if "cancel_rate" in market.columns else None
            msg = f"**Insight:** **{biggest}** brings in the most bookings in the current selection."
            if riskiest_row is not None:
                msg += (f" **{riskiest_row[dim]}** has the highest cancellation rate at "
                        f"**{riskiest_row['cancel_rate']:.1f}%** — a candidate for tighter deposit terms.")
            st.info(msg)

# ---- Correlation analysis ---------------------------------------------------------------------
with tab_map["🔗 Correlation Analysis"]:
    st.subheader("How do the key numeric drivers relate to each other?")
    corr_candidates = [c for c in [
        "lead_time", "adr", "total_stay", "is_canceled", "previous_cancellations",
        "booking_changes", "total_of_special_requests", "adults", "children",
        "days_in_waiting_list", "required_car_parking_spaces",
    ] if c in fdf.columns]
    sel_corr = st.multiselect("Variables to include", corr_candidates,
                               default=corr_candidates[:8], key="corr_vars")

    if len(sel_corr) >= 2:
        corr = fdf[sel_corr].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Correlation Heatmap")
        fig.update_layout(template="plotly_white", title_x=0.5)
        st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")

        if "is_canceled" in sel_corr:
            with_cancel = corr["is_canceled"].drop("is_canceled").sort_values(key=abs, ascending=False)
            if not with_cancel.empty:
                top_driver = with_cancel.index[0]
                st.info(f"**Insight:** among the selected variables, **{top_driver}** has the strongest "
                        f"linear relationship with cancellation (correlation = **{with_cancel.iloc[0]:.2f}**). "
                        "Correlation doesn't imply causation, but it's a useful screen for what to "
                        "investigate further — see the ML tab for a model-based view.")
        st.caption("Pearson correlation on the current filtered selection. Categorical fields "
                   "(hotel, market segment, etc.) aren't included here since correlation needs "
                   "numeric data — see Market Intelligence for those breakdowns.")
    else:
        st.info("Pick at least two variables above to render the heatmap.")

# ---- Forecast (optional) --------------------------------------------------------------------
if has_period:
    with tab_map["🔮 Forecast"]:
        st.subheader("Simple trend projection")
        st.caption("A lightweight linear-trend projection on monthly totals — useful for spotting "
                   "direction, not a substitute for a full time-series model.")
        horizon = st.slider("Months to project forward", 1, 12, 3, key="forecast_horizon")

        monthly = (fdf.groupby(["period", "period_label"], observed=True)
                   .agg(bookings=("hotel", "count"),
                        cancel_rate=("is_canceled", "mean") if "is_canceled" in fdf.columns else ("hotel", "count"))
                   .reset_index().sort_values("period"))

        if len(monthly) >= 3:
            x = monthly["period"].values.astype(float)
            y = monthly["bookings"].values.astype(float)
            coeffs = np.polyfit(x, y, 1)
            future_x = np.arange(x.max() + 1, x.max() + 1 + horizon)
            future_y = np.polyval(coeffs, future_x)
            future_y = np.clip(future_y, 0, None)

            hist_plot = monthly[["period_label", "bookings"]].copy()
            hist_plot["type"] = "Actual"
            future_labels = [f"+{i+1}mo" for i in range(horizon)]
            fut_plot = pd.DataFrame({"period_label": future_labels, "bookings": future_y, "type": "Forecast"})
            combo = pd.concat([hist_plot, fut_plot], ignore_index=True)

            fig = px.line(combo, x="period_label", y="bookings", color="type", markers=True,
                         title="Booking Volume: Actual vs. Projected")
            fig.update_layout(template="plotly_white", title_x=0.5, xaxis_title="", yaxis_title="Bookings")
            st.plotly_chart(fig, use_container_width=True)

            trend_word = "rising" if coeffs[0] > 0 else "falling" if coeffs[0] < 0 else "flat"
            st.info(f"**Insight:** the linear trend on the current selection is **{trend_word}** "
                    f"(~{coeffs[0]:+.0f} bookings/month). Projected bookings over the next "
                    f"{horizon} month(s): **{future_y.sum():,.0f}**.")
        else:
            st.info("Not enough distinct months in the current filter to fit a trend — widen the "
                    "year/month filters in the sidebar.")

# ---- What-if simulator -------------------------------------------------------------------------
with tab_map["🧪 What-If Simulator"]:
    st.subheader("What if we reduced cancellations?")
    st.caption("Model a policy change (deposits, reminders, stricter terms) as a flat percentage "
               "reduction in cancellations, and see the projected effect.")

    deposit_effect = st.slider("Expected cancellation reduction (%)", 0, 50, 20, key="whatif_slider")

    if "is_canceled" in fdf.columns:
        current_cancel = int(fdf["is_canceled"].sum())
        new_cancel = current_cancel * (1 - deposit_effect / 100)
        saved_bookings = current_cancel - new_cancel

        w1, w2, w3 = st.columns(3)
        w1.metric("Current cancellations", f"{current_cancel:,}")
        w2.metric("Projected cancellations", f"{new_cancel:,.0f}")
        w3.metric("Projected saved bookings", f"{saved_bookings:,.0f}")

        if has_revenue:
            avg_rev_per_cancel = fdf.loc[fdf["is_canceled"] == 1, "estimated_revenue"].mean()
            avg_rev_per_cancel = 0 if pd.isna(avg_rev_per_cancel) else avg_rev_per_cancel
            recovered_revenue = saved_bookings * avg_rev_per_cancel
            st.metric("Projected recovered revenue", f"{recovered_revenue:,.0f}")

        st.info(f"**Insight:** a **{deposit_effect}%** reduction in cancellations would save roughly "
                f"**{saved_bookings:,.0f}** bookings in the current selection. This is a simple linear "
                f"policy model — real-world uptake and guest response would vary.")
    else:
        st.info("Needs an `is_canceled` column, which wasn't found in this dataset.")

# ---- Hotel comparison ---------------------------------------------------------------------------
with tab_map["⚖ Hotel Comparison"]:
    st.subheader("City Hotel vs. Resort Hotel")
    metrics = []
    labels = []
    if "adr" in fdf.columns:
        metrics.append("adr"); labels.append("Avg. ADR")
    if "is_canceled" in fdf.columns:
        metrics.append("is_canceled"); labels.append("Cancellation Rate (%)")
    if "total_stay" in fdf.columns:
        metrics.append("total_stay"); labels.append("Avg. Stay (nights)")
    if "lead_time" in fdf.columns:
        metrics.append("lead_time"); labels.append("Avg. Lead Time (days)")

    if len(hotels) >= 2 and metrics:
        agg = fdf.groupby("hotel")[metrics].mean()
        # normalize 0-100 per metric so a radar chart is readable across different units
        norm = (agg - agg.min()) / (agg.max() - agg.min()).replace(0, 1) * 100

        fig = go.Figure()
        for hotel in agg.index:
            fig.add_trace(go.Scatterpolar(
                r=norm.loc[hotel].tolist() + [norm.loc[hotel].tolist()[0]],
                theta=labels + [labels[0]],
                fill="toself", name=hotel,
                line_color=COLOR_MAP.get(hotel),
            ))
        fig.update_layout(template="plotly_white", title="Normalized Hotel Comparison (Radar)",
                           title_x=0.5, polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(agg.rename(columns=dict(zip(metrics, labels))).round(1), use_container_width=True)
        st.caption("Radar values are min-max normalized per metric (0-100) so different units are "
                   "comparable on one chart — see the table above for actual values.")
    else:
        st.info("Comparison needs at least two hotel types selected in the sidebar filters.")

# ---- ML cancellation prediction (optional) -----------------------------------------------------
if has_ml:
    with tab_map["🤖 ML Cancellation Prediction"]:
        st.subheader("Predict the cancellation risk of a booking")
        st.caption("Random Forest classifier trained on the full cleaned dataset (not just the "
                   "current filter), so predictions stay stable while you explore other tabs.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Hold-out accuracy", f"{ml_bundle['metrics']['accuracy']*100:.1f}%")
        m2.metric("Hold-out ROC-AUC", f"{ml_bundle['metrics']['roc_auc']:.3f}")
        m3.metric("Trained on", f"{ml_bundle['metrics']['n_train']:,} bookings")

        st.markdown("#### Try a prediction")
        with st.form("cancel_predict_form"):
            inputs = {}
            f1, f2 = st.columns(2)
            num_cols = ml_bundle["num_cols"]
            cat_cols = ml_bundle["cat_cols"]
            for i, col in enumerate(num_cols):
                target = f1 if i % 2 == 0 else f2
                col_data = df[col].dropna()
                default = float(col_data.median()) if len(col_data) else 0.0
                lo = float(col_data.min()) if len(col_data) else 0.0
                hi = float(max(col_data.max(), default + 1)) if len(col_data) else 100.0
                inputs[col] = target.number_input(
                    col.replace("_", " ").title(), min_value=float(lo), max_value=float(hi),
                    value=default, key=f"ml_num_{col}"
                )
            for i, col in enumerate(cat_cols):
                target = f1 if i % 2 == 0 else f2
                options = sorted(df[col].dropna().unique().tolist())
                inputs[col] = target.selectbox(col.replace("_", " ").title(), options, key=f"ml_cat_{col}")
            submitted = st.form_submit_button("Predict cancellation risk")

        if submitted:
            row = pd.DataFrame([inputs])[num_cols + cat_cols]
            proba = ml_bundle["pipeline"].predict_proba(row)[0, 1]
            pct = proba * 100
            if pct >= 60:
                st.error(f"🔴 **Cancellation probability: {pct:.1f}% — HIGH RISK**")
            elif pct >= 30:
                st.warning(f"🟡 **Cancellation probability: {pct:.1f}% — MEDIUM RISK**")
            else:
                st.success(f"🟢 **Cancellation probability: {pct:.1f}% — LOW RISK**")

        st.markdown("#### What drives the model's predictions?")
        fig = px.bar(ml_bundle["importance"].sort_values("importance"), x="importance", y="feature",
                     orientation="h", title="Top Feature Importances")
        fig.update_layout(template="plotly_white", title_x=0.5, yaxis_title="", xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True, key="ml_feature_importance")
        st.caption("This model is a decision-support tool trained on historical patterns, not a "
                   "guarantee — always pair predictions with business judgment.")

# ---- Risk analysis (optional) ------------------------------------------------------------------
if has_ml:
    with tab_map["🔥 Risk Analysis"]:
        st.subheader("Cancellation risk & revenue-at-risk, current selection")
        num_cols = ml_bundle["num_cols"]
        cat_cols = ml_bundle["cat_cols"]
        score_df = fdf[num_cols + cat_cols].copy()
        valid_mask = score_df.notna().all(axis=1)
        scored = fdf.loc[valid_mask].copy()
        if scored.empty:
            st.info("Not enough complete rows in the current filter to score risk.")
        else:
            proba = ml_bundle["pipeline"].predict_proba(score_df.loc[valid_mask])[:, 1]
            scored["risk_score"] = proba * 100
            scored["risk_band"] = pd.cut(
                scored["risk_score"], bins=[-1, 30, 60, 101],
                labels=["🟢 Low Risk (<30%)", "🟡 Medium Risk (30-60%)", "🔴 High Risk (>60%)"]
            )

            band_counts = scored["risk_band"].value_counts().reindex(
                ["🟢 Low Risk (<30%)", "🟡 Medium Risk (30-60%)", "🔴 High Risk (>60%)"]
            ).fillna(0).astype(int)

            r1, r2, r3 = st.columns(3)
            r1.metric("Low-risk bookings", f"{band_counts.iloc[0]:,}")
            r2.metric("Medium-risk bookings", f"{band_counts.iloc[1]:,}")
            r3.metric("High-risk bookings", f"{band_counts.iloc[2]:,}")

            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(names=band_counts.index, values=band_counts.values, hole=0.4,
                             color=band_counts.index,
                             color_discrete_map={"🟢 Low Risk (<30%)": "#2ca02c",
                                                  "🟡 Medium Risk (30-60%)": "#f5b400",
                                                  "🔴 High Risk (>60%)": "#d62728"},
                             title="Bookings by Risk Band")
                fig.update_layout(template="plotly_white", title_x=0.5)
                st.plotly_chart(fig, use_container_width=True, key="risk_band_pie")
            with c2:
                if has_revenue:
                    rev_at_risk = scored.groupby("risk_band", observed=True)["estimated_revenue"].sum()
                    rev_at_risk = rev_at_risk.reindex(band_counts.index).fillna(0)
                    fig = px.bar(x=rev_at_risk.index, y=rev_at_risk.values,
                                 color=rev_at_risk.index,
                                 color_discrete_map={"🟢 Low Risk (<30%)": "#2ca02c",
                                                      "🟡 Medium Risk (30-60%)": "#f5b400",
                                                      "🔴 High Risk (>60%)": "#d62728"},
                                 text_auto=".2s", title="Revenue at Risk by Band")
                    fig.update_layout(template="plotly_white", title_x=0.5, showlegend=False,
                                       xaxis_title="", yaxis_title="Revenue")
                    st.plotly_chart(fig, use_container_width=True, key="revenue_at_risk_bar")

            if has_revenue:
                high_risk_rev = scored.loc[scored["risk_band"] == "🔴 High Risk (>60%)", "estimated_revenue"].sum()
                st.info(f"**Insight:** roughly **{high_risk_rev:,.0f}** in estimated revenue sits in "
                        "high-risk bookings (>60% predicted cancellation probability) in the current "
                        "selection. Targeting these first with a deposit or confirmation-reminder "
                        "policy would protect the most revenue for the least operational effort.")

            st.markdown("#### Highest-risk bookings")
            show_cols = [c for c in ["hotel", "arrival_date_month", "arrival_date_year", "lead_time",
                                      "adr", "total_stay", "risk_score"] if c in scored.columns]
            st.dataframe(
                scored.sort_values("risk_score", ascending=False)[show_cols].head(20).round(1),
                use_container_width=True
            )

# ---- AI business assistant ---------------------------------------------------------------------
with tab_map["🤖 AI Business Assistant"]:
    st.subheader("Ask a question about the current filtered data")
    st.caption("A lightweight rule-based assistant — it answers directly from the currently "
               "filtered dataframe, no external API calls involved.")

    sample_qs = [
        "Which hotel has the highest cancellation risk?",
        "Which month generates the most revenue?",
        "How can we reduce cancellations?",
        "Which customer segment should we target?",
    ]
    st.caption("Try: " + " · ".join(f"*{q}*" for q in sample_qs))
    question = st.text_input("Your question", key="ai_assistant_q", placeholder=sample_qs[0])

    def answer_question(q: str) -> str:
        ql = q.lower()

        if "cancel" in ql and ("risk" in ql or "highest" in ql or "hotel" in ql) and "is_canceled" in fdf.columns:
            by_hotel = fdf.groupby("hotel")["is_canceled"].mean().mul(100).sort_values(ascending=False)
            if by_hotel.empty:
                return "No hotel data available in the current filter."
            top = by_hotel.index[0]
            extra = ""
            if "lead_bin" in fdf.columns:
                by_lead = fdf.groupby("lead_bin", observed=True)["is_canceled"].mean().mul(100)
                if not by_lead.empty:
                    extra = (f" Risk climbs further for bookings made **{by_lead.idxmax()}** "
                              f"before arrival (**{by_lead.max():.1f}%** cancelled there).")
            return (f"**{top}** has the highest cancellation rate at **{by_hotel.iloc[0]:.1f}%** "
                    f"in the current selection.{extra}")

        if "revenue" in ql and "month" in ql and has_revenue and "arrival_date_month" in fdf.columns:
            by_month = fdf.groupby("arrival_date_month", observed=True)["estimated_revenue"].sum().sort_values(ascending=False)
            if by_month.empty:
                return "No revenue data available in the current filter."
            return (f"**{by_month.index[0]}** generates the most estimated revenue, at "
                    f"**{by_month.iloc[0]:,.0f}** in the current selection.")

        if "reduce" in ql and "cancel" in ql:
            tips = ["requiring a deposit or partial payment on long-lead-time bookings",
                    "sending active confirmation reminders 30-60 days before arrival",
                    "tightening cancellation terms for stays over 2 weeks"]
            if "lead_bin" in fdf.columns and "is_canceled" in fdf.columns:
                by_lead = fdf.groupby("lead_bin", observed=True)["is_canceled"].mean()
                if not by_lead.empty:
                    tips.insert(0, f"targeting the **{by_lead.idxmax()}** lead-time bucket first, "
                                   "since it has the highest cancellation rate in this data")
            return "To reduce cancellations, consider " + "; ".join(tips) + \
                   ". See the What-If Simulator tab to size the revenue impact of any of these."

        if "segment" in ql and "target" in ql and has_segment and "is_canceled" in fdf.columns:
            seg_vol = fdf["guest_type"].value_counts()
            seg_cancel = fdf.groupby("guest_type")["is_canceled"].mean().mul(100)
            best = (seg_vol / seg_vol.sum() * 100 - seg_cancel).sort_values(ascending=False)
            if best.empty:
                return "No customer-segment data available in the current filter."
            return (f"**{best.index[0]}** looks like the best segment to target — it combines a solid "
                    f"share of bookings (**{seg_vol[best.index[0]]:,}**) with a comparatively low "
                    f"cancellation rate (**{seg_cancel[best.index[0]]:.1f}%**).")

        if ("hotel" in ql and "most" in ql) or "booked most" in ql:
            share = fdf["hotel"].value_counts()
            if share.empty:
                return "No hotel data available."
            return f"**{share.index[0]}** is booked most often, with **{share.iloc[0]:,}** bookings in the current selection."

        if "adr" in ql or "average daily rate" in ql:
            if "adr" in fdf.columns:
                return f"The average ADR in the current selection is **{fdf['adr'].mean():.0f}**."

        if "lead time" in ql:
            if "lead_time" in fdf.columns:
                return f"Average lead time in the current selection is **{fdf['lead_time'].mean():.0f} days**."

        return ("I can answer questions about cancellation risk by hotel, revenue by month, how to "
                "reduce cancellations, which segment to target, booking volume, ADR, and lead time. "
                "Try rephrasing, or use one of the sample questions above.")

    if st.button("Ask", key="ai_assistant_btn") or question:
        if question:
            st.success(answer_question(question))

# ---- Data quality ---------------------------------------------------------------------------------
with tab_map["🧹 Data Quality"]:
    st.subheader("Cleaning summary (full dataset, before sidebar filters)")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows before cleaning", f"{quality['rows_before']:,}")
    q2.metric("Rows after cleaning", f"{quality['rows_after']:,}")
    q3.metric("Duplicate rows removed", f"{quality['duplicates']:,}")
    q4.metric("Columns", f"{quality['cols']}")

    q5, q6 = st.columns(2)
    q5.metric("Negative-ADR rows dropped", f"{quality['negative_adr']:,}")
    q6.metric("Zero-guest rows dropped", f"{quality['zero_guest_rows']:,}")

    st.markdown("#### Missing values (original data)")
    if len(quality["missing"]) > 0:
        miss_df = quality["missing"].reset_index()
        miss_df.columns = ["column", "missing_count"]
        miss_df["missing_pct"] = (miss_df["missing_count"] / quality["rows_before"] * 100).round(2)
        fig = px.bar(miss_df, x="column", y="missing_pct", text_auto=".1f",
                     title="Missing Values by Column (%)")
        fig.update_layout(template="plotly_white", title_x=0.5, xaxis_title="", yaxis_title="% Missing")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(miss_df, use_container_width=True)
    else:
        st.success("No missing values found in the original data.")

# ---- Summary --------------------------------------------------------------------------------------
with tab_map["📋 Summary & Recommendations"]:
    st.subheader("📋 Key Findings")
    st.markdown("""
- **Hotel type & seasonality:** City Hotel is booked roughly twice as often as Resort Hotel, and
  both properties see demand build through spring, peak in September-October, and bottom out
  January-March.
- **Stay duration:** Cancellation risk rises with the length of stay, most sharply for City Hotel.
- **Lead time:** The single biggest driver of cancellation risk is booking far in advance — both
  hotels see cancellation rates climb steadily as lead time grows, and City Hotel's long-lead-time
  bookings are the riskiest segment overall.
- **Revenue:** cancellations translate directly into lost estimated revenue, concentrated in the
  same high-risk segments identified above.
    """)
    st.subheader("✅ Recommendations")
    st.markdown("""
1. **Grow Resort Hotel share** with off-peak packages and cross-promotions during City Hotel's
   busiest months, smoothing demand across the portfolio.
2. **Tighten cancellation terms for long stays** (e.g. non-refundable or partial-deposit rates for
   stays over 2 weeks) to protect revenue from the highest-risk segment.
3. **Require a deposit or send active confirmation reminders for bookings made 90+ days out** —
   this is where cancellation risk is highest and the most preventable, since guests still have
   time to firm up (or cancel) their plans well before arrival.
4. **Highest-impact single action:** tackle long-lead-time City Hotel bookings first — it's the
   combination with the highest cancellation rate in the data, so a deposit/reminder policy
   targeted there would recover the most lost revenue for the least operational change. Use the
   **What-If Simulator** tab to size the expected impact of that policy.
    """)
    st.caption("Figures above reflect the current sidebar filter selection, not the full dataset.")

    with st.expander("Preview filtered data"):
        st.dataframe(fdf.head(200), use_container_width=True)

# ---- Export reports ---------------------------------------------------------------------------
with tab_map["📥 Export Reports"]:
    st.subheader("Download a multi-sheet Excel report")
    st.caption("Builds an Excel workbook from the current filtered selection, with one sheet per "
               "analysis area.")

    @st.cache_data(show_spinner=False)
    def build_excel_report(_fdf: pd.DataFrame) -> bytes:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            kpi_rows = [("Total bookings", len(_fdf))]
            if "is_canceled" in _fdf.columns:
                kpi_rows.append(("Cancellation rate (%)", round(_fdf["is_canceled"].mean() * 100, 2)))
            if "lead_time" in _fdf.columns:
                kpi_rows.append(("Avg. lead time (days)", round(_fdf["lead_time"].mean(), 1)))
            if "adr" in _fdf.columns:
                kpi_rows.append(("Avg. ADR", round(_fdf["adr"].mean(), 1)))
            if "estimated_revenue" in _fdf.columns:
                kpi_rows.append(("Est. total revenue", round(_fdf["estimated_revenue"].sum(), 0)))
            pd.DataFrame(kpi_rows, columns=["KPI", "Value"]).to_excel(
                writer, sheet_name="Executive Summary", index=False)

            if "hotel" in _fdf.columns:
                _fdf.groupby("hotel").size().reset_index(name="bookings").to_excel(
                    writer, sheet_name="Hotel Summary", index=False)

            if "arrival_date_month" in _fdf.columns:
                monthly = _fdf.groupby("arrival_date_month", observed=True).size().reset_index(name="bookings")
                monthly.to_excel(writer, sheet_name="Monthly Analysis", index=False)

            if "is_canceled" in _fdf.columns and "lead_bin" in _fdf.columns:
                canc = (_fdf.groupby("lead_bin", observed=True)["is_canceled"].mean().mul(100)
                        .reset_index(name="cancellation_rate_pct"))
                canc.to_excel(writer, sheet_name="Cancellation Analysis", index=False)

            if "estimated_revenue" in _fdf.columns and "hotel" in _fdf.columns:
                rev = _fdf.groupby("hotel")["estimated_revenue"].sum().reset_index()
                rev.to_excel(writer, sheet_name="Revenue Analysis", index=False)

            if "guest_type" in _fdf.columns:
                _fdf["guest_type"].value_counts().reset_index().rename(
                    columns={"index": "guest_type", "guest_type": "bookings"}
                ).to_excel(writer, sheet_name="Customer Segments", index=False)

            if "market_segment" in _fdf.columns:
                _fdf.groupby("market_segment").size().reset_index(name="bookings").to_excel(
                    writer, sheet_name="Market Segments", index=False)

            _fdf.head(5000).to_excel(writer, sheet_name="Filtered Data (5k rows)", index=False)
        return buffer.getvalue()

    excel_bytes = build_excel_report(fdf)
    st.download_button(
        "⬇ Download Excel Report",
        data=excel_bytes,
        file_name="hotel_business_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    st.subheader("Download the filtered data as CSV")
    st.download_button(
        "⬇ Download CSV",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="hotel_bookings_filtered.csv",
        mime="text/csv",
        key="export_tab_csv",
    )