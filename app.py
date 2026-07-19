# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Alarm Management Dashboard", layout="wide")

st.title("Alarm Management Dashboard")
st.subheader("Nuisance Alarm and Bad Actor Identification")

DATA_FILE = "alarm_occurrences_clean.csv"

PALETTE = ["#6b4423", "#a8703e", "#c9a227", "#5b7c6f", "#8c5a5a",
           "#7d8ca3", "#b08968", "#4f6d7a", "#9c6f4a", "#6a7b53"]

PRIORITY_ORDER = ["1 - Low", "2 - Medium", "3 - High", "4 - Critical"]
DURATION_ORDER = ["<1 min", "1-5 min", "5-60 min", "1-24 hr", ">24 hr", "Still Active"]
ACK_ORDER = ["Operator", "Auto-Acknowledged", "System Service", "Unacknowledged"]


# ---------------------------------------------------------------
# STEP 1 - Load Data
# ---------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["active_time", "ack_time", "clear_time"])
    df["active_date"] = pd.to_datetime(df["active_date"]).dt.date
    for col in ["sms_notification", "is_cleared", "is_acknowledged",
                "is_standing", "is_fleeting", "ack_after_clear"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower() == "true"
    return df


REQUIRED = ["pad", "equipment_type", "equipment_name", "label_clean", "alarm_tag",
            "priority_label", "active_time", "active_date", "active_month",
            "duration_sec", "duration_bucket", "ack_type", "sms_notification",
            "is_acknowledged", "is_fleeting", "status"]

try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"Dataset file '{DATA_FILE}' not found in the repository.")
    st.stop()
except Exception as e:
    st.error("An unexpected error occurred while loading the dataset.")
    st.write(e)
    st.stop()

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.write(list(df.columns))
    st.stop()


# ---------------------------------------------------------------
# STEP 2 - Sidebar Filters
# ---------------------------------------------------------------
st.sidebar.header("Filters")

min_date = df["active_date"].min()
max_date = df["active_date"].max()

date_range = st.sidebar.date_input(
    "Time frame",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Select the start and end date for the alarm profile.",
)

if isinstance(date_range, (list, tuple)):
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0]
else:
    start_date = end_date = date_range

if start_date > end_date:
    st.sidebar.error("Start date must be on or before the end date.")
    st.stop()

st.sidebar.markdown("---")

scope = st.sidebar.radio(
    "Alarm scope",
    options=["All alarms", "SMS-generating only (> 5 min)"],
    help="Only alarms active longer than five minutes trigger an SMS notification "
         "to the on-call operator. Still-active alarms are included.",
)


def build_options(series):
    return ["All"] + sorted(series.dropna().astype(str).unique().tolist())


selected_pads = st.sidebar.multiselect(
    "Well pad", options=build_options(df["pad"]), default=["All"])
selected_equip = st.sidebar.multiselect(
    "Equipment type", options=build_options(df["equipment_type"]), default=["All"])
selected_priority = st.sidebar.multiselect(
    "Priority", options=build_options(df["priority_label"]), default=["All"])
selected_ack = st.sidebar.multiselect(
    "Acknowledgment", options=build_options(df["ack_type"]), default=["All"])


# ---------------------------------------------------------------
# STEP 3 - Filtering Logic
# ---------------------------------------------------------------
dff = df[(df["active_date"] >= start_date) & (df["active_date"] <= end_date)]

if scope.startswith("SMS"):
    dff = dff[dff["sms_notification"]]

def apply_filter(data, col, selected):
    """Filter on any specific values chosen. 'All' is ignored, so the user does
    not have to remove it before a selection takes effect. If nothing but 'All'
    (or nothing at all) is chosen, no filter is applied."""
    picks = [v for v in selected if v != "All"]
    if picks:
        return data[data[col].astype(str).isin(picks)]
    return data


dff = apply_filter(dff, "pad", selected_pads)
dff = apply_filter(dff, "equipment_type", selected_equip)
dff = apply_filter(dff, "priority_label", selected_priority)
dff = apply_filter(dff, "ack_type", selected_ack)

if dff.empty:
    st.warning("No alarms match the selected filters. Widen the time frame or clear a filter.")
    st.stop()

n_days = max((end_date - start_date).days + 1, 1)

st.caption(
    f"Showing **{scope}** from **{start_date}** to **{end_date}** "
    f"({n_days} days) - {len(dff):,} alarm occurrences."
)


# ---------------------------------------------------------------
# STEP 4 - Key Metrics
# ---------------------------------------------------------------
total = len(dff)
sms_count = int(dff["sms_notification"].sum())
unique_tags = dff["alarm_tag"].nunique()
unacked_pct = 100 * (1 - dff["is_acknowledged"].mean())
fleeting_pct = 100 * dff["is_fleeting"].mean()
median_dur = dff["duration_sec"].median()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total alarms", f"{total:,}")
c2.metric("Alarms per day", f"{total / n_days:,.0f}")
c3.metric("Unique alarm tags", f"{unique_tags:,}")
c4.metric("SMS notifications", f"{sms_count:,}", f"{100 * sms_count / total:.1f}% of total")
c5.metric("Unacknowledged", f"{unacked_pct:.1f}%")
c6.metric("Median duration", f"{median_dur:,.0f} sec" if pd.notna(median_dur) else "n/a")

st.markdown("---")


# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------
def count_bar(data, col, title, order=None, horizontal=False, top_n=None):
    agg = data[col].value_counts().reset_index()
    agg.columns = [col, "alarms"]
    if order:
        agg[col] = pd.Categorical(agg[col], categories=order, ordered=True)
        agg = agg.sort_values(col)
    elif top_n:
        agg = agg.head(top_n)
    if horizontal:
        agg = agg.sort_values("alarms")
        fig = px.bar(agg, x="alarms", y=col, orientation="h", title=title,
                     color_discrete_sequence=PALETTE)
    else:
        fig = px.bar(agg, x=col, y="alarms", title=title,
                     color_discrete_sequence=PALETTE)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=20, r=20, t=50, b=40), showlegend=False)
    return fig


tab1, tab2, tab3, tab4 = st.tabs(
    ["Alarm Profile", "Reduction Over Time", "Bad Actors", "Data"])


# ---------------------------------------------------------------
# TAB 1 - Alarm Profile
# ---------------------------------------------------------------
with tab1:
    a, b = st.columns(2)
    with a:
        st.plotly_chart(count_bar(dff, "pad", "Alarms by Well Pad", horizontal=True),
                        width="stretch")
    with b:
        st.plotly_chart(count_bar(dff, "equipment_type", "Alarms by Equipment Type",
                                  horizontal=True), width="stretch")

    c, d = st.columns(2)
    with c:
        st.plotly_chart(count_bar(dff, "priority_label", "Alarms by Priority",
                                  order=PRIORITY_ORDER), width="stretch")
    with d:
        st.plotly_chart(count_bar(dff, "duration_bucket", "Alarms by Duration",
                                  order=DURATION_ORDER), width="stretch")

    e, f = st.columns(2)
    with e:
        st.plotly_chart(count_bar(dff, "ack_type", "Alarms by Acknowledgment Type",
                                  order=ACK_ORDER), width="stretch")
    with f:
        st.plotly_chart(count_bar(dff, "label_clean", "Top 15 Alarm Types",
                                  horizontal=True, top_n=15), width="stretch")

    st.markdown("##### Priority mix by equipment type")
    heat = (dff.groupby(["equipment_type", "priority_label"])
              .size().reset_index(name="alarms"))
    fig = px.density_heatmap(heat, x="priority_label", y="equipment_type", z="alarms",
                             category_orders={"priority_label": PRIORITY_ORDER},
                             color_continuous_scale="Oranges", text_auto=True)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_title="Priority", yaxis_title="Equipment Type",
                      margin=dict(l=20, r=20, t=30, b=40))
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------
# TAB 2 - Reduction Over Time
# ---------------------------------------------------------------
with tab2:
    grain = st.radio("Time grain", ["Daily", "Weekly", "Monthly"],
                     horizontal=True, index=1)
    ser = pd.to_datetime(dff["active_time"])
    if grain == "Daily":
        key = ser.dt.date
    elif grain == "Weekly":
        key = ser.dt.to_period("W").dt.start_time.dt.date
    else:
        key = ser.dt.to_period("M").dt.start_time.dt.date

    tmp = dff.assign(period=key)

    trend = tmp.groupby("period").size().reset_index(name="alarms")
    fig = px.line(trend, x="period", y="alarms", markers=True,
                  title=f"Alarm Volume Over Time ({grain})",
                  color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_title="Period", yaxis_title="Alarm Count",
                      margin=dict(l=20, r=20, t=50, b=40))
    st.plotly_chart(fig, width="stretch")

    if len(trend) >= 2:
        first, last = trend["alarms"].iloc[0], trend["alarms"].iloc[-1]
        peak = trend["alarms"].max()
        g1, g2, g3 = st.columns(3)
        g1.metric("First period", f"{first:,}")
        g2.metric("Latest period", f"{last:,}",
                  f"{100 * (last - first) / first:+.1f}% vs first" if first else None)
        g3.metric("Reduction from peak", f"{100 * (peak - last) / peak:.1f}%"
                  if peak else "n/a")

    st.markdown("##### Breakdown of the trend")
    split = st.selectbox("Split volume by",
                         ["priority_label", "equipment_type", "pad", "ack_type"],
                         format_func=lambda c: c.replace("_", " ").title())
    stacked = tmp.groupby(["period", split]).size().reset_index(name="alarms")
    fig2 = px.area(stacked, x="period", y="alarms", color=split,
                   color_discrete_sequence=PALETTE,
                   category_orders={"priority_label": PRIORITY_ORDER,
                                    "ack_type": ACK_ORDER})
    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_title="Period", yaxis_title="Alarm Count",
                       margin=dict(l=20, r=20, t=30, b=40))
    st.plotly_chart(fig2, width="stretch")

    st.markdown("##### When do alarms arrive?")
    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(count_bar(dff, "active_hour", "Alarms by Hour of Day"),
                        width="stretch")
    with h2:
        dow = ["Monday", "Tuesday", "Wednesday", "Thursday",
               "Friday", "Saturday", "Sunday"]
        st.plotly_chart(count_bar(dff, "active_dow", "Alarms by Day of Week",
                                  order=dow), width="stretch")


# ---------------------------------------------------------------
# TAB 3 - Bad Actors
# ---------------------------------------------------------------
with tab3:
    st.markdown(
        "A **bad actor** is a single alarm tag that generates a disproportionate share "
        "of the total load. A **nuisance alarm** is one that fires often but clears on "
        "its own almost immediately, or is routinely auto-acknowledged, suggesting no "
        "operator action was needed."
    )

    top_n = st.slider("Number of alarm tags to show", 5, 40, 15, step=5)

    ranked = (dff.groupby("alarm_tag")
                .agg(alarms=("occurrence_id", "count"),
                     sms=("sms_notification", "sum"),
                     fleeting_pct=("is_fleeting", "mean"),
                     unacked_pct=("is_acknowledged", lambda s: 1 - s.mean()),
                     median_sec=("duration_sec", "median"),
                     total_hours=("duration_sec", lambda s: s.sum() / 3600))
                .reset_index()
                .sort_values("alarms", ascending=False))
    ranked["fleeting_pct"] = (100 * ranked["fleeting_pct"]).round(1)
    ranked["unacked_pct"] = (100 * ranked["unacked_pct"]).round(1)
    ranked["pct_of_total"] = (100 * ranked["alarms"] / total).round(2)
    ranked["total_hours"] = ranked["total_hours"].round(1)
    ranked["median_sec"] = ranked["median_sec"].round(0)

    head = ranked.head(top_n).sort_values("alarms")
    fig = px.bar(head, x="alarms", y="alarm_tag", orientation="h",
                 title=f"Top {top_n} Alarm Tags by Occurrence Count",
                 color="fleeting_pct", color_continuous_scale="Oranges",
                 labels={"fleeting_pct": "% under 10 sec"})
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      yaxis_title="", xaxis_title="Occurrences",
                      height=max(400, 26 * top_n),
                      margin=dict(l=20, r=20, t=50, b=40))
    st.plotly_chart(fig, width="stretch")

    share = ranked.head(top_n)["alarms"].sum()
    st.info(
        f"These {top_n} alarm tags account for **{share:,} of {total:,} alarms "
        f"({100 * share / total:.1f}%)** out of {unique_tags:,} distinct tags in scope. "
        "High occurrence combined with a high share under 10 seconds is the classic "
        "nuisance signature and the strongest candidate for suppression or setpoint review."
    )

    st.markdown("##### Bad actor detail")
    st.dataframe(
        ranked.head(top_n).rename(columns={
            "alarm_tag": "Alarm Tag", "alarms": "Occurrences",
            "pct_of_total": "% of Total", "sms": "SMS Sent",
            "fleeting_pct": "% Under 10 Sec", "unacked_pct": "% Unacked",
            "median_sec": "Median Sec", "total_hours": "Total Hours Active",
        }).reset_index(drop=True),
        width="stretch", hide_index=True,
    )

    st.markdown("##### Highest total time in alarm")
    by_time = ranked.sort_values("total_hours", ascending=False).head(top_n)
    fig3 = px.bar(by_time.sort_values("total_hours"), x="total_hours", y="alarm_tag",
                  orientation="h", color_discrete_sequence=[PALETTE[3]],
                  title="Standing Alarm Burden (Total Hours Active)")
    fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       yaxis_title="", xaxis_title="Hours",
                       height=max(400, 26 * top_n),
                       margin=dict(l=20, r=20, t=50, b=40))
    st.plotly_chart(fig3, width="stretch")


# ---------------------------------------------------------------
# TAB 4 - Data
# ---------------------------------------------------------------
with tab4:
    display_cols = ["active_time", "pad", "equipment_type", "equipment_name",
                    "measurement", "well_name", "label_clean", "priority_label",
                    "event_value", "setpoint_a", "duration_sec", "duration_bucket",
                    "ack_type", "time_to_ack_sec", "sms_notification", "status"]
    table = dff.loc[:, [c for c in display_cols if c in dff.columns]]
    st.dataframe(table.reset_index(drop=True), width="stretch",
                 hide_index=True)
    st.download_button(
        "Download filtered data as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="alarm_profile_filtered.csv",
        mime="text/csv",
    )
