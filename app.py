# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Alarm Management Dashboard", layout="wide")

st.title("Alarm Management Dashboard")
st.subheader("Nuisance Alarm and Bad Actor Identification")

DATA_FILE = "alarm_occurrences_clean.csv"

# Accent palette (navy primary; amber = nuisance highlight)
NAVY = "#1b2a3a"
AMBER = "#d9a441"
SLATE = "#7d9bb0"
PALETTE = ["#1b2a3a", "#4f6d7a", "#7d9bb0", "#5b7c6f", "#a8703e",
           "#8c5a5a", "#b08968", "#6a7b53", "#9c6f4a", "#c9a227"]

# Nuisance / non-nuisance two-color scheme (used on every stacked chart)
NUIS_COLORS = {
    "Fleeting nuisance": AMBER,
    "Standing nuisance": "#b5772e",
    "Actionable": NAVY,
    "Borderline": SLATE,
}
CLASS_ORDER = ["Fleeting nuisance", "Standing nuisance", "Borderline", "Actionable"]
# simple two-way split used for the stacked breakdown charts
BINARY_COLORS = {"Nuisance": AMBER, "Not nuisance": NAVY}
BINARY_ORDER = ["Nuisance", "Not nuisance"]

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
# STEP 1b - Global nuisance classification
# ---------------------------------------------------------------
# Nuisance is defined ONCE across the whole dataset, not within the filtered
# window, so a tag's classification does not change when the user zooms into a
# date range. The label rides along on every row. This function is cached on
# the threshold values, so it only recomputes when a slider actually moves.
@st.cache_data
def classify_nuisance(data, fleeting_secs, fleeting_freq, standing_freq):
    d = data.copy()

    # SMS-eligible: active longer than 5 minutes OR never cleared (still active)
    sms = d["sms_notification"].astype(bool)

    # Per-tag firing counts (global)
    tag_total = d.groupby("alarm_tag")["occurrence_id"].transform("count")
    # Per-tag SMS firing counts (global)
    sms_counts = d.loc[sms].groupby("alarm_tag")["occurrence_id"].count()
    tag_sms = d["alarm_tag"].map(sms_counts).fillna(0)

    is_short = d["duration_sec"] < fleeting_secs
    is_chronic = tag_total > fleeting_freq
    is_sms_chronic = tag_sms > standing_freq

    fleeting = is_short & is_chronic & ~sms
    standing = sms & is_sms_chronic
    borderline = is_short & ~is_chronic & ~sms  # short but not repetitive

    cls = pd.Series("Actionable", index=d.index)
    cls[borderline] = "Borderline"
    cls[fleeting] = "Fleeting nuisance"
    cls[standing] = "Standing nuisance"
    d["nuisance_class"] = cls
    d["is_nuisance"] = d["nuisance_class"].isin(["Fleeting nuisance", "Standing nuisance"])
    d["nuisance_binary"] = d["is_nuisance"].map({True: "Nuisance", False: "Not nuisance"})
    return d


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
# STEP 2b - Adjustable nuisance parameters (collapsible)
# ---------------------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Adjust Nuisance Parameters", expanded=False):
    st.caption(
        "Nuisance alarms are defined globally across the full dataset. "
        "Adjust the thresholds to test how sensitive the results are; the "
        "default values are the definition used in our analysis."
    )
    fleeting_secs = st.slider(
        "Fleeting: clears in under (seconds)", 5, 120, 60, step=5,
        help="A fleeting alarm resolves faster than this.")
    fleeting_freq = st.slider(
        "Fleeting: tag fires more than (times)", 10, 100, 50, step=5,
        help="…and the tag fires more than this many times across 6 months.")
    standing_freq = st.slider(
        "Standing: tag sends more than (SMS)", 10, 100, 50, step=5,
        help="A standing nuisance tag sends more than this many SMS alarms.")

    st.markdown(
        f"**Current definition**  \n"
        f"• *Fleeting nuisance* — clears under **{fleeting_secs}s** "
        f"and tag fires **>{fleeting_freq}×**  \n"
        f"• *Standing nuisance* — SMS-eligible and tag sends **>{standing_freq}** texts"
    )

# Apply the classification globally with the chosen thresholds
df = classify_nuisance(df, fleeting_secs, fleeting_freq, standing_freq)


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
nuisance_pct = 100 * dff["is_nuisance"].mean()
unacked_pct = 100 * (1 - dff["is_acknowledged"].mean())
median_dur = dff["duration_sec"].median()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total alarms", f"{total:,}")
c2.metric("Nuisance alarms", f"{nuisance_pct:.0f}%",
          help="Share classified as fleeting or standing nuisance, "
               "using the current parameter settings.")
c3.metric("Alarms per day", f"{total / n_days:,.0f}")
c4.metric("Unique alarm tags", f"{unique_tags:,}")
c5.metric("SMS notifications", f"{sms_count:,}", f"{100 * sms_count / total:.1f}% of total")
c6.metric("Median duration", f"{median_dur:,.0f} sec" if pd.notna(median_dur) else "n/a")

st.markdown("---")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def stacked_bar(data, col, title, order=None, horizontal=False, top_n=None):
    """Bar chart where each bar is split into nuisance vs. not-nuisance."""
    agg = (data.groupby([col, "nuisance_binary"])
               .size().reset_index(name="alarms"))

    # category ordering for the axis
    if order:
        cats = order
    else:
        totals = data[col].value_counts()
        if top_n:
            totals = totals.head(top_n)
        cats = totals.index.tolist()
        agg = agg[agg[col].isin(cats)]
        if horizontal:
            cats = cats[::-1]  # largest at top for horizontal bars

    cat_orders = {col: cats, "nuisance_binary": BINARY_ORDER}

    if horizontal:
        fig = px.bar(agg, x="alarms", y=col, color="nuisance_binary",
                     orientation="h", title=title, barmode="stack",
                     color_discrete_map=BINARY_COLORS, category_orders=cat_orders)
        fig.update_layout(xaxis_title="Alarms", yaxis_title="")
    else:
        fig = px.bar(agg, x=col, y="alarms", color="nuisance_binary",
                     title=title, barmode="stack",
                     color_discrete_map=BINARY_COLORS, category_orders=cat_orders)
        fig.update_layout(xaxis_title="", yaxis_title="Alarms")

    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      legend_title_text="", legend=dict(orientation="h",
                      yanchor="bottom", y=1.02, xanchor="right", x=1),
                      margin=dict(l=20, r=20, t=60, b=40))
    return fig


def count_bar(data, col, title, order=None, horizontal=False, top_n=None):
    """Plain single-color count bar (used where a nuisance split doesn't apply)."""
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
                     color_discrete_sequence=[NAVY])
    else:
        fig = px.bar(agg, x=col, y="alarms", title=title,
                     color_discrete_sequence=[NAVY])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=20, r=20, t=50, b=40), showlegend=False)
    return fig


tab1, tab2, tab3, tab4 = st.tabs(
    ["Alarm Profile", "Reduction Over Time", "Bad Actors", "Data"])


# ---------------------------------------------------------------
# TAB 1 - Alarm Profile
# ---------------------------------------------------------------
with tab1:
    # --- Nuisance composition (answers RQ1) ---
    st.markdown("##### Alarm composition")
    comp = (dff["nuisance_class"]
            .value_counts()
            .reindex(CLASS_ORDER, fill_value=0)
            .reset_index())
    comp.columns = ["nuisance_class", "alarms"]
    comp["y"] = "All alarms"
    fig_comp = px.bar(comp, x="alarms", y="y", color="nuisance_class",
                      orientation="h", barmode="stack",
                      color_discrete_map=NUIS_COLORS,
                      category_orders={"nuisance_class": CLASS_ORDER},
                      title="How much of the alarm load is nuisance?")
    fig_comp.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_title="Alarms", yaxis_title="",
                           legend_title_text="", height=230,
                           legend=dict(orientation="h", yanchor="bottom",
                                       y=1.02, xanchor="right", x=1),
                           margin=dict(l=20, r=20, t=60, b=30))
    st.plotly_chart(fig_comp, width="stretch")
    st.caption(
        f"**{nuisance_pct:.0f}%** of alarms in view are nuisance "
        "(fleeting or standing). *Fleeting* = short and repetitive (screen clutter); "
        "*standing* = long-lived, repetitive SMS senders (notification load); "
        "*borderline* = short but not repetitive."
    )

    st.markdown("---")
    st.markdown("##### Breakdowns (each bar split by nuisance share)")

    a, b = st.columns(2)
    with a:
        st.plotly_chart(stacked_bar(dff, "pad", "Alarms by Well Pad", horizontal=True),
                        width="stretch")
    with b:
        st.plotly_chart(stacked_bar(dff, "equipment_type", "Alarms by Equipment Type",
                                    horizontal=True), width="stretch")

    c, d = st.columns(2)
    with c:
        st.plotly_chart(stacked_bar(dff, "priority_label", "Alarms by Priority",
                                    order=PRIORITY_ORDER), width="stretch")
    with d:
        st.plotly_chart(stacked_bar(dff, "duration_bucket", "Alarms by Duration",
                                    order=DURATION_ORDER), width="stretch")

    e, f = st.columns(2)
    with e:
        st.plotly_chart(stacked_bar(dff, "ack_type", "Alarms by Acknowledgment Type",
                                    order=ACK_ORDER), width="stretch")
    with f:
        st.plotly_chart(stacked_bar(dff, "label_clean", "Top 15 Alarm Types",
                                    horizontal=True, top_n=15), width="stretch")


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

    # Trend split by nuisance vs not, so the reduction story shows composition
    trend = (tmp.groupby(["period", "nuisance_binary"])
                .size().reset_index(name="alarms"))
    fig = px.area(trend, x="period", y="alarms", color="nuisance_binary",
                  title=f"Alarm Volume Over Time ({grain})",
                  color_discrete_map=BINARY_COLORS,
                  category_orders={"nuisance_binary": BINARY_ORDER})
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_title="Period", yaxis_title="Alarm Count",
                      legend_title_text="", margin=dict(l=20, r=20, t=50, b=40))
    st.plotly_chart(fig, width="stretch")

    totals_by_period = tmp.groupby("period").size()
    if len(totals_by_period) >= 2:
        first, last = totals_by_period.iloc[0], totals_by_period.iloc[-1]
        peak = totals_by_period.max()
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
        "of the total load. Use the scope toggle in the sidebar to switch between all "
        "alarms (screen-clutter view) and SMS-only (notification-load view). Tags are "
        "flagged as nuisance using the current parameter settings."
    )

    top_n = st.slider("Number of alarm tags to show", 5, 40, 15, step=5)

    ranked = (dff.groupby("alarm_tag")
                .agg(alarms=("occurrence_id", "count"),
                     sms=("sms_notification", "sum"),
                     nuisance=("is_nuisance", "max"),
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
    ranked["flag"] = ranked["nuisance"].map({True: "Nuisance", False: "Not nuisance"})

    head = ranked.head(top_n).sort_values("alarms")
    fig = px.bar(head, x="alarms", y="alarm_tag", orientation="h",
                 title=f"Top {top_n} Alarm Tags by Occurrence Count",
                 color="flag", color_discrete_map=BINARY_COLORS,
                 category_orders={"flag": BINARY_ORDER})
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      yaxis_title="", xaxis_title="Occurrences",
                      legend_title_text="", height=max(400, 26 * top_n),
                      margin=dict(l=20, r=20, t=50, b=40))
    st.plotly_chart(fig, width="stretch")

    share = ranked.head(top_n)["alarms"].sum()
    nuis_in_top = int(ranked.head(top_n)["nuisance"].sum())
    st.info(
        f"These {top_n} alarm tags account for **{share:,} of {total:,} alarms "
        f"({100 * share / total:.1f}%)** out of {unique_tags:,} distinct tags in scope, "
        f"and **{nuis_in_top} of the {top_n}** are flagged nuisance. This ranked list is "
        "the starting point for deciding which tags to suppress or retune first."
    )

    st.markdown("##### Bad actor detail")
    st.dataframe(
        ranked.head(top_n).rename(columns={
            "alarm_tag": "Alarm Tag", "alarms": "Occurrences",
            "pct_of_total": "% of Total", "sms": "SMS Sent", "flag": "Flag",
            "fleeting_pct": "% Under 10 Sec", "unacked_pct": "% Unacked",
            "median_sec": "Median Sec", "total_hours": "Total Hours Active",
        })[["Alarm Tag", "Occurrences", "% of Total", "SMS Sent", "Flag",
            "% Under 10 Sec", "% Unacked", "Median Sec", "Total Hours Active"]]
        .reset_index(drop=True),
        width="stretch", hide_index=True,
    )

    st.markdown("##### Highest total time in alarm")
    by_time = ranked.sort_values("total_hours", ascending=False).head(top_n)
    fig3 = px.bar(by_time.sort_values("total_hours"), x="total_hours", y="alarm_tag",
                  orientation="h", color="flag", color_discrete_map=BINARY_COLORS,
                  category_orders={"flag": BINARY_ORDER},
                  title="Standing Alarm Burden (Total Hours Active)")
    fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       yaxis_title="", xaxis_title="Hours",
                       legend_title_text="", height=max(400, 26 * top_n),
                       margin=dict(l=20, r=20, t=50, b=40))
    st.plotly_chart(fig3, width="stretch")


# ---------------------------------------------------------------
# TAB 4 - Data
# ---------------------------------------------------------------
with tab4:
    display_cols = ["active_time", "pad", "equipment_type", "equipment_name",
                    "measurement", "well_name", "label_clean", "priority_label",
                    "event_value", "setpoint_a", "duration_sec", "duration_bucket",
                    "ack_type", "time_to_ack_sec", "sms_notification",
                    "nuisance_class", "status"]
    table = dff.loc[:, [c for c in display_cols if c in dff.columns]]
    st.dataframe(table.reset_index(drop=True), width="stretch",
                 hide_index=True)
    st.download_button(
        "Download filtered data as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="alarm_profile_filtered.csv",
        mime="text/csv",
    )
