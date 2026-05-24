import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os

# BASE_DIR points to the folder this script lives in (works on cloud too)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================
# STEP 1 — DATA LOADING & FEATURE ENGINEERING
# =============================================================
@st.cache_data
def load_data():
    file_path = os.path.join(BASE_DIR, "Solar_Panel_Performance_Dataset_840_Rows.xlsx")
    df = pd.read_excel(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Energy_Output(kWh)", "Solar_Irradiance(w/m2)"])
    df["Month"]     = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Day"]       = df["Date"].dt.day
    df["Timestamp"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")
    df["Performance_Ratio"] = (
        df["Energy_Output(kWh)"]
        / df["Solar_Irradiance(w/m2)"].replace(0, np.nan)
    ).fillna(0)
    def hour_bin(h):
        if h < 6:    return "Night (0-5)"
        elif h < 12: return "Morning (6-11)"
        elif h < 18: return "Afternoon (12-17)"
        else:        return "Evening (18-23)"
    df["Hour_Period"] = df["Hour"].apply(hour_bin)
    return df

# =============================================================
# STEP 2 — PAGE CONFIG  (must be the very first Streamlit call)
# =============================================================
st.set_page_config(
    page_title="Solar Panel Performance Dashboard",
    page_icon="☀️", layout="wide"
)

# Load data AFTER set_page_config
df = load_data()

# =============================================================
# CSS STYLING
# =============================================================
st.markdown("""
<style>
    .kpi-box {
        background: #1a2535; border-radius: 12px;
        padding: 18px 22px; text-align: center;
        border-left: 4px solid #f5a623; margin-bottom: 8px;
    }
    .kpi-label { color: #8fa8c8; font-size: 13px; font-weight: 600; margin-bottom: 4px; }
    .kpi-value { color: #ffffff; font-size: 26px; font-weight: 700; }
    .kpi-sub   { color: #f5a623; font-size: 12px; margin-top: 4px; }
    .action-alert {
        background: #2d1515; border-radius: 10px;
        padding: 14px 18px; border-left: 4px solid #ef4444;
        margin-bottom: 8px;
    }
    .action-ok {
        background: #0f2d1a; border-radius: 10px;
        padding: 14px 18px; border-left: 4px solid #22c55e;
        margin-bottom: 8px;
    }
    .action-warn {
        background: #2d2010; border-radius: 10px;
        padding: 14px 18px; border-left: 4px solid #f59e0b;
        margin-bottom: 8px;
    }
    .action-title { color: #ffffff; font-size: 14px; font-weight: 700; }
    .action-body  { color: #aac0d0; font-size: 12px; margin-top: 4px; }
    section[data-testid="stSidebar"] { background-color: #0f1c2b; }
</style>
""", unsafe_allow_html=True)

# =============================================================
# SIDEBAR FILTERS
# =============================================================
st.sidebar.image("https://img.icons8.com/fluency/96/sun.png", width=56)
st.sidebar.title("Solar Panel Filters")
st.sidebar.markdown("---")

all_panels = sorted(df["Panel_ID"].unique(), key=lambda x: int(x[1:]))
panels = st.sidebar.multiselect("Solar Panel", options=all_panels, default=all_panels)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
months = st.sidebar.multiselect(
    "Month", options=sorted(df["Month"].unique()),
    default=sorted(df["Month"].unique()),
    format_func=lambda m: MONTH_NAMES[m-1]
)

PERIOD_ORDER = ["Night (0-5)", "Morning (6-11)", "Afternoon (12-17)", "Evening (18-23)"]
periods = st.sidebar.multiselect("Hour Period", options=PERIOD_ORDER, default=PERIOD_ORDER)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Action Layer Settings")
alert_threshold = st.sidebar.slider(
    "Alert threshold (% below predicted)",
    min_value=1, max_value=20, value=5, step=1,
    help="Panels producing this % below their predicted output will be flagged"
)
st.sidebar.caption("Dataset: Kaggle — Solar Panel Performance Data (820 rows)")
st.sidebar.caption("Data Analysis in Smart Systems")

filtered = df[
    df["Panel_ID"].isin(panels) &
    df["Month"].isin(months) &
    df["Hour_Period"].isin(periods)
]

st.markdown("## ☀️ Solar Panel Performance Dashboard")
st.caption(
    f"**{len(filtered):,}** records  ·  "
    f"Panels: {', '.join(panels)}  ·  "
    f"Months: {', '.join(MONTH_NAMES[m-1] for m in months)}"
)
st.markdown("---")

if filtered.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar.")
    st.stop()

# =============================================================
# STEP 3 — KPI METRIC CARDS
# =============================================================
total_energy   = filtered["Energy_Output(kWh)"].sum()
avg_irradiance = filtered["Solar_Irradiance(w/m2)"].mean()
best_panel     = filtered.groupby("Panel_ID")["Energy_Output(kWh)"].sum().idxmax()
peak_energy    = filtered["Energy_Output(kWh)"].max()
avg_perf_ratio = filtered["Performance_Ratio"].mean()

kpi_data = [
    ("Total Energy Output",   f"{total_energy:,.2f} kWh",    "all panels combined"),
    ("Avg Solar Irradiance",  f"{avg_irradiance:,.1f} W/m²", "mean sunlight intensity"),
    ("Best Performing Panel", best_panel,                     "highest total energy output"),
    ("Peak Energy Reading",   f"{peak_energy:.3f} kWh",      "single highest hourly record"),
    ("Avg Performance Ratio", f"{avg_perf_ratio:.5f}",       "energy per W/m² of irradiance"),
]

for col, (label, value, subtitle) in zip(st.columns(5), kpi_data):
    col.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================
# STEP 4 — TIME SERIES & PANEL BAR CHARTS
# =============================================================
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 Daily Energy Output Over Time")
    ts = (
        filtered.groupby("Date")["Energy_Output(kWh)"]
        .sum().reset_index()
        .rename(columns={"Energy_Output(kWh)": "Total Energy (kWh)"})
    )
    fig1 = px.line(ts, x="Date", y="Total Energy (kWh)", markers=True,
                   color_discrete_sequence=["#f5a623"])
    fig1.update_layout(template="plotly_dark", height=300,
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("🔆 Total Energy by Panel")
    panel_df = (
        filtered.groupby("Panel_ID")["Energy_Output(kWh)"]
        .sum().reset_index()
        .rename(columns={"Energy_Output(kWh)": "Total Energy (kWh)"})
        .sort_values("Total Energy (kWh)", ascending=True)
    )
    fig2 = px.bar(panel_df, x="Total Energy (kWh)", y="Panel_ID",
                  orientation="h", color="Total Energy (kWh)",
                  color_continuous_scale=["#3a5f8a", "#f5a623"])
    fig2.update_layout(template="plotly_dark", height=300,
                       showlegend=False, coloraxis_showscale=False,
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# =============================================================
# STEP 5 — IRRADIANCE BY HOUR PERIOD & HEATMAP
# =============================================================
c3, c4 = st.columns([1, 2])

with c3:
    st.subheader("🌤️ Avg Irradiance by Hour Period")
    period_df = (
        filtered.groupby("Hour_Period")["Solar_Irradiance(w/m2)"]
        .mean().reset_index()
        .rename(columns={"Solar_Irradiance(w/m2)": "Avg Irradiance (W/m²)"})
    )
    period_df["Hour_Period"] = pd.Categorical(
        period_df["Hour_Period"], categories=PERIOD_ORDER, ordered=True
    )
    period_df = period_df.sort_values("Hour_Period")
    fig3 = px.bar(period_df, x="Hour_Period", y="Avg Irradiance (W/m²)",
                  color="Avg Irradiance (W/m²)",
                  color_continuous_scale=["#1a2535", "#f5a623"])
    fig3.update_layout(template="plotly_dark", height=300,
                       showlegend=False, coloraxis_showscale=False,
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.subheader("🕐 Hourly Energy Heatmap  (Hour × Day of Week)")
    heat = (
        filtered.groupby(["DayOfWeek", "Hour"])["Energy_Output(kWh)"]
        .mean().reset_index()
    )
    pivot = heat.pivot(index="DayOfWeek", columns="Hour",
                       values="Energy_Output(kWh)")
    day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    pivot.index = [day_map.get(i, i) for i in pivot.index]
    fig4 = px.imshow(pivot, color_continuous_scale="YlOrRd",
                     labels=dict(x="Hour of Day", y="Day of Week",
                                 color="Avg kWh"), aspect="auto")
    fig4.update_layout(template="plotly_dark", height=300,
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# =============================================================
# STEP 6 — SCATTER PLOTS
# =============================================================
sample = filtered.sample(min(600, len(filtered)), random_state=42)
c5, c6 = st.columns(2)

with c5:
    st.subheader("⚡ Irradiance vs Energy Output")
    corr_val = filtered["Solar_Irradiance(w/m2)"].corr(filtered["Energy_Output(kWh)"])
    st.caption(f"Pearson r = **{corr_val:.4f}** — "
               f"{'very strong' if corr_val > 0.9 else 'strong'} positive relationship")
    fig5 = px.scatter(sample, x="Solar_Irradiance(w/m2)", y="Energy_Output(kWh)",
                      color="Hour_Period", opacity=0.65,
                      color_discrete_sequence=px.colors.qualitative.Bold,
                      labels={"Solar_Irradiance(w/m2)": "Solar Irradiance (W/m²)",
                              "Energy_Output(kWh)": "Energy Output (kWh)",
                              "Hour_Period": "Time of Day"})
    fig5.update_layout(template="plotly_dark", height=340,
                       margin=dict(l=0, r=0, t=10, b=0),
                       legend=dict(orientation="h", yanchor="bottom",
                                   y=1.01, xanchor="left", x=0))
    st.plotly_chart(fig5, use_container_width=True)

with c6:
    st.subheader("🕛 Hour of Day vs Energy Output")
    st.caption("Daily solar bell curve — coloured by panel")
    fig6 = px.scatter(sample, x="Hour", y="Energy_Output(kWh)",
                      color="Panel_ID", opacity=0.65,
                      color_discrete_sequence=px.colors.qualitative.Vivid,
                      labels={"Hour": "Hour of Day (0-23)",
                              "Energy_Output(kWh)": "Energy Output (kWh)",
                              "Panel_ID": "Panel"})
    for hour_mark, label in [(6, "Morning"), (12, "Afternoon"), (18, "Evening")]:
        fig6.add_vline(x=hour_mark, line_dash="dash",
                       line_color="rgba(255,255,255,0.25)",
                       annotation_text=label, annotation_position="top right",
                       annotation_font_color="rgba(255,255,255,0.5)")
    fig6.update_layout(template="plotly_dark", height=340,
                       margin=dict(l=0, r=0, t=10, b=0),
                       legend=dict(orientation="h", yanchor="bottom",
                                   y=1.01, xanchor="left", x=0))
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# =============================================================
# STEP 7 — REGRESSION MODEL & LIVE PREDICTOR
# =============================================================
st.subheader("📐 Regression Model — Predicting Energy Output")
st.caption("Multiple linear regression using Solar Irradiance and Hour of Day as predictors")

reg_df  = filtered[["Solar_Irradiance(w/m2)", "Hour", "Energy_Output(kWh)"]].dropna()
X_reg   = reg_df[["Solar_Irradiance(w/m2)", "Hour"]]
y_reg   = reg_df["Energy_Output(kWh)"]
model   = LinearRegression().fit(X_reg, y_reg)
y_hat   = model.predict(X_reg)
r2_val  = r2_score(y_reg, y_hat)
rmse_val= np.sqrt(mean_squared_error(y_reg, y_hat))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Multiple R",   f"{np.sqrt(r2_val):.4f}")
m2.metric("R² Score",     f"{r2_val:.4f}")
m3.metric("RMSE",         f"{rmse_val:.4f} kWh")
m4.metric("Observations", f"{len(y_reg):,}")

coeff_df = pd.DataFrame({
    "Variable":       ["Intercept", "Solar_Irradiance(w/m2)", "Hour"],
    "Coefficient":    [model.intercept_, model.coef_[0], model.coef_[1]],
    "Interpretation": [
        "Baseline energy when irradiance and hour are both zero",
        "Energy added per 1 W/m² increase in irradiance",
        "Energy added per 1 hour later in the day",
    ]
})
st.dataframe(coeff_df.style.format({"Coefficient": "{:.6f}"}),
             use_container_width=True, hide_index=True)

r1, r2_col = st.columns(2)

with r1:
    st.markdown("**Actual vs Predicted Energy (kWh)**")
    st.caption("Dots close to the red line = accurate predictions")
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=y_reg, y=y_hat, mode="markers",
                              marker=dict(color="#f5a623", opacity=0.4, size=5),
                              name="Predictions"))
    fig7.add_trace(go.Scatter(x=[y_reg.min(), y_reg.max()],
                              y=[y_reg.min(), y_reg.max()],
                              mode="lines",
                              line=dict(color="red", dash="dash", width=2),
                              name="Perfect Fit"))
    fig7.update_layout(template="plotly_dark", height=350,
                       xaxis_title="Actual Energy (kWh)",
                       yaxis_title="Predicted Energy (kWh)",
                       margin=dict(l=0, r=0, t=10, b=0),
                       legend=dict(orientation="h", yanchor="bottom",
                                   y=1.01, xanchor="left", x=0))
    st.plotly_chart(fig7, use_container_width=True)

with r2_col:
    st.markdown("**🔮 Live Energy Predictor**")
    st.caption("Adjust the sliders — prediction updates instantly")
    irr_input = st.slider("Solar Irradiance (W/m²)", 0.0, 2149.5, 750.0, 10.0)
    hr_input  = st.slider("Hour of Day", 0, 23, 12, 1, format="%d:00")
    prediction = max(model.predict([[irr_input, hr_input]])[0], 0)
    st.success(f"**Predicted Energy Output: {prediction:.4f} kWh**")
    st.info(
        f"**Formula:**  \n"
        f"Energy = ({model.coef_[0]:.6f} × {irr_input}) "
        f"+ ({model.coef_[1]:.6f} × {hr_input}) "
        f"+ ({model.intercept_:.6f})  \n"
        f"= **{prediction:.4f} kWh**"
    )
    if irr_input < 50:      context = "🌑 Near-zero irradiance — nighttime or heavy cloud cover."
    elif irr_input < 400:   context = "🌥️ Low irradiance — early morning, late evening or overcast."
    elif irr_input < 900:   context = "🌤️ Moderate irradiance — partly cloudy daytime conditions."
    else:                   context = "☀️ High irradiance — clear sky peak solar conditions."
    st.caption(context)

st.markdown("---")

# =============================================================
# STEP 8 — ACTION LAYER
# =============================================================
st.subheader("🚨 Action Layer — Automated Panel Alerts & Recommendations")
st.caption(
    "DIKWA Layer 5 — the system evaluates every panel against its regression-predicted "
    "baseline and generates automatic action signals. Threshold is adjustable in the sidebar."
)

full_X = df[["Solar_Irradiance(w/m2)", "Hour"]]
full_y = df["Energy_Output(kWh)"]
action_model = LinearRegression().fit(full_X, full_y)
df["Predicted_kWh"] = action_model.predict(full_X).clip(min=0)

panel_action = (
    df[df["Panel_ID"].isin(panels)]
    .groupby("Panel_ID")
    .agg(
        Actual_kWh   =("Energy_Output(kWh)", "sum"),
        Predicted_kWh=("Predicted_kWh",      "sum"),
        Records      =("Energy_Output(kWh)", "count")
    )
    .reset_index()
)

panel_action["Gap_kWh"] = panel_action["Actual_kWh"] - panel_action["Predicted_kWh"]
panel_action["Gap_Pct"] = (
    panel_action["Gap_kWh"] / panel_action["Predicted_kWh"] * 100
).round(2)

panel_action["_sort"] = panel_action["Panel_ID"].str[1:].astype(int)
panel_action = panel_action.sort_values("_sort").drop("_sort", axis=1)

def classify_panel(gap_pct, threshold):
    if gap_pct < -threshold:
        return "🔴 ALERT",    "red",    f"Producing {abs(gap_pct):.1f}% BELOW predicted. Schedule immediate inspection."
    elif gap_pct < 0:
        return "🟡 MONITOR",  "yellow", f"Producing {abs(gap_pct):.1f}% below predicted. Watch for further decline."
    elif gap_pct < 5:
        return "🟢 NORMAL",   "green",  f"Performing within expected range (+{gap_pct:.1f}%)."
    else:
        return "⭐ EXCELLENT", "green",  f"Outperforming prediction by {gap_pct:.1f}%. Benchmark against other panels."

panel_action["Status"], panel_action["Color"], panel_action["Action"] = zip(
    *panel_action["Gap_Pct"].apply(lambda g: classify_panel(g, alert_threshold))
)

n_alert = (panel_action["Color"] == "red").sum()
n_warn  = (panel_action["Color"] == "yellow").sum()
n_ok    = (panel_action["Color"] == "green").sum()

a1, a2, a3, a4 = st.columns(4)
a1.metric("🔴 Panels Alerting",   n_alert, delta=f">{alert_threshold}% below predicted", delta_color="inverse")
a2.metric("🟡 Panels to Monitor", n_warn,  delta="0% to threshold below",                delta_color="off")
a3.metric("🟢 Normal / Excellent",n_ok,    delta="At or above predicted",                delta_color="normal")
a4.metric("Alert Threshold Set",  f"{alert_threshold}%", delta="Adjustable in sidebar",  delta_color="off")

st.markdown("&nbsp;")
st.markdown("**Per-Panel Action Signals**")
cols_per_row = 5
rows = [panel_action.iloc[i:i+cols_per_row] for i in range(0, len(panel_action), cols_per_row)]

for row_df in rows:
    cols = st.columns(len(row_df))
    for col, (_, row) in zip(cols, row_df.iterrows()):
        css = ("action-alert" if row["Color"] == "red"
               else "action-warn" if row["Color"] == "yellow"
               else "action-ok")
        col.markdown(f"""
        <div class="{css}">
            <div class="action-title">{row['Panel_ID']}  {row['Status']}</div>
            <div class="action-body">
                Actual:  {row['Actual_kWh']:.1f} kWh<br>
                Predicted: {row['Predicted_kWh']:.1f} kWh<br>
                Gap: {row['Gap_Pct']:+.1f}%<br><br>
                {row['Action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("&nbsp;")
st.markdown("**Full Action Summary Table**")

display_df = panel_action[[
    "Panel_ID", "Actual_kWh", "Predicted_kWh",
    "Gap_kWh", "Gap_Pct", "Records", "Status"
]].copy()
display_df.columns = [
    "Panel", "Actual (kWh)", "Predicted (kWh)",
    "Gap (kWh)", "Gap (%)", "Records", "Action Status"
]

def colour_gap(val):
    if val < -alert_threshold: return "color: #ef4444; font-weight: bold"
    elif val < 0:              return "color: #f59e0b; font-weight: bold"
    else:                      return "color: #22c55e; font-weight: bold"

st.dataframe(
    display_df.style
    .format({
        "Actual (kWh)":    "{:.2f}",
        "Predicted (kWh)": "{:.2f}",
        "Gap (kWh)":       "{:+.2f}",
        "Gap (%)":         "{:+.1f}%",
    })
    .map(colour_gap, subset=["Gap (%)"]),
    use_container_width=True,
    hide_index=True
)

st.markdown("&nbsp;")
st.markdown("**🤖 Automated System Recommendations**")

alerts = panel_action[panel_action["Color"] == "red"]
warns  = panel_action[panel_action["Color"] == "yellow"]

if len(alerts) > 0:
    panels_list = ", ".join(alerts["Panel_ID"].tolist())
    st.error(
        f"⚡ **Immediate Action Required** — Panel(s) **{panels_list}** "
        f"are producing more than {alert_threshold}% below their predicted output. "
        f"Inspect for soiling, shading, wiring faults, or hardware degradation."
    )

if len(warns) > 0:
    panels_list = ", ".join(warns["Panel_ID"].tolist())
    st.warning(
        f"👁️ **Monitor Closely** — Panel(s) **{panels_list}** are underperforming "
        f"but within the alert threshold. Review performance over the next 7 days."
    )

if len(alerts) == 0 and len(warns) == 0:
    st.success(
        "✅ **All panels are performing at or above predicted levels.** "
        "No immediate action required. Continue routine monitoring schedule."
    )

best_action = panel_action.loc[panel_action["Gap_Pct"].idxmax()]
st.info(
    f"⭐ **Best Practice Reference** — Panel **{best_action['Panel_ID']}** "
    f"is the top performer (+{best_action['Gap_Pct']:.1f}% above predicted). "
    f"Use its installation conditions as the benchmark for all future panel deployments."
)

# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.caption(
    "© 2026 Douglas Kilyobas Titus  ·  MSc Data Analytics  ·  "
    "Framework: DIKWA (Data · Information · Knowledge · Wisdom · Action)"
)
