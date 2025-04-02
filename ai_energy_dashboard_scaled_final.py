
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Energy Dashboard", layout="wide")
st.title("🔋 AI Inference Energy Usage Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("combined_ai_energy_with_estimates.csv", parse_dates=["created_at"])
    df["hour"] = df["created_at"].dt.floor("H")
    df["weekday"] = df["created_at"].dt.day_name()

    # Apply corrected energy scaling factor to measured energy
    scale_factor = 50493  # determined empirically
    df["measured_energy_joules"] = pd.to_numeric(df["energy_consumption_llm_total"], errors="coerce") * scale_factor
    df["estimated_energy_joules"] = df["total_duration"] / 1e9
    df["final_energy_joules"] = df["measured_energy_joules"].fillna(df["estimated_energy_joules"])

    df["energy_kwh"] = df["final_energy_joules"] / 3.6e6
    df["co2_kg"] = df["energy_kwh"] * 0.475
    df["lightbulb_hours"] = df["final_energy_joules"] / (50 * 3600)
    return df

df = load_data()

# Sidebar filters
model_options = df["model_name"].unique()
device_options = df["device"].dropna().unique()
energy_source_options = {
    "Final Energy (measured → estimated fallback)": "final_energy_joules",
    "Measured Only (energy_consumption_llm_total)": "measured_energy_joules",
    "Estimated from Duration (total_duration)": "estimated_energy_joules"
}
unit_options = {
    "Joules (J)": "raw",
    "Kilowatt-hours (kWh)": "kwh",
    "CO₂ Emissions (kg)": "co2",
    "50W Lightbulb Hours": "bulb"
}

selected_models = st.sidebar.multiselect("Select Models", model_options, default=list(model_options))
selected_devices = st.sidebar.multiselect("Select Devices", device_options, default=list(device_options))
selected_energy_source_label = st.sidebar.radio("Select Energy Source", list(energy_source_options.keys()))
selected_energy_column = energy_source_options[selected_energy_source_label]
selected_unit_label = st.sidebar.radio("Display Unit", list(unit_options.keys()))
selected_unit_type = unit_options[selected_unit_label]

# Filter data
df_filtered = df[(df["model_name"].isin(selected_models)) & (df["device"].isin(selected_devices))]
df_filtered["base_energy"] = df_filtered[selected_energy_column]

# Apply unit conversion
if selected_unit_type == "kwh":
    df_filtered["display_energy"] = df_filtered["base_energy"] / 3.6e6
elif selected_unit_type == "co2":
    df_filtered["display_energy"] = (df_filtered["base_energy"] / 3.6e6) * 0.475
elif selected_unit_type == "bulb":
    df_filtered["display_energy"] = df_filtered["base_energy"] / (50 * 3600)
else:
    df_filtered["display_energy"] = df_filtered["base_energy"]

unit_suffix = selected_unit_label.split(" ")[-1].strip("()")

# KPIs
st.subheader("Key Metrics")
total_val = df_filtered["display_energy"].sum()
st.metric(f"Total Energy ({unit_suffix})", f"{total_val:,.2f}")
st.info(f"💡 Currently showing: {selected_energy_source_label} in {selected_unit_label}")

# Line chart by model
st.subheader(f"📊 Energy Usage Over Time by Model ({unit_suffix})")
hourly_model = df_filtered.groupby(["hour", "model_name"])["display_energy"].sum().reset_index()
fig_model_time = px.line(hourly_model, x="hour", y="display_energy", color="model_name",
                         title=f"Hourly {selected_unit_label} by Model",
                         labels={"display_energy": selected_unit_label, "hour": "Time"})
st.plotly_chart(fig_model_time, use_container_width=True)

# Cross: Device vs Model
st.subheader(f"🧮 Energy Usage by Device and Model ({unit_suffix})")
cross_data = df_filtered.groupby(["model_name", "device"])["display_energy"].sum().reset_index()
fig_cross = px.bar(cross_data, x="model_name", y="display_energy", color="device", barmode="group",
                   labels={"display_energy": selected_unit_label}, title="Energy by Model and Device")
st.plotly_chart(fig_cross, use_container_width=True)

# Forecast
st.subheader(f"🔮 7-Day Forecast by Model ({unit_suffix})")
avg_hourly = df_filtered.groupby("model_name")["display_energy"].sum() / df_filtered.groupby("model_name")["hour"].nunique()
forecast_df = pd.DataFrame({
    "model_name": avg_hourly.index,
    "forecast_value": avg_hourly.values * 24 * 7
}).sort_values("forecast_value", ascending=False)

fig_forecast = px.bar(forecast_df, x="model_name", y="forecast_value", color="model_name",
                      labels={"forecast_value": f"7-Day Total {selected_unit_label}"})
st.plotly_chart(fig_forecast, use_container_width=True)

# Heatmap
st.subheader("🗓️ Energy Usage Heatmap (Weekday vs Hour)")
df_filtered["hour_of_day"] = df_filtered["created_at"].dt.hour
pivot = df_filtered.pivot_table(values="display_energy", index="hour_of_day", columns="weekday", aggfunc="sum").fillna(0)
pivot = pivot[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
fig_heatmap = px.imshow(pivot, aspect="auto", title=f"{selected_unit_label} by Hour and Weekday",
                        labels=dict(x="Weekday", y="Hour", color=selected_unit_label))
st.plotly_chart(fig_heatmap, use_container_width=True)

# Token vs Energy
st.subheader(f"🔍 Token Length vs {selected_unit_label}")
fig_tokens = px.scatter(df_filtered, x="response_token_length", y="display_energy", color="model_name",
                        title="Response Tokens vs Energy",
                        labels={"response_token_length": "Tokens", "display_energy": selected_unit_label})
st.plotly_chart(fig_tokens, use_container_width=True)

# Complexity
st.subheader("📘 Complexity vs Energy")
color_by = st.radio("Color By", ["device", "model_name"], index=0, horizontal=True)
complexity_cols = ["prompt_token_length", "response_token_length", "avg_sentence_length", "lexical_diversity"]
available_cols = [col for col in complexity_cols if col in df_filtered.columns]
if available_cols:
    selected_complexity = st.selectbox("Choose Complexity Feature", available_cols)
    fig_complexity = px.scatter(df_filtered, x=selected_complexity, y="display_energy", color=df_filtered[color_by],
                                title=f"{selected_complexity} vs {selected_unit_label}",
                                labels={selected_complexity: selected_complexity, "display_energy": selected_unit_label})
    st.plotly_chart(fig_complexity, use_container_width=True)
else:
    st.info("No complexity features available in dataset.")
