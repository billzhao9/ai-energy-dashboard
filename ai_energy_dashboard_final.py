
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Inference Energy Dashboard", layout="wide")
st.title("🔋 AI Inference Energy Usage Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("combined_ai_energy_with_complexity_and_device.csv", parse_dates=["created_at"])
    df["hour"] = df["created_at"].dt.floor("H")
    df["weekday"] = df["created_at"].dt.day_name()
    df["energy_joules"] = df["total_duration"] / 1e9
    df["energy_kwh"] = df["energy_joules"] / 3.6e6
    df["co2_kg"] = df["energy_kwh"] * 0.475
    df["lightbulb_hours"] = df["energy_joules"] / (50 * 3600)
    return df

df = load_data()

# Sidebar filters
model_options = df["model_name"].unique()
device_options = df["device"].dropna().unique()
unit_options = {
    "Joules (J)": "energy_joules",
    "Kilowatt-hours (kWh)": "energy_kwh",
    "CO₂ Emissions (kg)": "co2_kg",
    "50W Lightbulb Hours": "lightbulb_hours"
}
selected_models = st.sidebar.multiselect("Select Models", model_options, default=list(model_options))
selected_devices = st.sidebar.multiselect("Select Devices", device_options, default=list(device_options))
selected_unit_label = st.sidebar.radio("Choose Unit to Display", list(unit_options.keys()))
selected_unit = unit_options[selected_unit_label]

# Filter data
df_filtered = df[(df["model_name"].isin(selected_models)) & (df["device"].isin(selected_devices))]

# KPIs
st.subheader("Key Metrics")
unit_suffix = selected_unit_label.split(" ")[-1].strip("()")
total_val = df_filtered[selected_unit].sum()
st.metric(f"Total Inference ({unit_suffix})", f"{total_val:,.2f}")
st.info(f"💡 This is equivalent in {selected_unit_label} for the selected filters.")

# Line chart: Energy over time by model
st.subheader(f"📊 Energy Usage Over Time by Model ({unit_suffix})")
hourly_model = df_filtered.groupby(["hour", "model_name"])[selected_unit].sum().reset_index()
fig_model_time = px.line(hourly_model, x="hour", y=selected_unit, color="model_name",
                         title=f"Hourly {selected_unit_label} by Model",
                         labels={selected_unit: selected_unit_label, "hour": "Time"})
st.plotly_chart(fig_model_time, use_container_width=True)

# Device comparison
st.subheader(f"🖥️ Total Energy Usage by Device ({unit_suffix})")
device_energy = df_filtered.groupby("device")[selected_unit].sum().reset_index().sort_values(by=selected_unit, ascending=False)
fig_device = px.bar(device_energy, x="device", y=selected_unit, color="device",
                    title=f"Total {selected_unit_label} by Device",
                    labels={"device": "Device", selected_unit: selected_unit_label})
st.plotly_chart(fig_device, use_container_width=True)

# Cross Analysis: Device vs Model
st.subheader(f"📊 Device vs Model Energy Usage ({unit_suffix})")
cross_data = df_filtered.groupby(["model_name", "device"])[selected_unit].sum().reset_index()
fig_cross = px.bar(cross_data, x="model_name", y=selected_unit, color="device", barmode="group",
                   title=f"{selected_unit_label} by Model and Device",
                   labels={selected_unit: selected_unit_label, "model_name": "Model"})
st.plotly_chart(fig_cross, use_container_width=True)

# Forecast: 7-Day Projection
st.subheader(f"🔮 7-Day Forecast Based on Average Hourly Usage ({unit_suffix})")
avg_hourly = df_filtered.groupby("model_name")[selected_unit].sum() / df_filtered.groupby("model_name")["hour"].nunique()
forecast_hours = 24 * 7
forecast_df = pd.DataFrame({
    "model_name": avg_hourly.index,
    "forecast_value": avg_hourly.values * forecast_hours
}).sort_values("forecast_value", ascending=False)

fig_forecast = px.bar(forecast_df, x="model_name", y="forecast_value", color="model_name",
                      labels={"forecast_value": f"7-Day Total {selected_unit_label}"},
                      title=f"7-Day Projected {selected_unit_label} by Model")
st.plotly_chart(fig_forecast, use_container_width=True)

# Heatmap
st.subheader("🗓️ Energy Usage Heatmap (Weekday vs Hour)")
df_filtered["hour_of_day"] = df_filtered["created_at"].dt.hour
pivot = df_filtered.pivot_table(values=selected_unit, index="hour_of_day", columns="weekday", aggfunc="sum").fillna(0)
pivot = pivot[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
fig_heatmap = px.imshow(pivot, aspect="auto", title=f"{selected_unit_label} by Hour and Weekday",
                        labels=dict(x="Weekday", y="Hour", color=selected_unit_label))
st.plotly_chart(fig_heatmap, use_container_width=True)

# Scatter plot for tokens vs energy
st.subheader(f"🔍 Response Tokens vs {selected_unit_label}")
fig_tokens = px.scatter(df_filtered, x="response_token_length", y=selected_unit, color="model_name",
                        title=f"Token Length vs {selected_unit_label}",
                        labels={"response_token_length": "Response Tokens", selected_unit: selected_unit_label})
st.plotly_chart(fig_tokens, use_container_width=True)

# Complexity plot
st.subheader("📘 Language Complexity vs Energy")
complexity_cols = [col for col in df_filtered.columns if "diversity" in col or "length" in col]
for extra in ["prompt_token_length", "response_token_length"]:
    if extra in df_filtered.columns and extra not in complexity_cols:
        complexity_cols.append(extra)

if complexity_cols:
    selected_complexity = st.selectbox("Choose a Complexity Feature", complexity_cols)
    fig_complexity = px.scatter(df_filtered, x=selected_complexity, y=selected_unit, color="device",
                                title=f"{selected_complexity} vs {selected_unit_label}",
                                labels={selected_complexity: selected_complexity, selected_unit: selected_unit_label})
    st.plotly_chart(fig_complexity, use_container_width=True)
else:
    st.info("No complexity metrics found.")
