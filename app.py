import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# Load Model
model = joblib.load("Final_model_lgbm_tuned.pkl")
freq_mapping = joblib.load("freq_mapping.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Page icon
st.set_page_config(
    "Metro Traffic Volume Prediction",
    page_icon="🚦",
    layout="wide"
)


# Side bar inputs
st.sidebar.header("Traffic Volume Prediction")

weather_description = st.sidebar.selectbox(
    "Weather Description",
    options=sorted(freq_mapping["weather_description"].keys())
)

temp = st.sidebar.slider(
    "Temperature",
    min_value=242.691,
    max_value=310.07,
    value=281.255,
    step=0.01
)

clouds_all = st.sidebar.slider(
    "Cloud Coverage (%)",
    min_value=0,
    max_value=100,
    value=49,
    step=1
)

hour = st.sidebar.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=12,
    step=1
)

day = st.sidebar.slider(
    "Day",
    min_value=1,
    max_value=31,
    value=15,
    step=1
)

month = st.sidebar.selectbox(
    "Month",
    options=["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
)

year = st.sidebar.slider(
    "year",
    min_value=2012,
    max_value=2018,
    value=2015,
    step=1
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

is_weekend = st.sidebar.selectbox(
    "Is Weekend",
    options=["Yes", "No"]
)

holiday = st.sidebar.selectbox(
    "Holiday",
    options=["Columbus Day", "Independence Day", "Labor Day", "Martin Luther King Jr Day", "Memorial Day",
             "New Years Day", "No Holiday", "State Fair", "Thanksgiving Day", "Veterans Day", "Washingtons Birthday"]
)

weather = st.sidebar.selectbox(
    "Weather",
    options=["Main Clouds", "Main Drizzle", "Main Fog", "Main Haze", "Main Mist", "Main Rain", "Main Smoke",
             "Main Snow", "Main Squall", "Main Thunderstorm"]
)


# Default Value
prediction = None

# Prediction Button
predict_button = st.sidebar.button("Predict Traffic Volume")

if predict_button:

    # Weather Description Frequency Encoding
    weather_description_encoded = freq_mapping["weather_description"].get(
        weather_description,
        0
    )

    # Month Mapping
    month_mapping = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }

    month_encoded = month_mapping[month]

    # Day of Week Mapping
    day_of_week_mapping = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    day_of_week_encoded = day_of_week_mapping[day_of_week]

    # Weekend Mapping
    is_weekend_encoded = 1 if is_weekend == "Yes" else 0

    # Weather Main Mapping
    weather_main = weather.replace("Main ", "")

    # Create Input Data
    input_data = {
        "temp": temp,
        "rain_1h": 0,
        "snow_1h": 0,
        "clouds_all": clouds_all,
        "weather_description": weather_description_encoded,
        "Hour": hour,
        "Day": day,
        "Month": month_encoded,
        "Year": year,
        "Day_of_Week": day_of_week_encoded,
        "Is_Weekend": is_weekend_encoded,

        "holiday_Columbus Day": 1 if holiday == "Columbus Day" else 0,
        "holiday_Independence Day": 1 if holiday == "Independence Day" else 0,
        "holiday_Labor Day": 1 if holiday == "Labor Day" else 0,
        "holiday_Martin Luther King Jr Day": 1 if holiday == "Martin Luther King Jr Day" else 0,
        "holiday_Memorial Day": 1 if holiday == "Memorial Day" else 0,
        "holiday_New Years Day": 1 if holiday == "New Years Day" else 0,
        "holiday_No Holiday": 1 if holiday == "No Holiday" else 0,
        "holiday_State Fair": 1 if holiday == "State Fair" else 0,
        "holiday_Thanksgiving Day": 1 if holiday == "Thanksgiving Day" else 0,
        "holiday_Veterans Day": 1 if holiday == "Veterans Day" else 0,
        "holiday_Washingtons Birthday": 1 if holiday == "Washingtons Birthday" else 0,

        "weather_main_Clouds": 1 if weather_main == "Clouds" else 0,
        "weather_main_Drizzle": 1 if weather_main == "Drizzle" else 0,
        "weather_main_Fog": 1 if weather_main == "Fog" else 0,
        "weather_main_Haze": 1 if weather_main == "Haze" else 0,
        "weather_main_Mist": 1 if weather_main == "Mist" else 0,
        "weather_main_Rain": 1 if weather_main == "Rain" else 0,
        "weather_main_Smoke": 1 if weather_main == "Smoke" else 0,
        "weather_main_Snow": 1 if weather_main == "Snow" else 0,
        "weather_main_Squall": 1 if weather_main == "Squall" else 0,
        "weather_main_Thunderstorm": 1 if weather_main == "Thunderstorm" else 0
    }

    input_df = pd.DataFrame([input_data])

    input_df = input_df[feature_columns]

    # Prediction
    prediction = model.predict(input_df)[0]


# Header
st.title("🚦 Traffic Volume Prediction")

st.caption("An end-to-end machine learning application for predicting metro traffic volume")


# Top Section

left, right = st.columns([1.2, 1])

with left:

    st.subheader("Prediction")

    if prediction is not None:
        st.success(f"Predicted Traffic Volume: {prediction:.0f} vehicles per hour")
        st.warning("Model Used : LightGBM Regressor (Tuned)")
    else:
        st.info("The prediction is based on the traffic, weather, time, and calendar details provided above.")


with right:
    st.subheader("Deployed Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Model",
            "LightGBM (Tuned)"
        )

    with col2:
        st.metric(
            "R² Score",
            "0.982"
        )

    with col3:
        st.metric(
            "RMSE",
            "268.029"
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "MAE",
            "167.2658"
        )

    with col5:
        st.metric(
            "MSE",
            "71839.5384"
        )

    with col6:
        st.metric(
            "Adjusted R²",
            "0.982"
        )
st.divider()


# Selected Traffic Volume Scenario
st.subheader("Selected Traffic Volume Scenario")

scenario_df = pd.DataFrame({
    "Features": [
        "Temperature",
        "Weather Description",
        "Cloud Coverage (%)",
        "Hour",
        "Day",
        "Month",
        "Year",
        "Day of Week",
        "Is Weekend",
        "Holiday",
        "Weather"
    ],
    "Values": [
        temp,
        weather_description,
        clouds_all,
        hour,
        day,
        month,
        year,
        day_of_week,
        is_weekend,
        holiday,
        weather
    ]
})

st.dataframe(
    scenario_df,
    use_container_width=True,
    hide_index=True
)



# Baseline Model Comparison
comparison_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "SVR",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ],
    "MAE": [
        1581.2093,
        1562.2278,
        730.5900,
        251.1319,
        202.2529,
        353.2431,
        204.1998,
        214.2520,
        210.0005
    ],
    "MSE": [
        3228034,
        3145281,
        1063123,
        230164,
        127681.5951,
        282565.1,
        108501.8125,
        113242.0025,
        114221.1408
    ],
    "RMSE": [
        1796.6730,
        1773.4940,
        1031.0786,
        479.7543,
        357.3256,
        531.5685,
        329.3961,
        336.5145,
        337.9662
    ],
    "R2 Score": [
        0.1926,
        0.2133,
        0.7341,
        0.9424,
        0.9681,
        0.9293,
        0.9729,
        0.9717,
        0.9714
    ],
    "Adjusted R2": [
        0.1899,
        0.2107,
        0.7332,
        0.9422,
        0.9680,
        0.9291,
        0.9728,
        0.9716,
        0.9713
    ]
})


# Sort Baseline Models by R² Score

comparison_df = comparison_df.sort_values(
    by="R2 Score",
    ascending=False
).reset_index(drop=True)



st.subheader("Baseline Model Performance Comparison")

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)


# Baseline Model Performance Visualization
fig, ax = plt.subplots(figsize=(13, 7))

colors = ["#5B7DB1"] * len(comparison_df)
colors[0] = "#E07A3F"


bars = ax.bar(
    comparison_df["Model"],
    comparison_df["R2 Score"],
    color=colors,
    edgecolor="#3A506B",
    linewidth=1.2
)


# Title

ax.set_title(
    "Baseline Regression Model Performance Comparison",
    fontsize=20,
    fontweight="bold",
    color="#1E3A5F",
    pad=22
)


# Subtitle

ax.text(
    0.5,
    1.02,
    "Higher R² Score indicates better predictive performance",
    transform=ax.transAxes,
    ha="center",
    fontsize=12,
    color="#6B7280",
    style="italic"
)


# Axis Labels

ax.set_xlabel(
    "Machine Learning Models",
    fontsize=13,
    fontweight="bold",
    color="#52616B"
)

ax.set_ylabel(
    "R² Score",
    fontsize=13,
    fontweight="bold",
    color="#52616B"
)


# Grid

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    alpha=0.3
)

ax.set_axisbelow(True)


# X-axis

ax.set_xticks(range(len(comparison_df)))

ax.set_xticklabels(
    comparison_df["Model"],
    rotation=30,
    ha="right",
    fontsize=11,
    fontweight="bold"
)


# Y-axis

ax.tick_params(
    axis="y",
    labelsize=11
)


# Remove unnecessary borders

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.3)
ax.spines["bottom"].set_linewidth(1.3)


# Add R² value labels

for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.006,
        f"{height:.3f}",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#243447"
    )


fig.tight_layout()

st.pyplot(fig)

plt.close(fig)



# Final Model Comparison After Hyperparameter Tuning

final_comparison_df = pd.DataFrame({
    "Model": [
        "LightGBM (Tuned)",
        "XGBoost (Tuned)",
        "XGBoost (Baseline)",
        "LightGBM (Baseline)",
        "CatBoost (Baseline)",
        "CatBoost (Tuned)",
        "Random Forest (Baseline)",
        "Random Forest (Tuned)"
    ],
    "MAE": [
        167.2658,
        167.6135,
        204.1998,
        214.2520,
        210.0005,
        213.6716,
        202.2529,
        224.8820
    ],
    "MSE": [
        71839.5384,
        72986.3125,
        108501.8125,
        113242.0025,
        114221.1408,
        119193.5336,
        127681.5951,
        153499.3364
    ],
    "RMSE": [
        268.0290,
        270.1598,
        329.3961,
        336.5145,
        337.9662,
        345.2442,
        357.3256,
        391.7899
    ],
    "R2 Score": [
        0.9820,
        0.9817,
        0.9729,
        0.9717,
        0.9714,
        0.9702,
        0.9681,
        0.9616
    ],
    "Adjusted R2": [
        0.9820,
        0.9817,
        0.9728,
        0.9716,
        0.9713,
        0.9701,
        0.9680,
        0.9615
    ]
})




final_comparison_df = final_comparison_df.sort_values(
    by="R2 Score",
    ascending=False
).reset_index(drop=True)


# Display Final Model Comparison
st.subheader("Final Model Performance Comparison")

st.dataframe(
    final_comparison_df,
    use_container_width=True,
    hide_index=True
)


# Final Model Performance Visualization
fig, ax = plt.subplots(figsize=(13, 7))


colors = ["#2A9D8F"] * len(final_comparison_df)
colors[0] = "#F28E2B"


bars = ax.bar(
    final_comparison_df["Model"],
    final_comparison_df["R2 Score"],
    color=colors,
    edgecolor="#374151",
    linewidth=1.2
)


# Title

ax.set_title(
    "Final Model Performance Comparison",
    fontsize=20,
    fontweight="bold",
    color="#1E3A5F",
    pad=22
)


# Subtitle

ax.text(
    0.5,
    1.02,
    "Higher R² Score indicates better predictive performance",
    transform=ax.transAxes,
    ha="center",
    fontsize=12,
    color="#6B7280",
    style="italic"
)


# Axis Labels

ax.set_xlabel(
    "Machine Learning Models",
    fontsize=13,
    fontweight="bold",
    color="#52616B"
)

ax.set_ylabel(
    "R² Score",
    fontsize=13,
    fontweight="bold",
    color="#52616B"
)


# Grid

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    alpha=0.3
)

ax.set_axisbelow(True)


# X-axis

ax.set_xticks(range(len(final_comparison_df)))

ax.set_xticklabels(
    final_comparison_df["Model"],
    rotation=30,
    ha="right",
    fontsize=11,
    fontweight="bold"
)


# Y-axis

ax.tick_params(
    axis="y",
    labelsize=11
)


# Remove unnecessary borders

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.3)
ax.spines["bottom"].set_linewidth(1.3)


# Add R² value labels

for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.006,
        f"{height:.3f}",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#264653"
    )


fig.tight_layout()

st.pyplot(fig)

plt.close(fig)



# Top 10 Feature Importance
feature_importance_df = pd.DataFrame({
    "Feature": [
        "temp",
        "Hour",
        "Day",
        "Day_of_Week",
        "Month",
        "Year",
        "clouds_all",
        "weather_description",
        "weather_main_Clouds",
        "Is_Weekend"
    ],
    "Importance": [
        14929,
        11248,
        11075,
        7054,
        6817,
        5619,
        4821,
        4126,
        838,
        759
    ]
})


# Display Feature Importance

st.subheader("Top 10 Feature Importance")

st.dataframe(
    feature_importance_df,
    use_container_width=True,
    hide_index=True
)


# Feature Importance Visualization

fig, ax = plt.subplots(figsize=(12, 7))


feature_plot_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=True
)


bars = ax.barh(
    feature_plot_df["Feature"],
    feature_plot_df["Importance"],
    color="#5B7DB1",
    edgecolor="#3A506B",
    linewidth=1.2
)


# Title

ax.set_title(
    "Top 10 Feature Importance — LightGBM",
    fontsize=19,
    fontweight="bold",
    color="#1E3A5F",
    pad=20
)


# Subtitle

ax.text(
    0.5,
    1.02,
    "Higher importance indicates greater contribution to the model",
    transform=ax.transAxes,
    ha="center",
    fontsize=11,
    color="#6B7280",
    style="italic"
)


# Axis Labels

ax.set_xlabel(
    "Importance Score",
    fontsize=12,
    fontweight="bold",
    color="#52616B"
)

ax.set_ylabel(
    "Features",
    fontsize=12,
    fontweight="bold",
    color="#52616B"
)


# Grid

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.8,
    alpha=0.3
)

ax.set_axisbelow(True)


# Remove unnecessary borders

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)


# Add importance values

for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 200,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.0f}",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#243447"
    )


fig.tight_layout()

st.pyplot(fig)

plt.close(fig)