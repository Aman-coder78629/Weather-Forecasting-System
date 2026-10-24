import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =====================================================
# 1. STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Delhi Weather Forecasting System",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Delhi Weather Forecasting System")
st.caption("Machine Learning | Climate Analytics | Weather Prediction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATHS = [
    os.path.join(BASE_DIR, "DailyDelhiClimateTest.csv"),
    r"C:\Users\amans\Downloads\DailyDelhiClimateTest.csv"
]

# =====================================================
# 2. LOAD DATASET
# =====================================================

@st.cache_data
def load_data(path, modified_time):
    df = pd.read_csv(path)

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    if "date" not in df.columns:
        raise ValueError("Dataset must contain a date column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    targets = df.select_dtypes(include=np.number).columns.tolist()

    if not targets:
        raise ValueError("No numeric weather columns found.")

    df[targets] = df[targets].interpolate(limit_direction="both")
    df = df.dropna(subset=targets).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("At least 10 valid rows are required.")

    return df, targets


with st.sidebar:
    st.header("Dashboard Settings")
    uploaded_file = st.file_uploader(
        "Upload weather CSV",
        type=["csv"]
    )

    forecast_days = st.slider(
        "Forecast Days",
        min_value=1,
        max_value=30,
        value=7
    )

try:
    if uploaded_file is not None:
        history = pd.read_csv(uploaded_file)

        history.columns = (
            history.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        if "date" not in history.columns:
            st.error("Uploaded CSV must contain a date column.")
            st.stop()

        history["date"] = pd.to_datetime(
            history["date"], errors="coerce"
        )

        history = (
            history.dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        for col in history.columns:
            if col != "date":
                history[col] = pd.to_numeric(
                    history[col], errors="coerce"
                )

        targets = history.select_dtypes(
            include=np.number
        ).columns.tolist()

        if not targets:
            st.error("No numeric weather variables found.")
            st.stop()

        history[targets] = history[targets].interpolate(
            limit_direction="both"
        )

        history = history.dropna(
            subset=targets
        ).reset_index(drop=True)

    else:
        dataset_path = next(
            (p for p in DATA_PATHS if os.path.isfile(p)),
            None
        )

        if dataset_path is None:
            st.error(
                "Dataset not found. Place DailyDelhiClimateTest.csv "
                "in the script folder or Downloads folder, "
                "or upload it using the sidebar."
            )
            st.stop()

        history, targets = load_data(
            dataset_path,
            os.path.getmtime(dataset_path)
        )

except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.stop()

if len(history) < 10:
    st.error("At least 10 valid records are required.")
    st.stop()

# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================

def add_features(df, target_columns):
    result = df.copy()

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day
    result["day_of_week"] = result["date"].dt.dayofweek
    result["day_of_year"] = result["date"].dt.dayofyear

    result["month_sin"] = np.sin(
        2 * np.pi * result["month"] / 12
    )

    result["month_cos"] = np.cos(
        2 * np.pi * result["month"] / 12
    )

    for col in target_columns:
        for lag in [1, 2, 3, 7]:
            result[f"{col}_lag_{lag}"] = (
                result[col].shift(lag)
            )

        result[f"{col}_rolling_3"] = (
            result[col].shift(1).rolling(3).mean()
        )

        result[f"{col}_rolling_7"] = (
            result[col].shift(1).rolling(7).mean()
        )

    return result


featured_data = add_features(history, targets)

feature_columns = [
    col for col in featured_data.columns
    if col not in ["date"] + targets
]

model_data = (
    featured_data
    .dropna(subset=feature_columns + targets)
    .reset_index(drop=True)
)

if len(model_data) < 8:
    st.error(
        "Not enough records after feature engineering. "
        "Use a larger dataset."
    )
    st.stop()

# =====================================================
# 4. MODEL EVALUATION
# =====================================================

model_options = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        random_state=42
    ),
    "Linear Regression": LinearRegression()
}


@st.cache_data
def evaluate_models(data, target_columns, features):
    split = int(len(data) * 0.8)
    split = max(1, min(split, len(data) - 1))

    train = data.iloc[:split]
    test = data.iloc[split:]

    rows = []

    for target in target_columns:
        for model_name, prototype in model_options.items():
            model = prototype

            model.fit(
                train[features],
                train[target]
            )

            predictions = model.predict(test[features])

            rows.append({
                "Target": target,
                "Model": model_name,
                "MAE": mean_absolute_error(
                    test[target], predictions
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(
                        test[target], predictions
                    )
                ),
                "R2": (
                    r2_score(test[target], predictions)
                    if len(test) > 1 else np.nan
                )
            })

    return pd.DataFrame(rows)


try:
    evaluation_results = evaluate_models(
        model_data,
        targets,
        feature_columns
    )

    trained_models = {}

    for target in targets:
        target_results = evaluation_results[
            evaluation_results["Target"] == target
        ]

        best_model_name = (
            target_results.sort_values("MAE")
            .iloc[0]["Model"]
        )

        best_model = model_options[best_model_name]

        best_model.fit(
            model_data[feature_columns],
            model_data[target]
        )

        trained_models[target] = best_model

except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()

# =====================================================
# 5. WEATHER FORECASTING FUNCTION
# =====================================================

def create_forecast(days):
    forecast_history = history.copy()
    last_date = forecast_history["date"].max()
    forecast_rows = []

    for step in range(1, days + 1):
        forecast_date = (
            last_date + pd.Timedelta(days=step)
        )

        row = {
            "year": forecast_date.year,
            "month": forecast_date.month,
            "day": forecast_date.day,
            "day_of_week": forecast_date.dayofweek,
            "day_of_year": forecast_date.dayofyear,
            "month_sin": np.sin(
                2 * np.pi * forecast_date.month / 12
            ),
            "month_cos": np.cos(
                2 * np.pi * forecast_date.month / 12
            )
        }

        for col in targets:
            values = (
                pd.to_numeric(
                    forecast_history[col],
                    errors="coerce"
                )
                .dropna()
                .astype(float)
                .tolist()
            )

            if len(values) < 7:
                raise ValueError(
                    f"Not enough history for {col}."
                )

            for lag in [1, 2, 3, 7]:
                row[f"{col}_lag_{lag}"] = values[-lag]

            row[f"{col}_rolling_3"] = np.mean(values[-3:])
            row[f"{col}_rolling_7"] = np.mean(values[-7:])

        features = pd.DataFrame(
            [[row[col] for col in feature_columns]],
            columns=feature_columns
        )

        forecast_row = {"date": forecast_date}

        for target, model in trained_models.items():
            prediction = float(
                np.asarray(
                    model.predict(features)
                ).ravel()[0]
            )

            forecast_row[target] = prediction

        forecast_rows.append(forecast_row)

        forecast_history = pd.concat(
            [
                forecast_history,
                pd.DataFrame([forecast_row])
            ],
            ignore_index=True
        )

    return pd.DataFrame(forecast_rows)


# =====================================================
# 6. DASHBOARD NAVIGATION
# =====================================================

page = st.sidebar.radio(
    "Navigate Dashboard",
    [
        "Overview",
        "Weather Forecast",
        "Historical Analysis",
        "Model Evaluation",
        "Dataset Explorer"
    ]
)

# =====================================================
# 7. OVERVIEW
# =====================================================

if page == "Overview":
    st.header("📊 Project Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Historical Records", len(history))
    c2.metric("Weather Variables", len(targets))
    c3.metric(
        "Latest Historical Date",
        history["date"].max().strftime("%d %b %Y")
    )

    st.subheader("Historical Weather Trends")

    selected = st.selectbox(
        "Select weather variable",
        targets
    )

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        history["date"],
        history[selected],
        linewidth=2
    )

    ax.set_title(f"Historical {selected}")
    ax.set_xlabel("Date")
    ax.set_ylabel(selected)
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Recent Dataset Records")

    st.dataframe(
        history.tail(10),
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# 8. WEATHER FORECAST
# =====================================================

elif page == "Weather Forecast":
    st.header("🌤️ Weather Forecast")

    st.caption(
        "These are model-generated estimates based on historical "
        "data, not official or live weather-service forecasts."
    )

    if st.button("Generate Forecast"):
        try:
            st.session_state["forecast_data"] = create_forecast(
                forecast_days
            )
            st.session_state["forecast_horizon"] = forecast_days

        except Exception as error:
            st.error(f"Forecast generation failed: {error}")

    forecast = st.session_state.get("forecast_data")

    if (
        forecast is None
        or st.session_state.get("forecast_horizon") != forecast_days
    ):
        try:
            forecast = create_forecast(forecast_days)
            st.session_state["forecast_data"] = forecast
            st.session_state["forecast_horizon"] = forecast_days

        except Exception as error:
            st.error(f"Forecast generation failed: {error}")
            st.stop()

    st.subheader("Forecast Results")

    st.dataframe(
        forecast.round(3),
        use_container_width=True,
        hide_index=True
    )

    selected = st.selectbox(
        "Select forecast variable",
        targets,
        key="forecast_variable"
    )

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        forecast["date"],
        forecast[selected],
        marker="o"
    )

    ax.set_title(f"Forecasted {selected}")
    ax.set_xlabel("Date")
    ax.set_ylabel(selected)
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download Forecast CSV",
        forecast.to_csv(index=False).encode("utf-8"),
        "weather_forecast.csv",
        "text/csv"
    )

# =====================================================
# 9. HISTORICAL ANALYSIS
# =====================================================

elif page == "Historical Analysis":
    st.header("📈 Historical Climate Analysis")

    selected = st.selectbox(
        "Select climate variable",
        targets,
        key="historical_variable"
    )

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        history["date"],
        history[selected]
    )

    ax.set_title(f"Historical {selected}")
    ax.set_xlabel("Date")
    ax.set_ylabel(selected)
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Statistical Summary")

    st.dataframe(
        history[targets].describe().T.round(3),
        use_container_width=True
    )

    st.subheader("Correlation Matrix")

    st.dataframe(
        history[targets].corr().round(3),
        use_container_width=True
    )

# =====================================================
# 10. MODEL EVALUATION
# =====================================================

elif page == "Model Evaluation":
    st.header("🤖 Machine Learning Model Evaluation")

    st.write(
        "Models are evaluated on a chronological 80/20 holdout."
    )

    st.dataframe(
        evaluation_results.round(4),
        use_container_width=True,
        hide_index=True
    )

    selected = st.selectbox(
        "Select target variable",
        targets,
        key="evaluation_target"
    )

    subset = evaluation_results[
        evaluation_results["Target"] == selected
    ].sort_values("MAE")

    if not subset.empty:
        best = subset.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric("Lowest-MAE Model", best["Model"])
        c2.metric("MAE", f'{best["MAE"]:.4f}')
        c3.metric("RMSE", f'{best["RMSE"]:.4f}')

    st.download_button(
        "Download Evaluation Results",
        evaluation_results.to_csv(index=False).encode("utf-8"),
        "model_results.csv",
        "text/csv"
    )

# =====================================================
# 11. DATASET EXPLORER
# =====================================================

elif page == "Dataset Explorer":
    st.header("🗂️ Dataset Explorer")

    st.write("Dataset shape:", history.shape)

    st.subheader("Dataset Preview")

    st.dataframe(
        history.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Missing Values")

    st.dataframe(
        history.isnull().sum().to_frame("Missing Values"),
        use_container_width=True
    )

    st.subheader("Column Data Types")

    st.dataframe(
        history.dtypes.astype(str).to_frame("Data Type"),
        use_container_width=True
    )

    st.download_button(
        "Download Cleaned Dataset",
        history.to_csv(index=False).encode("utf-8"),
        "cleaned_weather_dataset.csv",
        "text/csv"
    )

# =====================================================
# 12. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    **Delhi Weather Forecasting System**

    Built using Python, Pandas, NumPy, Scikit-learn,
    Matplotlib, and Streamlit.

    **Disclaimer:** Predictions are model-based estimates,
    not official meteorological forecasts.
    """
)
