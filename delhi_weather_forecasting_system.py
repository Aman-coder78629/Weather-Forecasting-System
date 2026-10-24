
# ================================================================
# DELHI WEATHER FORECASTING SYSTEM
# Interactive Streamlit Dashboard + Machine Learning
# ================================================================
# Features:
# 1. Modern dashboard with custom CSS
# 2. CSV upload and built-in dataset path
# 3. Data cleaning and missing-value handling
# 4. Weather KPI cards
# 5. Eight interactive Plotly visualizations
# 6. Correlation analysis
# 7. Machine-learning model comparison
# 8. Weather forecasting using lag features
# 9. Model evaluation using MAE, RMSE, and R2
# 10. Downloadable cleaned data and forecasts
# ================================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Delhi Weather Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(49, 130, 206, 0.13),
                transparent 32%
            ),
            linear-gradient(
                135deg,
                #0b1220 0%,
                #101b2d 50%,
                #111827 100%
            );
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #101827 0%,
            #172338 100%
        );
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(30, 41, 59, 0.92),
            rgba(15, 23, 42, 0.96)
        );
        border: 1px solid rgba(125, 211, 252, 0.20);
        padding: 18px 18px;
        border-radius: 18px;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.18);
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #7dd3fc;
        font-weight: 800;
    }

    .hero {
        padding: 30px;
        border-radius: 24px;
        margin-bottom: 24px;
        background: linear-gradient(
            115deg,
            rgba(14, 165, 233, 0.25),
            rgba(99, 102, 241, 0.22),
            rgba(168, 85, 247, 0.16)
        );
        border: 1px solid rgba(125, 211, 252, 0.30);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.20);
    }

    .hero h1 {
        font-size: clamp(28px, 4vw, 44px);
        color: #f8fafc;
        margin: 0;
        font-weight: 800;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 16px;
        margin-top: 12px;
        margin-bottom: 0;
    }

    .section-heading {
        color: #e2e8f0;
        font-weight: 800;
        font-size: 23px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.20);
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .info-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 18px;
        border-radius: 16px;
        min-height: 115px;
    }

    .info-card h4 {
        color: #7dd3fc;
        margin-top: 0;
    }

    .info-card p {
        color: #cbd5e1;
        margin-bottom: 0;
    }

    div[data-testid="stTabs"] button {
        color: #cbd5e1;
        font-weight: 600;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #7dd3fc;
    }

    .small-note {
        color: #94a3b8;
        font-size: 12px;
    }

    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 700;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# CONSTANTS AND CONFIGURATION
# ================================================================

DEFAULT_DATA_PATH = (
    r"C:\Users\amans\Downloads\DailyDelhiClimateTest.csv"
)

DATE_CANDIDATES = [
    "date",
    "datetime",
    "timestamp",
    "day",
]

TARGET_CANDIDATES = [
    "meantemp",
    "temperature",
    "temp",
    "mean_temp",
]

WEATHER_VARIABLES = [
    "meantemp",
    "humidity",
    "wind_speed",
    "meanpressure",
]

CHART_TEMPLATE = "plotly_dark"

CHART_COLORS = [
    "#38bdf8",
    "#818cf8",
    "#c084fc",
    "#34d399",
    "#fbbf24",
    "#fb7185",
    "#2dd4bf",
    "#a3e635",
]


# ================================================================
# GENERAL UTILITIES
# ================================================================

def normalise_column_name(name):
    """Convert a column name into a consistent format."""
    name = str(name).strip().lower()
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = name.replace("/", "_")
    return name


def format_number(value, decimals=2):
    """Format a numeric value for KPI cards."""
    if pd.isna(value):
        return "N/A"

    return f"{float(value):,.{decimals}f}"


def find_column(columns, candidates):
    """Find a column using a list of possible names."""
    normalised = {
        normalise_column_name(column): column
        for column in columns
    }

    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]

    return None


def set_chart_style(fig, height=390):
    """Apply a consistent style to Plotly figures."""
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=18, r=18, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.30)",
        font=dict(
            family="Inter, sans-serif",
            color="#e2e8f0",
            size=12,
        ),
        title_font=dict(
            size=17,
            color="#f8fafc",
        ),
        legend=dict(
            bgcolor="rgba(15,23,42,0.35)",
            borderwidth=0,
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.15)",
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.15)",
    )

    return fig


# ================================================================
# DATA LOADING
# ================================================================

@st.cache_data(show_spinner=False)
def load_csv_from_path(path, modified_time):
    """Read a CSV from a local path."""
    del modified_time
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes):
    """Read an uploaded CSV."""
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


def get_dataset(uploaded_file, local_path):
    """Load the uploaded CSV or the configured local file."""
    if uploaded_file is not None:
        try:
            data = load_csv_from_bytes(uploaded_file.getvalue())
            return data, f"Uploaded file: {uploaded_file.name}"
        except Exception as exc:
            st.error(f"Unable to read uploaded CSV: {exc}")
            return None, None

    path = Path(local_path)

    if not path.exists():
        return None, None

    try:
        modified_time = path.stat().st_mtime
        data = load_csv_from_path(
            str(path),
            modified_time,
        )
        return data, str(path)
    except Exception as exc:
        st.error(f"Unable to read local CSV: {exc}")
        return None, None


# ================================================================
# DATA CLEANING
# ================================================================

def clean_weather_data(raw_data):
    """
    Clean weather data while preserving the original measurements.

    Operations:
    - Standardise column names
    - Remove duplicate records
    - Parse a date column if present
    - Convert possible weather measurements to numeric values
    - Replace infinite values with missing values
    - Interpolate numeric measurements
    - Fill remaining missing values with medians
    """

    data = raw_data.copy()

    data.columns = [
        normalise_column_name(column)
        for column in data.columns
    ]

    original_rows = len(data)

    data = data.drop_duplicates().reset_index(drop=True)

    date_column = find_column(
        data.columns,
        DATE_CANDIDATES,
    )

    if date_column is not None:
        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce",
        )

        data = data.dropna(
            subset=[date_column]
        )

        data = data.sort_values(
            date_column
        ).reset_index(drop=True)

    for column in data.columns:
        if column == date_column:
            continue

        if data[column].dtype == object:
            converted = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            # Convert the column only when numeric conversion
            # successfully handles most non-missing values.
            non_missing = data[column].notna().sum()

            if non_missing == 0:
                continue

            converted_count = converted.notna().sum()

            if converted_count / non_missing >= 0.75:
                data[column] = converted

        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].replace(
                [np.inf, -np.inf],
                np.nan,
            )

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    for column in numeric_columns:
        data[column] = data[column].interpolate(
            method="linear",
            limit_direction="both",
        )

        if data[column].isna().any():
            median_value = data[column].median()

            if pd.isna(median_value):
                median_value = 0

            data[column] = data[column].fillna(
                median_value
            )

    report = {
        "original_rows": original_rows,
        "cleaned_rows": len(data),
        "duplicates_removed": original_rows - len(data),
        "missing_values_remaining": int(
            data.isna().sum().sum()
        ),
        "date_column": date_column,
    }

    return data, report


# ================================================================
# WEATHER COLUMN DETECTION
# ================================================================

def detect_weather_columns(data):
    """Map standard weather measurements to available columns."""

    detected = {}

    candidates = {
        "temperature": [
            "meantemp",
            "temperature",
            "temp",
            "mean_temp",
        ],
        "humidity": [
            "humidity",
            "relative_humidity",
        ],
        "wind_speed": [
            "wind_speed",
            "windspeed",
            "wind",
        ],
        "pressure": [
            "meanpressure",
            "pressure",
            "air_pressure",
        ],
    }

    for friendly_name, options in candidates.items():
        detected[friendly_name] = find_column(
            data.columns,
            options,
        )

    detected["date"] = find_column(
        data.columns,
        DATE_CANDIDATES,
    )

    return detected


def get_numeric_columns(data):
    """Return all numeric columns."""
    return data.select_dtypes(
        include=np.number
    ).columns.tolist()


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def create_lag_features(series, lag_count=7):
    """
    Create a supervised-learning dataset from a time series.

    Each row contains previous observations and the next target.
    """
    frame = pd.DataFrame({
        "target": pd.to_numeric(
            series,
            errors="coerce",
        )
    })

    for lag in range(1, lag_count + 1):
        frame[f"lag_{lag}"] = frame["target"].shift(lag)

    frame["rolling_mean_3"] = (
        frame["target"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    frame["rolling_mean_7"] = (
        frame["target"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    frame["rolling_std_7"] = (
        frame["target"]
        .shift(1)
        .rolling(window=7)
        .std()
    )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    frame = frame.dropna().reset_index(drop=True)

    return frame


# ================================================================
# MODEL TRAINING AND EVALUATION
# ================================================================

def get_regression_models():
    """Return a selection of regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            min_samples_leaf=2,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            random_state=42,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
        "Histogram Gradient Boosting":
            HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.05,
                random_state=42,
            ),
    }


def evaluate_regression_models(series, test_fraction=0.2):
    """
    Train models with chronological splitting.

    The final portion of the time series is used for testing.
    This avoids randomly mixing future observations into training.
    """
    feature_frame = create_lag_features(
        series,
        lag_count=7,
    )

    if len(feature_frame) < 15:
        return None, None, (
            "Not enough observations to evaluate models reliably. "
            "Try a longer historical dataset."
        )

    X = feature_frame.drop(columns=["target"])
    y = feature_frame["target"]

    split_index = int(
        len(feature_frame) * (1 - test_fraction)
    )

    split_index = max(
        10,
        min(split_index, len(feature_frame) - 3),
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    results = []
    predictions = {}
    trained_models = {}

    for model_name, model in get_regression_models().items():
        try:
            model.fit(X_train, y_train)

            predicted = model.predict(X_test)

            mae = mean_absolute_error(
                y_test,
                predicted,
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predicted,
                )
            )

            if len(y_test) > 1:
                r2 = r2_score(
                    y_test,
                    predicted,
                )
            else:
                r2 = np.nan

            results.append({
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            })

            predictions[model_name] = pd.DataFrame({
                "Actual": y_test.to_numpy(),
                "Predicted": predicted,
            })

            trained_models[model_name] = model

        except Exception as exc:
            st.warning(
                f"{model_name} could not be trained: {exc}"
            )

    if not results:
        return None, None, "All selected models failed."

    result_frame = pd.DataFrame(results)

    result_frame = result_frame.sort_values(
        "MAE",
        ascending=True,
    ).reset_index(drop=True)

    return result_frame, {
        "models": trained_models,
        "predictions": predictions,
        "X_test": X_test,
        "y_test": y_test,
        "feature_frame": feature_frame,
    }, None


def forecast_future(
    series,
    fitted_model,
    horizon=7,
    lag_count=7,
):
    """
    Produce recursive forecasts from the most recent observations.

    These are model-generated estimates, not official forecasts.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna().astype(float).tolist()

    if len(values) < lag_count:
        raise ValueError(
            f"At least {lag_count} valid observations are required."
        )

    predictions = []
    working_values = values.copy()

    for _ in range(horizon):
        recent = working_values[-lag_count:]

        feature_dict = {}

        for lag in range(1, lag_count + 1):
            feature_dict[f"lag_{lag}"] = (
                working_values[-lag]
            )

        feature_dict["rolling_mean_3"] = float(
            np.mean(working_values[-3:])
        )

        feature_dict["rolling_mean_7"] = float(
            np.mean(working_values[-7:])
        )

        feature_dict["rolling_std_7"] = float(
            np.std(working_values[-7:], ddof=1)
        )

        feature_row = pd.DataFrame(
            [feature_dict]
        )

        prediction = float(
            fitted_model.predict(feature_row)[0]
        )

        predictions.append(prediction)
        working_values.append(prediction)

    return predictions


# ================================================================
# CHART 1: HISTORICAL TEMPERATURE
# ================================================================

def chart_temperature_trend(data, date_col, temp_col):
    """Line chart for temperature over time."""
    plot_data = data.copy()

    if date_col:
        x_column = date_col
        x_title = "Date"
    else:
        plot_data["Observation"] = np.arange(
            1,
            len(plot_data) + 1,
        )
        x_column = "Observation"
        x_title = "Observation number"

    fig = px.line(
        plot_data,
        x=x_column,
        y=temp_col,
        title="🌡️ Temperature Trend",
        markers=True,
        labels={
            temp_col: "Temperature",
            x_column: x_title,
        },
    )

    fig.update_traces(
        line=dict(
            color=CHART_COLORS[0],
            width=3,
        ),
        marker=dict(size=5),
    )

    fig.update_layout(
        yaxis_title="Temperature (dataset units)",
    )

    return set_chart_style(fig)


# ================================================================
# CHART 2: HUMIDITY TREND
# ================================================================

def chart_humidity_trend(data, date_col, humidity_col):
    """Line chart for humidity over time."""
    plot_data = data.copy()

    if date_col:
        x_column = date_col
    else:
        plot_data["Observation"] = np.arange(
            1,
            len(plot_data) + 1,
        )
        x_column = "Observation"

    fig = px.line(
        plot_data,
        x=x_column,
        y=humidity_col,
        title="💧 Humidity Trend",
        markers=True,
        labels={
            humidity_col: "Humidity",
            x_column: "Date / Observation",
        },
    )

    fig.update_traces(
        line=dict(
            color=CHART_COLORS[1],
            width=3,
        )
    )

    return set_chart_style(fig)


# ================================================================
# CHART 3: WIND SPEED TREND
# ================================================================

def chart_wind_trend(data, date_col, wind_col):
    """Line chart for wind speed."""
    plot_data = data.copy()

    if date_col:
        x_column = date_col
    else:
        plot_data["Observation"] = np.arange(
            1,
            len(plot_data) + 1,
        )
        x_column = "Observation"

    fig = px.line(
        plot_data,
        x=x_column,
        y=wind_col,
        title="🌬️ Wind Speed Trend",
        markers=True,
        labels={
            wind_col: "Wind speed",
            x_column: "Date / Observation",
        },
    )

    fig.update_traces(
        line=dict(
            color=CHART_COLORS[2],
            width=3,
        )
    )

    return set_chart_style(fig)


# ================================================================
# CHART 4: PRESSURE TREND
# ================================================================

def chart_pressure_trend(data, date_col, pressure_col):
    """Line chart for air pressure."""
    plot_data = data.copy()

    if date_col:
        x_column = date_col
    else:
        plot_data["Observation"] = np.arange(
            1,
            len(plot_data) + 1,
        )
        x_column = "Observation"

    fig = px.line(
        plot_data,
        x=x_column,
        y=pressure_col,
        title="🧭 Mean Pressure Trend",
        markers=True,
        labels={
            pressure_col: "Pressure",
            x_column: "Date / Observation",
        },
    )

    fig.update_traces(
        line=dict(
            color=CHART_COLORS[3],
            width=3,
        )
    )

    return set_chart_style(fig)


# ================================================================
# CHART 5: CORRELATION HEATMAP
# ================================================================

def chart_correlation(data, numeric_columns):
    """Display correlation between numeric measurements."""
    if len(numeric_columns) < 2:
        return None

    correlation = data[
        numeric_columns
    ].corr(numeric_only=True)

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="🔗 Weather Variable Correlation",
    )

    return set_chart_style(fig, height=480)


# ================================================================
# CHART 6: WEATHER VARIABLE DISTRIBUTIONS
# ================================================================

def chart_distributions(data, numeric_columns):
    """Display box plots for numeric weather measurements."""
    if not numeric_columns:
        return None

    long_data = data[numeric_columns].melt(
        var_name="Weather Variable",
        value_name="Value",
    )

    fig = px.box(
        long_data,
        x="Weather Variable",
        y="Value",
        color="Weather Variable",
        points="outliers",
        title="📦 Distribution and Outlier Overview",
        color_discrete_sequence=CHART_COLORS,
    )

    return set_chart_style(fig, height=430)


# ================================================================
# CHART 7: HISTOGRAM
# ================================================================

def chart_histogram(data, selected_column):
    """Show the distribution of the selected variable."""
    fig = px.histogram(
        data,
        x=selected_column,
        nbins=25,
        marginal="box",
        title=f"📊 Distribution of {selected_column}",
        color_discrete_sequence=[CHART_COLORS[4]],
        labels={
            selected_column: selected_column,
        },
    )

    fig.update_traces(
        marker_line_width=0.5,
        opacity=0.85,
    )

    return set_chart_style(fig)


# ================================================================
# CHART 8: ACTUAL VS PREDICTED
# ================================================================

def chart_actual_vs_predicted(prediction_data):
    """Compare actual and predicted values."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=prediction_data["Actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(
                color=CHART_COLORS[0],
                width=3,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            y=prediction_data["Predicted"],
            mode="lines+markers",
            name="Predicted",
            line=dict(
                color=CHART_COLORS[3],
                width=3,
                dash="dash",
            ),
        )
    )

    fig.update_layout(
        title="🤖 Actual vs Predicted Values",
        xaxis_title="Test sample",
        yaxis_title="Temperature",
    )

    return set_chart_style(fig)


# ================================================================
# CHART 9: MODEL COMPARISON
# ================================================================

def chart_model_comparison(model_results, metric):
    """Compare models using the selected evaluation metric."""
    fig = px.bar(
        model_results.sort_values(metric),
        x=metric,
        y="Model",
        orientation="h",
        title=f"🏆 Model Comparison — {metric}",
        color=metric,
        color_continuous_scale="Blues",
        text_auto=".3f",
    )

    fig.update_layout(
        yaxis_title="Model",
        xaxis_title=metric,
    )

    return set_chart_style(fig, height=430)


# ================================================================
# CHART 10: DAILY CHANGE
# ================================================================

def chart_daily_change(data, date_col, temp_col):
    """Visualise day-to-day temperature changes."""
    plot_data = data.copy()

    plot_data["Temperature Change"] = (
        plot_data[temp_col].diff()
    )

    if date_col:
        x_column = date_col
    else:
        plot_data["Observation"] = np.arange(
            1,
            len(plot_data) + 1,
        )
        x_column = "Observation"

    plot_data = plot_data.dropna(
        subset=["Temperature Change"]
    )

    fig = px.bar(
        plot_data,
        x=x_column,
        y="Temperature Change",
        title="↕️ Day-to-Day Temperature Change",
        color="Temperature Change",
        color_continuous_scale="RdYlBu_r",
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#94a3b8",
    )

    return set_chart_style(fig)


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.markdown(
    """
    <div style="padding: 10px 0 20px 0;">
        <h2 style="color:#7dd3fc; margin-bottom:4px;">
            🌦️ Weather AI
        </h2>
        <p style="color:#94a3b8; margin-top:0;">
            Delhi Climate Intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### ⚙️ Dashboard Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload weather CSV",
    type=["csv"],
    help="Upload a CSV containing weather measurements.",
)

local_path = st.sidebar.text_input(
    "Local CSV path",
    value=DEFAULT_DATA_PATH,
    help="Used when no file is uploaded.",
)

forecast_horizon = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=1,
    max_value=30,
    value=7,
)

test_fraction = st.sidebar.slider(
    "Test data proportion",
    min_value=0.10,
    max_value=0.35,
    value=0.20,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Machine-learning predictions are estimates, "
    "not official weather alerts."
)


# ================================================================
# HEADER
# ================================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌦️ Delhi Weather Intelligence Dashboard</h1>
        <p>
            Explore historical climate patterns, understand
            weather-variable relationships, and generate
            experimental machine-learning forecasts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# LOAD DATASET
# ================================================================

with st.spinner("Loading weather dataset..."):
    raw_data, source_name = get_dataset(
        uploaded_file,
        local_path,
    )

if raw_data is None:
    st.warning(
        "Weather CSV file nahi mili. Sidebar mein correct path "
        "enter karein ya CSV upload karein."
    )

    st.code(
        r"C:\Users\amans\Downloads\DailyDelhiClimateTest.csv"
    )

    st.info(
        "File upload karne ke baad dashboard automatically "
        "load ho jayega."
    )

    st.stop()

if raw_data.empty:
    st.error("The selected CSV contains no data.")
    st.stop()

data, cleaning_report = clean_weather_data(
    raw_data
)

if data.empty:
    st.error(
        "No usable rows remain after cleaning the dataset."
    )
    st.stop()

detected = detect_weather_columns(data)
numeric_columns = get_numeric_columns(data)

date_column = detected["date"]
temperature_column = detected["temperature"]
humidity_column = detected["humidity"]
wind_column = detected["wind_speed"]
pressure_column = detected["pressure"]

st.caption(f"**Data source:** {source_name}")


# ================================================================
# DATA QUALITY NOTICES
# ================================================================

if len(data) < 30:
    st.warning(
        "Dataset mein 30 se kam rows hain. Forecasting aur "
        "model evaluation ko reliable banane ke liye longer "
        "historical data use karein."
    )

if date_column is None:
    st.info(
        "Date column detect nahi hua. Charts observation number "
        "ke against display honge."
    )

if temperature_column is None:
    st.warning(
        "Temperature column detect nahi hua. Forecasting "
        "features available nahi honge. CSV column names check karein."
    )


# ================================================================
# DASHBOARD NAVIGATION
# ================================================================

overview_tab, analytics_tab, ml_tab, data_tab = st.tabs([
    "🏠 Overview",
    "📊 Analytics",
    "🤖 Machine Learning",
    "🗃️ Dataset",
])


# ================================================================
# OVERVIEW TAB
# ================================================================

with overview_tab:

    st.markdown(
        '<div class="section-heading">Weather at a Glance</div>',
        unsafe_allow_html=True,
    )

    kpi_columns = st.columns(4)

    if temperature_column:
        current_temp = data[
            temperature_column
        ].iloc[-1]

        average_temp = data[
            temperature_column
        ].mean()

        min_temp = data[
            temperature_column
        ].min()

        max_temp = data[
            temperature_column
        ].max()

        kpi_columns[0].metric(
            "Latest Temperature",
            f"{format_number(current_temp)}",
        )

        kpi_columns[1].metric(
            "Average Temperature",
            f"{format_number(average_temp)}",
        )

        kpi_columns[2].metric(
            "Minimum Temperature",
            f"{format_number(min_temp)}",
        )

        kpi_columns[3].metric(
            "Maximum Temperature",
            f"{format_number(max_temp)}",
        )

    else:
        kpi_columns[0].metric(
            "Dataset Rows",
            f"{len(data):,}",
        )

        kpi_columns[1].metric(
            "Dataset Columns",
            f"{len(data.columns):,}",
        )

        kpi_columns[2].metric(
            "Numeric Variables",
            f"{len(numeric_columns):,}",
        )

        kpi_columns[3].metric(
            "Missing Values",
            f"{int(data.isna().sum().sum()):,}",
        )

    st.markdown(
        '<div class="section-heading">Climate Indicators</div>',
        unsafe_allow_html=True,
    )

    indicator_columns = st.columns(3)

    if humidity_column:
        indicator_columns[0].metric(
            "Average Humidity",
            format_number(data[humidity_column].mean()),
        )
    else:
        indicator_columns[0].metric(
            "Average Humidity",
            "Unavailable",
        )

    if wind_column:
        indicator_columns[1].metric(
            "Average Wind Speed",
            format_number(data[wind_column].mean()),
        )
    else:
        indicator_columns[1].metric(
            "Average Wind Speed",
            "Unavailable",
        )

    if pressure_column:
        indicator_columns[2].metric(
            "Average Pressure",
            format_number(data[pressure_column].mean()),
        )
    else:
        indicator_columns[2].metric(
            "Average Pressure",
            "Unavailable",
        )

    st.markdown(
        '<div class="section-heading">Historical Weather</div>',
        unsafe_allow_html=True,
    )

    if temperature_column:
        st.plotly_chart(
            chart_temperature_trend(
                data,
                date_column,
                temperature_column,
            ),
            use_container_width=True,
        )

    left, right = st.columns(2)

    if humidity_column:
        with left:
            st.plotly_chart(
                chart_humidity_trend(
                    data,
                    date_column,
                    humidity_column,
                ),
                use_container_width=True,
            )

    if wind_column:
        with right:
            st.plotly_chart(
                chart_wind_trend(
                    data,
                    date_column,
                    wind_column,
                ),
                use_container_width=True,
            )

    st.markdown(
        '<div class="section-heading">Data Quality Summary</div>',
        unsafe_allow_html=True,
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Original Rows",
        f"{cleaning_report['original_rows']:,}",
    )

    q2.metric(
        "Cleaned Rows",
        f"{cleaning_report['cleaned_rows']:,}",
    )

    q3.metric(
        "Duplicate Rows Removed",
        f"{cleaning_report['duplicates_removed']:,}",
    )

    q4.metric(
        "Remaining Missing Values",
        f"{cleaning_report['missing_values_remaining']:,}",
    )


# ================================================================
# ANALYTICS TAB
# ================================================================

with analytics_tab:

    st.markdown(
        '<div class="section-heading">Exploratory Data Analysis</div>',
        unsafe_allow_html=True,
    )

    if not numeric_columns:
        st.warning(
            "No numeric columns are available for analytics."
        )

    else:
        selected_variable = st.selectbox(
            "Select a weather variable",
            options=numeric_columns,
            index=(
                numeric_columns.index(temperature_column)
                if temperature_column in numeric_columns
                else 0
            ),
        )

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                chart_histogram(
                    data,
                    selected_variable,
                ),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                chart_distributions(
                    data,
                    numeric_columns,
                ),
                use_container_width=True,
            )

        if len(numeric_columns) >= 2:
            st.plotly_chart(
                chart_correlation(
                    data,
                    numeric_columns,
                ),
                use_container_width=True,
            )

        if temperature_column:
            st.plotly_chart(
                chart_daily_change(
                    data,
                    date_column,
                    temperature_column,
                ),
                use_container_width=True,
            )

        st.markdown(
            '<div class="section-heading">Descriptive Statistics</div>',
            unsafe_allow_html=True,
        )

        descriptive_stats = data[
            numeric_columns
        ].describe().T

        descriptive_stats["median"] = data[
            numeric_columns
        ].median()

        descriptive_stats["missing_values"] = data[
            numeric_columns
        ].isna().sum()

        st.dataframe(
            descriptive_stats.round(3),
            use_container_width=True,
        )

        st.markdown(
            '<div class="section-heading">Correlation Table</div>',
            unsafe_allow_html=True,
        )

        if len(numeric_columns) >= 2:
            correlation_table = data[
                numeric_columns
            ].corr(numeric_only=True)

            st.dataframe(
                correlation_table.round(3),
                use_container_width=True,
            )


# ================================================================
# MACHINE LEARNING TAB
# ================================================================

with ml_tab:

    st.markdown(
        '<div class="section-heading">Machine Learning Studio</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>How this forecasting model works</h4>
            <p>
                The app uses previous temperature observations
                as lag features. The data is split chronologically:
                earlier observations are used for training and later
                observations are used for testing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if temperature_column is None:

        st.error(
            "A temperature column is required for this forecasting "
            "workflow. Use a CSV containing meantemp or temperature."
        )

    elif len(data) < 25:

        st.error(
            "The dataset is too small for the current model "
            "comparison. Please provide at least 25 valid temperature "
            "observations; a substantially longer history is preferable."
        )

    else:

        temperature_series = data[
            temperature_column
        ].astype(float)

        results, model_bundle, model_error = (
            evaluate_regression_models(
                temperature_series,
                test_fraction=test_fraction,
            )
        )

        if model_error:
            st.error(model_error)

        elif results is not None:

            best_model_name = results.iloc[0]["Model"]

            best_mae = results.iloc[0]["MAE"]
            best_rmse = results.iloc[0]["RMSE"]
            best_r2 = results.iloc[0]["R2"]

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Best Model",
                best_model_name,
            )

            m2.metric(
                "Test MAE",
                format_number(best_mae),
            )

            m3.metric(
                "Test RMSE",
                format_number(best_rmse),
            )

            st.caption(
                "Lower MAE and RMSE generally indicate smaller "
                "prediction errors. R² can be negative when a model "
                "performs worse than a simple mean baseline."
            )

            st.markdown(
                '<div class="section-heading">Model Comparison</div>',
                unsafe_allow_html=True,
            )

            metric_to_compare = st.selectbox(
                "Compare models using",
                options=["MAE", "RMSE", "R2"],
                index=0,
            )

            st.plotly_chart(
                chart_model_comparison(
                    results,
                    metric_to_compare,
                ),
                use_container_width=True,
            )

            st.dataframe(
                results.round(4),
                use_container_width=True,
            )

            st.markdown(
                '<div class="section-heading">Prediction Performance</div>',
                unsafe_allow_html=True,
            )

            prediction_data = model_bundle[
                "predictions"
            ][best_model_name]

            st.plotly_chart(
                chart_actual_vs_predicted(
                    prediction_data,
                ),
                use_container_width=True,
            )

            st.markdown(
                '<div class="section-heading">Future Forecast</div>',
                unsafe_allow_html=True,
            )

            best_model = model_bundle[
                "models"
            ][best_model_name]

            # Retrain the selected model using all available
            # lag-feature rows before generating future estimates.
            full_feature_frame = create_lag_features(
                temperature_series,
                lag_count=7,
            )

            full_X = full_feature_frame.drop(
                columns=["target"]
            )

            full_y = full_feature_frame["target"]

            best_model.fit(
                full_X,
                full_y,
            )

            try:
                future_values = forecast_future(
                    temperature_series,
                    best_model,
                    horizon=forecast_horizon,
                    lag_count=7,
                )

                if date_column:
                    last_date = pd.to_datetime(
                        data[date_column].iloc[-1]
                    )

                    future_dates = [
                        last_date + timedelta(days=step)
                        for step in range(
                            1,
                            forecast_horizon + 1,
                        )
                    ]

                else:
                    future_dates = [
                        f"Day +{step}"
                        for step in range(
                            1,
                            forecast_horizon + 1,
                        )
                    ]

                forecast_frame = pd.DataFrame({
                    "Forecast Date": future_dates,
                    "Predicted Temperature": future_values,
                })

                fig_forecast = go.Figure()

                if date_column:
                    history_x = pd.to_datetime(
                        data[date_column]
                    )

                    forecast_x = pd.to_datetime(
                        forecast_frame["Forecast Date"]
                    )

                else:
                    history_x = np.arange(
                        len(temperature_series)
                    )

                    forecast_x = np.arange(
                        len(temperature_series),
                        len(temperature_series)
                        + forecast_horizon,
                    )

                fig_forecast.add_trace(
                    go.Scatter(
                        x=history_x,
                        y=temperature_series,
                        mode="lines",
                        name="Historical Temperature",
                        line=dict(
                            color=CHART_COLORS[0],
                            width=3,
                        ),
                    )
                )

                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast_x,
                        y=future_values,
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(
                            color=CHART_COLORS[4],
                            width=3,
                            dash="dash",
                        ),
                        marker=dict(size=8),
                    )
                )

                fig_forecast.update_layout(
                    title=(
                        f"🌤️ Temperature Forecast "
                        f"for the Next {forecast_horizon} Days"
                    ),
                    xaxis_title="Date / Observation",
                    yaxis_title="Predicted Temperature",
                )

                st.plotly_chart(
                    set_chart_style(
                        fig_forecast,
                        height=450,
                    ),
                    use_container_width=True,
                )

                st.dataframe(
                    forecast_frame,
                    use_container_width=True,
                )

                st.download_button(
                    label="⬇️ Download Forecast CSV",
                    data=forecast_frame.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="delhi_weather_forecast.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as exc:
                st.error(
                    f"Unable to generate future forecasts: {exc}"
                )


# ================================================================
# DATASET TAB
# ================================================================

with data_tab:

    st.markdown(
        '<div class="section-heading">Dataset Explorer</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Explore the cleaned dataset, inspect column types, "
        "and download the processed data."
    )

    search_text = st.text_input(
        "Search column names",
        placeholder="Type a column name...",
    )

    visible_columns = [
        column
        for column in data.columns
        if search_text.lower() in column.lower()
    ]

    if not visible_columns:
        st.warning("No columns match your search.")
    else:
        st.dataframe(
            data[visible_columns],
            use_container_width=True,
            height=420,
        )

    st.markdown(
        '<div class="section-heading">Column Information</div>',
        unsafe_allow_html=True,
    )

    column_info = pd.DataFrame({
        "Column": data.columns,
        "Data Type": [
            str(data[column].dtype)
            for column in data.columns
        ],
        "Non-null Count": [
            int(data[column].notna().sum())
            for column in data.columns
        ],
        "Missing Count": [
            int(data[column].isna().sum())
            for column in data.columns
        ],
        "Unique Values": [
            int(data[column].nunique(dropna=True))
            for column in data.columns
        ],
    })

    st.dataframe(
        column_info,
        use_container_width=True,
    )

    st.download_button(
        label="⬇️ Download Cleaned Dataset",
        data=data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="delhi_weather_cleaned.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label="⬇️ Download Column Information",
        data=column_info.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="weather_column_information.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:18px 0;">
        <h4 style="color:#7dd3fc;">
            🌦️ Delhi Weather Intelligence
        </h4>
        <p style="color:#94a3b8; font-size:13px;">
            Built with Python, Streamlit, Plotly, Pandas
            and Scikit-learn.
        </p>
        <p style="color:#64748b; font-size:12px;">
            Forecasts are experimental model estimates.
            They are not a substitute for official meteorological
            forecasts or emergency weather warnings.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
