"""
dynamicforecast.py
Compatible with your Streamlit app.py
Includes:
 - DynamicForecast (Prophet wrapper)
 - forecast_demand
 - plot_price_sensitivity
 - plot_category_price_scan
 - plot_forecast
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from prophet import Prophet
import statsmodels.api as sm


# -----------------------------
# Dataset utilities (optional)
# -----------------------------
DEFAULT_CSV = Path(r"C:\Users\krish\Downloads\ecommerce_dynamic_pricing_dataset.csv")


def load_dataset(csv_path=None):
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def prepare_dataset(ecommerce: pd.DataFrame):
    ecommerce = ecommerce.copy()
    ecommerce["Purchase_Timestamp"] = pd.to_datetime(
        ecommerce["Purchase_Timestamp"], errors="coerce"
    )
    ecommerce["Date"] = ecommerce["Purchase_Timestamp"].dt.date
    ecommerce["Month"] = ecommerce["Purchase_Timestamp"].dt.month
    ecommerce["DayOfWeek"] = ecommerce["Purchase_Timestamp"].dt.dayofweek

    ecommerce["demand"] = 1

    cols_to_check = [c for c in ecommerce.columns if c != "Transaction_ID"]
    ecommerce = (
        ecommerce.groupby(cols_to_check, as_index=False)
        .agg({"demand": "sum"})
    )

    unique_rows = ecommerce.drop_duplicates()
    daily_demand = (
        unique_rows.groupby(["Date", "Product_Category"])["demand"]
        .sum()
        .reset_index()
    )
    daily_demand["Date"] = pd.to_datetime(daily_demand["Date"])

    if "competitor1_price" not in ecommerce.columns:
        ecommerce["competitor1_price"] = (
            ecommerce["Price"].astype(float)
            * np.random.uniform(0.9, 1.1, len(ecommerce))
        ).astype(int)

    if "competitor2_price" not in ecommerce.columns:
        ecommerce["competitor2_price"] = (
            ecommerce["Price"].astype(float)
            * np.random.uniform(0.85, 1.15, len(ecommerce))
        ).astype(int)

    if "Discount" in ecommerce.columns:
        ecommerce["Effective_Price"] = ecommerce["Price"] - ecommerce["Discount"]
    else:
        ecommerce["Effective_Price"] = ecommerce["Price"]

    return ecommerce, daily_demand


# -----------------------------
# Elasticity estimation (OLS)
# -----------------------------
def estimate_betas(ecommerce: pd.DataFrame, min_samples: int = 10):
    betas = {}
    for cat in ecommerce["Product_Category"].unique():
        df_cat = ecommerce[ecommerce["Product_Category"] == cat]
        if len(df_cat) > min_samples and (df_cat["Price"] > 0).any():
            X = np.log(df_cat["Price"] + 1e-9)
            y = np.log(df_cat["demand"] + 1e-9)
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            betas[cat] = float(model.params[1])
    return betas


# -----------------------------
# Prophet forecasting wrapper
# -----------------------------
class DynamicForecast:
    def __init__(self):
        self.model = None

    def fit(self, df, regressors=None):
        self.model = Prophet(yearly_seasonality=True)
        if regressors:
            for r in regressors:
                self.model.add_regressor(r)
        self.model.fit(df)
        return self

    def forecast(self, df, periods=30, regressors=None):
        if self.model is None:
            self.fit(df, regressors=list(regressors.keys()) if regressors else None)

        future = self.model.make_future_dataframe(periods=periods)

        if regressors:
            for r, v in regressors.items():
                future[r] = v

        return self.model.predict(future)


def forecast_demand(history_df: pd.DataFrame, periods: int = 30):
    return DynamicForecast().forecast(history_df, periods=periods)


# -----------------------------
# SIMPLE Sensitivity Plots (compatible with app.py)
# -----------------------------
def plot_price_sensitivity(Q0, P0, betas, price_min, price_max, n=80, cost=0):
    """
    Simple Demand/Revenue/Profit curves.
    100% compatible with Streamlit app.py.
    """
    prices = np.linspace(price_min, price_max, n)

    demand = Q0 * (P0 / prices) ** abs(betas)
    revenue = prices * demand
    profit = (prices - cost) * demand

    fig_demand = px.line(
        x=prices, y=demand,
        title="Demand vs Price",
        labels={"x": "Price", "y": "Demand"}
    )

    fig_revenue = px.line(
        x=prices, y=revenue,
        title="Revenue vs Price",
        labels={"x": "Price", "y": "Revenue"}
    )

    fig_profit = px.line(
        x=prices, y=profit,
        title="Profit vs Price",
        labels={"x": "Price", "y": "Profit"}
    )

    return fig_demand, fig_revenue, fig_profit


# -----------------------------
# SIMPLE Category Scan (Heatmap + Histogram)
# -----------------------------
def plot_category_price_scan(df_final, category, betas):
    df = df_final[df_final["Product_Category"] == category]

    if df.empty:
        return None, None

    df = df.copy()
    base_price = df["Price"].mean()
    base_demand = df["demand"].mean()

    df["predicted_demand"] = base_demand * (base_price / df["Price"]) ** abs(betas)
    df["revenue"] = df["Price"] * df["predicted_demand"]

    heatmap = px.scatter(
        df,
        x="Price", y="predicted_demand",
        color="revenue",
        title=f"Price–Demand–Revenue Heatmap ({category})"
    )

    hist = px.histogram(
        df,
        x="Price",
        nbins=20,
        title=f"Price Distribution for {category}"
    )

    return heatmap, hist


# -----------------------------
# Forecast Plot (matplotlib)
# -----------------------------
def plot_forecast(combined: pd.DataFrame, category: str):
    df_cat = combined[combined.get("Product_Category") == category]

    if df_cat.empty:
        raise ValueError("No data for selected category")

    history = df_cat[df_cat.get("type") == "history"]
    forecast = df_cat[df_cat.get("type") == "forecast"]

    plt.figure(figsize=(10, 5))

    if "y" in history.columns:
        plt.plot(history["ds"], history["y"], label="History", color="blue")

    if "yhat" in forecast.columns:
        plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="red")

    if not forecast.empty:
        plt.axvline(forecast["ds"].min(), color="black", linestyle="--", label="Forecast Start")

    plt.title(f"Forecast for category: {category}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------
# Public API
# -----------------------------
__all__ = [
    "DynamicForecast",
    "forecast_demand",
    "plot_price_sensitivity",
    "plot_category_price_scan",
    "plot_forecast",
    "load_dataset",
    "prepare_dataset",
    "estimate_betas",
]


# Standalone demo (ignored when imported)
if __name__ == "__main__":
    print("dynamicforecast module loaded successfully")
