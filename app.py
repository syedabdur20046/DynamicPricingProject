import streamlit as st
import pandas as pd
import plotly.express as px
from importlib import import_module

# Try import Prophet (optional). If missing or broken, fallback to SARIMAX.
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# Lazy import for dynamicforecast
def _get_dynamicforecast():
    try:
        mod = import_module("dynamicforecast")
    except Exception:
        return None, None, None

    forecast_fn = getattr(mod, "forecast_demand", None)
    plot_price_sensitivity_fn = getattr(mod, "plot_price_sensitivity", None)
    plot_category_price_scan_fn = getattr(mod, "plot_category_price_scan", None)

    # Fallback to DynamicForecast class if exists
    if forecast_fn is None and hasattr(mod, "DynamicForecast"):
        def forecast_fn(df, periods=30):
            return mod.DynamicForecast().forecast(df, periods)

    return forecast_fn, plot_price_sensitivity_fn, plot_category_price_scan_fn


# ---------- Streamlit UI Config ----------
st.set_page_config(page_title="Dynamic Price Prediction Dashboard", layout="wide", page_icon="💹")

# ---------- Sidebar ----------
st.sidebar.title("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.sidebar.markdown("---")
page = st.sidebar.radio("Choose Page", ["🏠 Home", "📊 EDA", "📈 Forecasting",
                                        "💰 Price Recommendation", "🔬 Sensitivity"])

st.sidebar.markdown("---")
st.sidebar.info("Demand = number of transactions per Product_ID per day.")

st.title("💹 Dynamic Price Prediction Dashboard")

# ---------- Load CSV ----------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
    except Exception as e:
        st.error("Failed to read CSV file.")
        st.exception(e)
        st.stop()
else:
    df = st.session_state.get("df", None)


# ---------- Required Columns ----------
REQUIRED_MIN = ["Purchase_Timestamp", "Product_ID", "Product_Category", "Price"]

def check_required(df):
    return [c for c in REQUIRED_MIN if c not in df.columns]


# ---------- Process Data ----------
if df is not None:
    missing = check_required(df)

    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    df["Purchase_Timestamp"] = pd.to_datetime(df["Purchase_Timestamp"], errors="coerce")

    df["ds"] = df["Purchase_Timestamp"].dt.date

    demand_df = (
        df.groupby(["ds", "Product_ID", "Product_Category"])
          .size()
          .reset_index(name="demand")
    )

    merge_cols = ["Price"]
    if "Discount" in df.columns:
        merge_cols.append("Discount")

    temp_agg = (
        df.groupby(["ds", "Product_ID", "Product_Category"])[merge_cols]
          .mean()
          .reset_index()
    )

    df_final = demand_df.merge(temp_agg, on=["ds", "Product_ID", "Product_Category"])

    # Cost Calculation
    if "cost_per_unit" not in df.columns:
        if "Discount" in df_final.columns:
            disc = df_final["Discount"].fillna(0)
            max_disc = disc.max()

            if max_disc <= 1.0:
                cost = df_final["Price"] * (1 - disc)
            elif max_disc <= 100.0:
                cost = df_final["Price"] * (1 - disc / 100)
            else:
                cost = df_final["Price"] - disc

            cost = cost.clip(lower=0)
            df_final["cost_per_unit"] = cost

        else:
            df_final["cost_per_unit"] = df_final["Price"] * 0.8
    else:
        cost_agg = df.groupby(["ds", "Product_ID", "Product_Category"])["cost_per_unit"].mean().reset_index()
        df_final = df_final.merge(cost_agg, on=["ds", "Product_ID", "Product_Category"], how="left")

    df_final["profit_per_unit"] = df_final["Price"] - df_final["cost_per_unit"]
    df_final["total_profit"] = df_final["profit_per_unit"] * df_final["demand"]

    st.session_state["df_final"] = df_final

    # =====================================================================
    #                              PAGE: HOME
    # =====================================================================
    if page == "🏠 Home":
        st.subheader("Welcome!")
        st.write("Upload your dataset to begin.")
        st.dataframe(df_final.head(20))

    # =====================================================================
    #                               PAGE: EDA
    # =====================================================================
    elif page == "📊 EDA":
        st.subheader("📊 Exploratory Data Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", len(df))
        c2.metric("Unique Products", df["Product_ID"].nunique())
        c3.metric("Categories", df["Product_Category"].nunique())
        c4.metric("Total Demand", df_final["demand"].sum())

        st.markdown("### Category Summary")
        cat_summary = df_final.groupby("Product_Category").agg(
            total_demand=("demand", "sum"),
            avg_price=("Price", "mean"),
            revenue=("Price", lambda x: (x * df_final.loc[x.index, "demand"]).sum())
        ).reset_index()
        st.dataframe(cat_summary)

        st.markdown("### Demand Over Time")
        df_plot = df_final.copy()
        df_plot["ds"] = pd.to_datetime(df_plot["ds"])
        fig = px.line(df_plot, x="ds", y="demand", color="Product_Category",
                      title="Daily Demand Trend")
        st.plotly_chart(fig, width="stretch")

    # =====================================================================
    #                           PAGE: FORECASTING
    # =====================================================================
    elif page == "📈 Forecasting":
        st.subheader("📈 Demand Forecasting")

        df_final = st.session_state.get("df_final", None)
        if df_final is None:
            st.error("Upload data first.")
            st.stop()

        category = st.selectbox("Select Category:", df_final["Product_Category"].unique())
        cat_data = df_final[df_final["Product_Category"] == category]

        fc_data = cat_data.groupby("ds")["demand"].sum().reset_index()
        fc_data["ds"] = pd.to_datetime(fc_data["ds"])
        fc_data = fc_data.set_index("ds").asfreq("D").fillna(0)

        period = st.slider("Forecast Days:", 30, 180, 60)

        # ---------------- Prophet Forecast ----------------
        if PROPHET_AVAILABLE:
            if st.button("Run Prophet Forecast"):
                try:
                    m = Prophet()
                    df_p = fc_data.reset_index().rename(columns={"ds": "ds", "demand": "y"})
                    m.fit(df_p)
                    future = m.make_future_dataframe(periods=period)
                    forecast = m.predict(future)

                    fig = px.line(forecast, x="ds", y="yhat",
                                  title=f"Prophet Forecast - {category}")
                    st.plotly_chart(fig, width="stretch")

                except Exception as e:
                    st.error("Prophet failed — using SARIMAX.")
                    st.exception(e)

        # ---------------- SARIMAX Forecast ----------------
        if st.button("Run SARIMAX Forecast"):
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                model = SARIMAX(fc_data["demand"], order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 7))
                fit = model.fit(disp=False)
                pred = fit.get_forecast(period)
                idx = pd.date_range(fc_data.index[-1] + pd.Timedelta(days=1),
                                    periods=period)
                yhat = pd.Series(pred.predicted_mean.values, index=idx)

                fig = px.line(title=f"SARIMAX Forecast - {category}")
                fig.add_scatter(x=fc_data.index, y=fc_data["demand"],
                                name="Observed")
                fig.add_scatter(x=yhat.index, y=yhat.values, name="Forecast")

                st.plotly_chart(fig, width="stretch")

            except Exception as e:
                st.error("SARIMAX Failed")
                st.exception(e)

        # ---------------- DynamicForecast Plugin ----------------
        forecast_fn, _, _ = _get_dynamicforecast()
        if forecast_fn:
            try:
                df_plugin = forecast_fn(fc_data.reset_index(), period)
                st.markdown("### Plugin Forecast Output")
                st.dataframe(df_plugin)
            except Exception:
                pass

    # =====================================================================
    #                    PAGE: PRICE RECOMMENDATION
    # =====================================================================
    elif page == "💰 Price Recommendation":
        st.subheader("💰 Price Optimization")

        df_final = st.session_state.get("df_final", None)
        if df_final is None:
            st.error("Upload data first.")
            st.stop()

        category = st.selectbox("Select Category:", df_final["Product_Category"].unique())
        cat = df_final[df_final["Product_Category"] == category]

        base_price = cat["Price"].mean()
        base_demand = cat["demand"].mean()
        base_cost = cat["cost_per_unit"].mean()

        st.write(f"Base Price: ₹{base_price:.2f}")
        st.write(f"Base Demand: {base_demand:.2f}")
        st.write(f"Cost per Unit: ₹{base_cost:.2f}")

        elasticity = st.slider("Elasticity β", 0.5, 2.5, 1.2, 0.1)

        pr_min = int(base_price * 0.5)
        pr_max = int(base_price * 1.5)

        price_range = st.slider("Price Range", pr_min, pr_max,
                                (int(base_price * 0.8), int(base_price * 1.2)))

        if st.button("Optimize Price"):
            prices = []
            profits = []
            demands = []

            for p in range(price_range[0], price_range[1] + 1):
                demand_est = base_demand * (base_price / p) ** elasticity
                profit = (p - base_cost) * demand_est

                prices.append(p)
                demands.append(demand_est)
                profits.append(profit)

            result = pd.DataFrame({"Price": prices, "Demand": demands, "Profit": profits})
            best = result.loc[result["Profit"].idxmax()]

            st.success(f"🎯 Best Price: ₹{best['Price']:.2f}")
            st.write(f"Demand: {best['Demand']:.2f}")
            st.write(f"Profit: ₹{best['Profit']:.2f}")

            fig = px.line(result, x="Price", y="Profit", title="Profit Curve")
            fig.add_vline(x=best["Price"], line_color="green", line_dash="dash")
            st.plotly_chart(fig, width="stretch")

    # =====================================================================
    #                        PAGE: SENSITIVITY
    # =====================================================================
    elif page == "🔬 Sensitivity":
        st.subheader("🔬 Sensitivity Analysis")

        df_final = st.session_state.get("df_final", None)
        if df_final is None:
            st.error("Upload data.")
            st.stop()

        category = st.selectbox("Select Category:", df_final["Product_Category"].unique())
        cat = df_final[df_final["Product_Category"] == category]

        base_price = cat["Price"].mean()
        base_demand = cat["demand"].mean()
        base_cost = cat["cost_per_unit"].mean()

        st.write(f"Base Price: ₹{base_price:.2f}")
        st.write(f"Base Demand: {base_demand:.2f}")
        st.write(f"Base Cost: ₹{base_cost:.2f}")

        elasticity = st.slider("Elasticity (β)", 0.5, 2.5, 1.2, 0.1)

        price_min = st.number_input("Min Price", value=int(base_price * 0.5))
        price_max = st.number_input("Max Price", value=int(base_price * 1.5))
        n = st.slider("Grid Points", 20, 200, 80)

        show_scan = st.checkbox("Show Category Scan (Heatmap + Histogram)")

        if st.button("Show Sensitivity Plots"):
            beta = -float(elasticity)
            _, plot_price_sensitivity_fn, plot_category_price_scan_fn = _get_dynamicforecast()

            # --- Price Sensitivity Plots ---
            if plot_price_sensitivity_fn:
                try:
                    dfig, rfig, pfig = plot_price_sensitivity_fn(
                        Q0=base_demand,
                        P0=base_price,
                        betas=beta,
                        price_min=price_min,
                        price_max=price_max,
                        n=n,
                        cost=base_cost
                    )

                    st.plotly_chart(dfig, width="stretch")
                    st.plotly_chart(rfig, width="stretch")
                    st.plotly_chart(pfig, width="stretch")

                except Exception as e:
                    st.error("Sensitivity plot failed.")
                    st.exception(e)

            # --- Category Scan ---
            if show_scan and plot_category_price_scan_fn:
                try:
                    scat, hist = plot_category_price_scan_fn(df_final, category, betas=beta)

                    if scat:
                        st.plotly_chart(scat, width="stretch")
                    if hist:
                        st.plotly_chart(hist, width="stretch")

                except Exception as e:
                    st.error("Category scan failed.")
                    st.exception(e)
