"""
Plotting utilities for time series forecasting visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Set


def create_forecast_plot(series: pd.DataFrame, results: list, future_df: pd.DataFrame = None,
                        show_only_test: bool = True, split_ds = None,
                        hide_non_chosen_models: bool = True, best_name: str = None,
                        visible_models: Optional[Set[str]] = None,
                        visible_test_models: Optional[Set[str]] = None,
                        visible_pred_models: Optional[Set[str]] = None,
                        trend_series: pd.DataFrame = None):
    """
    Create a comprehensive forecast plot with actuals and model predictions.
    
    Args:
        visible_models: Legacy parameter for backward compatibility (will be ignored if visible_test_models/visible_pred_models provided)
        visible_test_models: Models to show test performance for
        visible_pred_models: Models to show predictions for
    """
    fig, ax = plt.subplots(figsize=(20, 4.5))
    
    # Handle backward compatibility
    if visible_test_models is None and visible_pred_models is None and visible_models is not None:
        visible_test_models = visible_models
        visible_pred_models = visible_models
    elif visible_test_models is None:
        visible_test_models = set()
    elif visible_pred_models is None:
        visible_pred_models = set()

    # Ensure series ds is datetime normalized
    visible_series = series.copy()
    try:
        if not pd.api.types.is_datetime64_any_dtype(visible_series["ds"]):
            visible_series["ds"] = pd.to_datetime(visible_series["ds"], errors="coerce")
        visible_series["ds"] = visible_series["ds"].dt.normalize()
    except Exception:
        pass
    
    # Check if there are any future predictions to determine if we need connection points
    has_any_future = (future_df is not None and not future_df.empty) or any(
        (res.get("future_df") is not None and not res.get("future_df").empty) 
        for res in results
    )
    
    if show_only_test and split_ds is not None:
        test_data = series[series["ds"] >= split_ds]
        # Only include the last training point if there are future predictions to connect to
        if has_any_future:
            last_train_point = series[series["ds"] < split_ds].tail(1)
            if not last_train_point.empty:
                visible_series = pd.concat([last_train_point, test_data], ignore_index=True)
            else:
                visible_series = test_data
        else:
            # No future predictions, so no need for connection points
            visible_series = test_data

    # Calculate fixed y-axis limits based on actuals and visible model predictions
    # This ensures consistent scaling regardless of which models are shown
    y_min, y_max = visible_series["y"].min(), visible_series["y"].max()
    # Track the latest date that is actually visible so we can cap the trend line
    try:
        x_max_date = pd.to_datetime(visible_series["ds"], errors="coerce").max()
    except Exception:
        x_max_date = None

    # Consider forecasts and future predictions for visible models
    for res in results:
        name = res.get("name")
        # Include if visible for test or prediction
        if name not in visible_test_models and name not in visible_pred_models:
            continue
        fcast = res.get("forecast_df")
        if fcast is not None and not fcast.empty:
            y_min = min(y_min, fcast["yhat"].min())
            y_max = max(y_max, fcast["yhat"].max())
            try:
                x_max_date = max(x_max_date, pd.to_datetime(fcast["ds"], errors="coerce").max()) if x_max_date is not None else pd.to_datetime(fcast["ds"], errors="coerce").max()
            except Exception:
                pass
        fdf = res.get("future_df")
        if fdf is not None and not fdf.empty:
            try:
                if not pd.api.types.is_datetime64_any_dtype(fdf["ds"]):
                    fdf["ds"] = pd.to_datetime(fdf["ds"], errors="coerce")
                fdf["ds"] = fdf["ds"].dt.normalize()
            except Exception:
                pass
            y_min = min(y_min, fdf["yhat"].min())
            y_max = max(y_max, fdf["yhat"].max())
            try:
                x_max_date = max(x_max_date, pd.to_datetime(fdf["ds"], errors="coerce").max()) if x_max_date is not None else pd.to_datetime(fdf["ds"], errors="coerce").max()
            except Exception:
                pass
    # Also consider single future_df if passed explicitly
    if future_df is not None and not future_df.empty:
        y_min = min(y_min, future_df["yhat"].min())
        y_max = max(y_max, future_df["yhat"].max())

    # Add some padding to the y-axis limits (5% on each side)
    y_range = y_max - y_min
    y_padding = y_range * 0.05
    y_min -= y_padding
    y_max += y_padding

    # Build a stable color mapping so the same model keeps the same color
    try:
        default_colors = plt.rcParams.get('axes.prop_cycle').by_key().get('color', [])
    except Exception:
        default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Determine which model names will be plotted (respect visibility/hide flags)
    names_in_plot = []
    for res in results:
        name = res.get("name")
        # Include if visible for test or prediction
        if name not in visible_test_models and name not in visible_pred_models:
            continue
        names_in_plot.append(name)

    name_to_color = {}
    if default_colors:
        for idx, name in enumerate(names_in_plot):
            name_to_color[name] = default_colors[idx % len(default_colors)]

    # Plot actual values
    ax.plot(visible_series["ds"], visible_series["y"], linewidth=2, alpha=0.85,
            color="#222", label="Actual")

    # Plot each model's test forecast
    for res in results:
        name = res["name"]
        # Only show test performance if model is visible for test
        if name not in visible_test_models:
            continue

        forecast_df = res["forecast_df"]
        if forecast_df.empty:
            continue
        try:
            if not pd.api.types.is_datetime64_any_dtype(forecast_df["ds"]):
                forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], errors="coerce")
            forecast_df["ds"] = forecast_df["ds"].dt.normalize()
        except Exception:
            pass

        label = res['name']

        # Only create connection point if we have actual forecast data to connect to
        # and if there are future predictions that would benefit from the connection
        if has_any_future and not forecast_df.empty:
            # Use visible_series to ensure consistent connection points
            last_visible_for_forecast = visible_series.sort_values("ds").tail(1)
            if not last_visible_for_forecast.empty:
                # Add connection point at the exact same date as the last known point
                connection_point = pd.DataFrame({
                    "ds": [last_visible_for_forecast["ds"].iloc[0]],
                    "yhat": [last_visible_for_forecast["y"].iloc[0]]
                })
                plot_df = pd.concat([connection_point, forecast_df], ignore_index=True)
            else:
                plot_df = forecast_df
        else:
            # No future predictions or no forecast data, so no need for connection points
            plot_df = forecast_df

        ax.plot(
            plot_df["ds"], plot_df["yhat"],
            linestyle="--",
            linewidth=1.8,
            color=name_to_color.get(label, None),
            label=label,
        )

    # Overlay simple fitted trend line if provided
    try:
        if trend_series is not None and isinstance(trend_series, pd.DataFrame) and not trend_series.empty:
            ts = trend_series.copy()
            if not pd.api.types.is_datetime64_any_dtype(ts["ds"]):
                ts["ds"] = pd.to_datetime(ts["ds"], errors="coerce")
            ts["ds"] = ts["ds"].dt.normalize()
            # When show_only_test and split_ds provided, only show trend from split onward
            if show_only_test and split_ds is not None:
                ts = ts[ts["ds"] >= pd.to_datetime(split_ds)].reset_index(drop=True)
            # Cap trend at the latest actually visible date (actuals or predictions)
            try:
                if x_max_date is not None:
                    ts = ts[ts["ds"] <= pd.to_datetime(x_max_date)].reset_index(drop=True)
            except Exception:
                pass
            ax.plot(ts["ds"], ts["trend"], color="#555", linewidth=2.0, alpha=0.7, label="Trend")
    except Exception:
        pass

    # Plot future predictions. Prefer per-model forecasts if present; otherwise use single future_df
    has_per_model_future = any((res.get("future_df") is not None and not res.get("future_df").empty) for res in results)
    if has_per_model_future:
        last_known_actual = (
            visible_series.dropna(subset=["y"]).sort_values("ds").tail(1)
        )
        for res in results:
            name = res["name"]
            # Only show predictions if model is visible for predictions
            if name not in visible_pred_models:
                continue
            fdf = res.get("future_df")
            if fdf is None or fdf.empty:
                continue
            try:
                if not pd.api.types.is_datetime64_any_dtype(fdf["ds"]):
                    fdf["ds"] = pd.to_datetime(fdf["ds"], errors="coerce")
                fdf["ds"] = fdf["ds"].dt.normalize()
            except Exception:
                pass

            if not last_known_actual.empty:
                connection_point = pd.DataFrame({
                    "ds": [last_known_actual["ds"].iloc[0]],
                    "yhat": [last_known_actual["y"].iloc[0]]
                })
                f_plot = pd.concat([connection_point, fdf], ignore_index=True)
            else:
                f_plot = fdf

            is_best = (best_name is not None and res["name"] == best_name)
            ax.plot(
                f_plot["ds"], f_plot["yhat"],
                linestyle="-",
                linewidth=(2.2 if is_best else 1.8),
                color=name_to_color.get(res["name"], None),
                label=None,  # avoid duplicate legend entries; keep legend from forecast lines
            )
    elif future_df is not None and not future_df.empty:
        # Ensure prediction line starts from the last known NON-NaN actual to eliminate any gap
        last_known_actual = (
            visible_series.dropna(subset=["y"]).sort_values("ds").tail(1)
        )
        if not last_known_actual.empty:
            # Add a connection point at the exact same date as the last known point
            connection_point = pd.DataFrame({
                "ds": [last_known_actual["ds"].iloc[0]],
                "yhat": [last_known_actual["y"].iloc[0]]
            })
            # Combine connection point with future predictions
            future_plot_df = pd.concat([connection_point, future_df], ignore_index=True)
        else:
            future_plot_df = future_df

        # Use best model name in legend instead of generic "Prediction"
        # Use the best model's color for the shared future line (if known), keep legend from forecast
        best_color = name_to_color.get(best_name, None)
        ax.plot(
            future_plot_df["ds"], future_plot_df["yhat"],
            linestyle="-", linewidth=2.2, color=best_color, label=None
        )

    # Add vertical line for train/test split
    if split_ds is not None and not show_only_test:
        ax.axvline(split_ds, linestyle=":", linewidth=1, alpha=0.7)

    # Set fixed y-axis limits to maintain consistent scale
    ax.set_ylim(y_min, y_max)

    # Styling
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 0.8, 1))

    return fig





