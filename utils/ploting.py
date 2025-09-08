import matplotlib.pyplot as plt
import pandas as pd


def plot_alphas_vs_data(df: pd.DataFrame, df_pred: pd.DataFrame):
    """
    Plot original car price data against predictions obtained with different alpha values.

    Behavior
    --------
    - Plots the original data as a reference line.
    - Adds one line per alpha prediction for comparison.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['km'], df['price'], label = "original data")

    for col in df_pred.columns:
        if col !='km':
            plt.plot(df_pred['km'], df_pred[col], label = col)

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Car Price Predictions for Different Alpha Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_bestalpha_vs_data(df: pd.DataFrame, df_pred: pd.DataFrame):
    """
    Plot original car price data against predictions obtained with the best alpha value.

    Behavior
    --------
    - Plots the original data as a reference line.
    - Plots the prediction line for the best alpha value.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['km'], df['price'], label = "original data")
    plt.plot(df_pred['km'], df_pred['price'], label = "Predictions")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Car Price Predictions for Different Alpha Values")
    plt.legend()
    plt.grid(True)
    plt.show()
