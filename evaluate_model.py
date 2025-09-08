import numpy as np
import pandas as pd
from train_model import get_coeficients
from utils.scaling import scale, descale
from utils.ploting import plot_alphas_vs_data, plot_bestalpha_vs_data
from utils.treating_data import get_data, clean_data


def predict_price(mileage: float, df: pd.DataFrame, alpha: float, iterations: int) -> float:
    """
    Predict the car price for a given mileage using linear regression parameters.
    """
    intercept, coef, X_min, X_max, Y_min, Y_max = get_coeficients(df, alpha, iterations)
    mileage_scaled = (mileage - X_min) / (X_max - X_min)
    y_scaled = intercept + coef * mileage_scaled
    return descale(y_scaled, Y_min, Y_max)


def generate_predictions(df: pd.DataFrame, alphas: np.ndarray, km_values: np.ndarray, iterations: int) -> pd.DataFrame:
    """
    Generate predicted prices for multiple alpha values and km values.
    Returns a DataFrame with km as first column and one column per alpha.
    """
    df_pred = pd.DataFrame({"km": km_values})
    for alpha in alphas:
        predicted_prices = [predict_price(km, df, alpha, iterations) for km in km_values]
        df_pred[f"prices {alpha:.2f}"] = predicted_prices
    return df_pred


def save_predictions(df_pred: pd.DataFrame, output_file: str = "predicted_prices.csv") -> None:
    """
    Save predicted prices to a CSV file.
    """
    df_pred.to_csv(output_file, index=False)


def find_best_alpha(df: pd.DataFrame, alphas: np.ndarray, iterations: int) -> float:
    """
    Find the alpha that minimizes mean absolute error on the original data.
    """
    results = {}
    for alpha in alphas:
        predicted_prices = np.array([predict_price(km, df, alpha, iterations) for km in df['km']])
        errors = np.abs(df['price'].values - predicted_prices)
        mean_error = errors.mean()
        results[alpha] = mean_error
    best_alpha = min(results, key=results.get)
    return best_alpha


def regression_precision(true, pred):
    """
    Calculate R² (coefficient of determination) as a precision score for regression.
    """
    true = np.array(true)
    pred = np.array(pred)

    mean = true.mean()

    num = np.sum((true - pred)**2)
    denom = np.sum((true - mean)**2)

    precission_score = 1 - num/denom
    return precission_score



def evaluate_model(df: pd.DataFrame, iterations: int = 2000) -> float:
    """
    Full evaluation pipeline:
    - Generate predictions for multiple alphas
    - Save CSV
    - Plot all alphas vs data
    - Find best alpha
    - Plot best alpha vs original data
    Returns the best alpha.
    """
    km_values = np.arange(0, 250000, 20000)
    alphas = np.arange(0.01, 0.1, 0.011)

    print("Generating predictions for different alpha values, wait please...")
    df_pred_all = generate_predictions(df, alphas, km_values, iterations)
    save_predictions(df_pred_all)
    plot_alphas_vs_data(df, df_pred_all)

    print("Looking for the best alpha, wait please...")
    best_alpha = find_best_alpha(df, alphas, iterations)

    prices_best_alpha = [predict_price(km, df, best_alpha, iterations) for km in df['km']]
    df_pred_best = pd.DataFrame({"km": df['km'], "price": prices_best_alpha})
    plot_bestalpha_vs_data(df, df_pred_best)
    
    print(f"Best alpha: {best_alpha:.2f}")

    precission_score = regression_precision(df['price'], df_pred_best['price'])
    print(f"precission score is {precission_score}")
 
    return best_alpha


def main() -> None:
    """
    Main function for car price prediction.

    Workflow:
    1. Load historical car data from CSV.
    2. Evaluate models for different alpha values and find the best alpha.
    """
    df = get_data()
    df = clean_data(df)

    iterations = 2000

    evaluate_model(df, iterations)


if __name__ == "__main__":
    main()
