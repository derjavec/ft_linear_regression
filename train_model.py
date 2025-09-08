import numpy as np
from utils.treating_data import get_data, clean_data
from utils.scaling import scale, descale


def predict_values(intercept: float, coef: float, X: np.ndarray) -> np.ndarray:
    """
    Predict the output values for given inputs using a linear model.
    """
    try:
        intercept = float(intercept)
        coef = float(coef)
        return intercept + coef * X
    except Exception:
        raise ValueError("Invalid intercept or coefficient")


def calculate_error(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Calculate the error between true and predicted values.
    """
    if true.shape != pred.shape:
        raise ValueError("Data length mismatch between true and predicted arrays")
    return pred - true


def gradient_descent(X: np.ndarray, alpha: float, error: np.ndarray, intercept: float, coef: float):
    """
    Perform one step of gradient descent for linear regression.
    """
    d_intercept = np.mean(error)
    d_coef = np.mean(error * X)

    intercept -= alpha * d_intercept
    coef -= alpha * d_coef
    return intercept, coef


def train_model(X: np.ndarray, Y: np.ndarray, alpha: float, iterations: int) :
    """
    Train a linear regression model using gradient descent.
    """
    X_array = np.array(X, dtype=float)
    Y_array = np.array(Y, dtype=float)

    if X_array.shape != Y_array.shape:
        raise ValueError("Data length mismatch between X and Y")

    intercept, coef = 0.0, 0.0
    epsilon = 1e-6
    MSE_old = float('inf')

    scaled_X_array, X_min, X_max = scale(X_array)
    scaled_Y_array, Y_min, Y_max = scale(Y_array)

    for i in range(iterations):
        y = predict_values(intercept, coef, scaled_X_array)
        error = calculate_error(scaled_Y_array, y)
        MSE = np.mean(error ** 2) / 2
        if abs(MSE_old - MSE) < epsilon:
            break
        MSE_old = MSE
        intercept, coef = gradient_descent(scaled_X_array, alpha, error, intercept, coef)

    return intercept, coef, X_min, X_max, Y_min, Y_max, i + 1


def get_coeficients(df, alpha: float, iterations: int):
    """
    Get the trained coefficients (intercept and slope) for the data.

    Automatically increases iterations if convergence is not reached.
    """

    km = df['km'].values
    price = df['price'].values

    max_iterations = 50000
    while True:
        intercept, coef, X_min, X_max, Y_min, Y_max, i = train_model(km, price, alpha, iterations)
        if i < iterations or iterations >= max_iterations:
            break
        iterations *= 2

    return intercept, coef, X_min, X_max, Y_min, Y_max


def main() -> None:
    """
    Main script to train a linear regression model and display coefficients.
    """
    df = get_data()
    df = clean_data(df)

    alpha = 0.1
    iterations = 2000

    intercept, coef, X_min, X_max, Y_min, Y_max = get_coeficients(df, alpha, iterations)
    print(f"Intercept: {intercept}, First coefficient: {coef}")


if __name__ == "__main__":
    main()
