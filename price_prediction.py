import sys
import pandas as pd
import numpy as np

from evaluate_model import evaluate_model
from train_model import get_coeficients
from utils.scaling import scale, descale
from utils.treating_data import get_data, clean_data


def get_mileage() -> float:
    """
    Prompt the user to enter a car mileage and return it as a float.
    Continuously asks for input until the user provides a valid number.
    """
    while True:
        mileage = input("Enter the car mileage to estimate its price: ")

        try:
            mileage = float(mileage)
            break
        except ValueError:
            print("Please enter a valid number")
    return mileage


def predict_price(mileage: float, df: pd.DataFrame, alpha: float, iterations: int) -> float:
    """
    Predict the car price for a given mileage using a trained linear regression model.
    """
    intercept, coef, X_min, X_max, Y_min, Y_max = get_coeficients(df, alpha, iterations)
    mileage_scaled = (mileage - X_min) / (X_max - X_min)
    y_scaled = intercept + coef * mileage_scaled
    final_prediction = descale(y_scaled, Y_min, Y_max)
    return final_prediction


def main() -> None:
    """
    Main function for car price prediction.

    Workflow:
    1. Ask the user for the car mileage.
    2. Load historical car data from CSV.
    3. Evaluate models for different alpha values and find the best alpha.
    4. Predict the price for the user's mileage using the best alpha.
    5. Display the predicted price.
    """
    mileage = get_mileage()

    df = get_data()
    df = clean_data(df)

    alpha = evaluate_model(df)

    iterations = 2000
    predicted_price = predict_price(mileage, df, alpha, iterations)

    print(f"The predicted price for {mileage} km mileage is: {predicted_price:.2f}")


if __name__ == "__main__":
    main()
