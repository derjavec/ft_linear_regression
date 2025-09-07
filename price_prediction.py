import sys
import pandas as pd
import numpy as np



def scale(array):
    """
    Scales a NumPy array to the [0, 1] range.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array - min_val) / (max_val - min_val)
    return scaled, min_val, max_val

def descale(scaled_array, min_val, max_val):
    """
    Reverts a scaled array back to its original range.
    """
    return scaled_array * (max_val - min_val) + min_val



def get_mileage():
    """
    Prompts the user to input car mileage as a float.
    """
    while True:
        mileage = input("Enter the car mileage to estimate its price: ")

        try:
            mileage = float(mileage)
            break
        except ValueError:
                print("Please enter a valid number")
    return mileage


def get_data():
    """
    Reads CSV data and returns it as a DataFrame.
    """

    path = "data/data.csv"
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("Dataset is empty")
    return data


def clean_data(df):
    """
    Clean the dataset for linear regression:
    - Keep only numeric values in mileage and price
    - Drop missing values (NaN)
    - Drop duplicates
    - Ensure data types are float
    """
    required_columns = ['km', 'price']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_data = df[required_columns].copy()
    df_data = df_data.dropna()
    df_data = df_data.drop_duplicates()
    return df_data
    

def predict_values(intercept, coef, true):
    """
    Predicts Y given intercept and coefficient.
    """
    try:
        intercept = float(intercept)
        coef = float(coef)
        pred = intercept + coef * true
    except : 
        raise ValueError("incorrect value")
  
    return pred


def calculate_error(true, pred):

    if true.shape != pred.shape:
        raise ValueError("data length is not equal")
    e = pred - true
    
    return e
    

def gradient_descent(X, alpha, e, intercept, coef):
    """
    Performs one step of gradient descent for linear regression.
    """

    d_intercept = np.mean(e)
    d_coef = np.mean(e * X)

    intercept -= alpha * d_intercept
    coef -= alpha * d_coef
    return intercept, coef


def train_model(X, Y, alpha, iterations):
    """
    Trains a simple linear regression model using gradient descent.
    """
    X_array = np.array(X, dtype=float)
    Y_array = np.array(Y, dtype=float)

    if X_array.shape != Y_array.shape:
        raise ValueError("data length is not equal")
    intercept, coef = 0.0, 0.0

    epsilon = 1e-6
    MSE_old = float('inf')

    scaled_X_array, X_min, X_max = scale(X_array)
    scaled_Y_array, Y_min, Y_max = scale(Y_array)
    for i in range(iterations):
        y = predict_values(intercept, coef, scaled_X_array)
        e = calculate_error(scaled_Y_array, y)
        MSE = np.mean(e**2) / 2
        if abs(MSE_old - MSE) < epsilon:
            print(f"Converged after {i} iterations")
            break
        MSE_old = MSE
        intercept, coef = gradient_descent(scaled_X_array, alpha, e, intercept, coef)

    return intercept, coef, X_min, X_max, Y_min, Y_max



def predict_price(mileage, df, alpha, iterations):
    """
    Predict price for a given mileage.
    """

    df = clean_data(df)
    km = df['km'].values
    price = df['price'].values
    intercept, coef, X_min, X_max, Y_min, Y_max = train_model(km, price, alpha, iterations)
    mileage_scaled = (mileage - X_min) / (X_max - X_min)
    y_scaled = intercept + coef * mileage_scaled
    final_prediction = descale(y_scaled, Y_min, Y_max)
    return final_prediction


def evaluate_model(df):
    km_values = np.arange(0, 1000000, 20000)
    predicted_prices = [predict_price(km, df, alpha, iterations) for km in km_values]
    df_pred = pd.DataFrame({
        "km": km_values,
        "predicted_price": predicted_prices
    })
    output_file = "predicted_prices.csv"
    df_pred.to_csv(output_file, index=False)


def main():
    mileage = get_mileage()
    df = get_data()
    predicted_price = predict_price(mileage, df)

    evaluate_model(df)
    
    print(f"The predicted price for {mileage } mileage is: {predicted_price}")


if __name__ == "__main__":
    main()