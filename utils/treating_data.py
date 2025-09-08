import pandas as pd


def clean_data(df: pd.DataFrame,) ->  pd.DataFrame:
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


def get_data() -> pd.DataFrame:
    """
    Reads CSV data and returns it as a DataFrame.
    """

    path = "data/data.csv"
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("Dataset is empty")
    return data

