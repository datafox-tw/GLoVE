# ginn0_dataset.py
import pandas as pd
from preprocess.data_cleaning import row_cleaning, generate_stock_code_list

CSV_PATH = "Dataset/data/ml_dataset_alpha101_volatility.csv"

def load_garch_dataset(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)
    stock_codes = generate_stock_code_list(df)
    cleaned_data = []

    for code in stock_codes:
        stock_data = df[df["code"] == code].copy()
        stock_data = row_cleaning(stock_data)
        cleaned_data.append(stock_data)

    final_df = pd.concat(cleaned_data, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])

    # 只保留這幾個：之後要用來算 metric
    final_df = final_df[["code", "date", "var_true_90", "garch_var_90"]].copy()
    return final_df
