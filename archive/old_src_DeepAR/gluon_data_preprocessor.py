from data_cleaning import row_cleaning, column_cleaning, generate_stock_code_list
from dataset_generator import generate_datasets_per_stock
import pandas as pd
import os
import json
import argparse
# start 22:11 大概三分鐘
def preprocess_gluon_data(data: pd.DataFrame, standardize: bool = False) -> dict:
    """
    Preprocess the input DataFrame for GluonTS dataset generation.

    Parameters:
    - data: pd.DataFrame - The input data containing stock information.
    - standardize: bool - Whether to standardize the columns.

    Returns:
    - dict: A dictionary containing datasets for each stock code.
    """
    # Clean the data
    stock_codes = generate_stock_code_list(data)
    cleaned_data = pd.DataFrame()
    for code in stock_codes:
        stock_data = data[data['code'] == code].copy()
        stock_data = row_cleaning(stock_data)
        cleaned_data = pd.concat([cleaned_data, stock_data], ignore_index=True)
    cleaned_data = column_cleaning(cleaned_data)

    # Generate datasets store direcotory if not exists
    # check if the output directory exists
    os.makedirs("gluonTS_Dataset/input", exist_ok=True)
    os.makedirs("gluonTS_Dataset/static_dict", exist_ok=True)


    # Generate stock code list
    stock_codes = generate_stock_code_list(cleaned_data)

    # Generate datasets for each stock code
    datasets = {}
    for code in stock_codes:
        dataset_json = generate_datasets_per_stock(cleaned_data, code, standardize)
        # json dumps to file
        datasets[code] = json.loads(dataset_json)
        with open(f"gluonTS_Dataset/input/stock_{code}_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(datasets[code], f, ensure_ascii=False, indent=4)

    return datasets

def main(input_path: str, standardize: bool = False):
    data = pd.read_csv(input_path)
    datasets = preprocess_gluon_data(data, standardize)
    return datasets

if __name__ == "__main__":
    # receive arguments
    parser = argparse.ArgumentParser(description="Preprocess GluonTS data")
    parser.add_argument("input_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--standardize", action="store_true", help="Whether to standardize the columns")
    args = parser.parse_args()

    main(args.input_path, args.standardize)