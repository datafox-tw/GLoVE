import json
import glob
import os
import pandas as pd
import numpy as np
from pathlib import Path
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocess.data_cleaning import row_cleaning, generate_stock_code_list
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Configuration
INPUT_FOLDER = "./gluonTS_Dataset/input/"
MODEL_DIR = "./trained_model"
CSV_PATH = "Dataset/data/ml_dataset_alpha101_volatility.csv"
PREDICTION_LENGTH = 5
CONTEXT_LENGTH = 30
FREQ = "D"

def load_garch_data(csv_path):
    print(f"Loading GARCH data from {csv_path}...")
    # Load full data to ensure cleaning matches training data processing
    df = pd.read_csv(csv_path)
    
    print("Applying data cleaning (matching training preprocessing)...")
    stock_codes = generate_stock_code_list(df)
    cleaned_data = pd.DataFrame()
    
    for code in stock_codes:
        stock_data = df[df['code'] == code].copy()
        stock_data = row_cleaning(stock_data)
        cleaned_data = pd.concat([cleaned_data, stock_data], ignore_index=True)
        
    # Select only necessary columns after cleaning
    final_df = cleaned_data[['code', 'date', 'var_true_90', 'garch_var_90']].copy()
    final_df['date'] = pd.to_datetime(final_df['date'])
    return final_df

def load_test_data_with_codes(json_folder, prediction_length, freq):
    file_paths = glob.glob(os.path.join(json_folder, "*.json"))
    file_paths.sort()
    
    test_list = []
    
    print(f"Found {len(file_paths)} json files.")
    
    for path in file_paths:
        # Extract code from filename: stock_2330_dataset.json -> 2330
        filename = os.path.basename(path)
        try:
            code = filename.split('_')[1]
        except IndexError:
            continue
        
        with open(path, 'r') as f:
            data = json.load(f)

        if "instances" in data:
            iterator = data["instances"]
        else:
            iterator = [data]
            
        for inst in iterator:
            target = inst["target"]
            start = inst["start"]
            statics = inst["feat_static_cat"]
            dynamics = np.array(inst["feat_dynamic_real"])
            
            # Fix Data Leakage (Same as train_deepar.py)
            shifted_dynamics = np.zeros_like(dynamics)
            shifted_dynamics[:, 1:] = dynamics[:, :-1]
            shifted_dynamics[:, 0] = 0
            dynamics = shifted_dynamics
            
            total_len = len(target)
            min_length = CONTEXT_LENGTH + prediction_length
            if total_len <= min_length:
                continue
                
            # We only care about the test portion (full sequence)
            test_list.append({
                "start": start,
                "target": target,
                "feat_static_cat": statics,
                "feat_dynamic_real": dynamics.tolist(),
                "item_id": code
            })
            
    return ListDataset(test_list, freq=freq)

def main():
    # 1. Load GARCH Data
    final_df = load_garch_data(CSV_PATH)
    
    # 2. Load Model
    print(f"Loading model from {MODEL_DIR}...")
    predictor = Predictor.deserialize(Path(MODEL_DIR))
    
    # 3. Load Test Data
    print("Loading test data...")
    test_ds = load_test_data_with_codes(INPUT_FOLDER, PREDICTION_LENGTH, FREQ)
    
    # 4. Predict
    print("Generating predictions...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    # 5. Compare
    deepar_mses = []
    garch_mses = []
    deepar_maes = []
    garch_maes = []
    
    print("\nComparing DeepAR vs GARCH...")
    
    results = []

    for i, (forecast, ts) in enumerate(zip(forecasts, tss)):
        code = forecast.item_id
        
        # DeepAR Prediction (Median)
        deepar_pred = forecast.median
        
        # Get corresponding data from DataFrame
        # Filter by code
        mask = (final_df['code'] == int(code))
        subset = final_df[mask]
        
        # We assume the test set in JSON corresponds to the end of the cleaned dataframe
        # So we take the last PREDICTION_LENGTH rows
        if len(subset) < PREDICTION_LENGTH:
            print(f"Warning: Not enough data for stock {code}. Need {PREDICTION_LENGTH}, got {len(subset)}")
            continue
            
        subset_tail = subset.iloc[-PREDICTION_LENGTH:]
        
        truth = subset_tail['var_true_90'].values
        garch_pred = subset_tail['garch_var_90'].values
        
        # Calculate Metrics
        deepar_mse = mean_squared_error(truth, deepar_pred)
        garch_mse = mean_squared_error(truth, garch_pred)
        
        deepar_mae = mean_absolute_error(truth, deepar_pred)
        garch_mae = mean_absolute_error(truth, garch_pred)
        
        deepar_mses.append(deepar_mse)
        garch_mses.append(garch_mse)
        deepar_maes.append(deepar_mae)
        garch_maes.append(garch_mae)
        
        results.append({
            "code": code,
            "deepar_mse": deepar_mse,
            "garch_mse": garch_mse,
            "deepar_mae": deepar_mae,
            "garch_mae": garch_mae,
            "mse_improvement": garch_mse - deepar_mse # Positive means DeepAR is better
        })

    # 6. Summary
    results_df = pd.DataFrame(results)
    
    print("\n====== Summary ======")
    print(f"Total Stocks Evaluated: {len(results_df)}")
    print(f"Average DeepAR MSE: {np.mean(deepar_mses):.8f}")
    print(f"Average GARCH MSE: {np.mean(garch_mses):.8f}")
    print(f"Average DeepAR MAE: {np.mean(deepar_maes):.8f}")
    print(f"Average GARCH MAE: {np.mean(garch_maes):.8f}")
    
    better_count = (results_df['mse_improvement'] > 0).sum()
    total_count = len(results_df)
    if total_count > 0:
        print(f"DeepAR better than GARCH in {better_count}/{total_count} stocks ({better_count/total_count:.2%})")
    
    # Save results
    results_df.to_csv("comparison_results.csv", index=False)
    print("Detailed results saved to comparison_results.csv")

if __name__ == "__main__":
    main()
