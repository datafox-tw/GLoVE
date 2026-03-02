import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocess.data_cleaning import row_cleaning, generate_stock_code_list
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Configuration
CSV_PATH = "Dataset/data/ml_dataset_alpha101_volatility.csv"
PREDICTION_LENGTH = 5


def load_garch_data(csv_path):
    """
    載入並清洗 GARCH 資料
    輸出欄位: [code, date, var_true_90, garch_var_90]
    """
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


def evaluate_garch(final_df):
    """
    評估 GARCH 預測結果
    對每個股票，取最後 PREDICTION_LENGTH 天的資料進行評估
    """
    print(f"\nEvaluating GARCH predictions...")
    print(f"Prediction length: {PREDICTION_LENGTH} days")
    
    results = []
    garch_mses = []
    garch_maes = []
    
    # 按股票代碼分組
    for code, group in final_df.groupby('code'):
        # 確保按日期排序
        group = group.sort_values('date').reset_index(drop=True)
        
        # 檢查資料長度是否足夠
        if len(group) < PREDICTION_LENGTH:
            print(f"Warning: Stock {code} has insufficient data. Need {PREDICTION_LENGTH}, got {len(group)}. Skipping.")
            continue
        
        # 取最後 PREDICTION_LENGTH 天作為測試集
        test_data = group.iloc[-PREDICTION_LENGTH:].copy()
        
        truth = test_data['var_true_90'].values
        garch_pred = test_data['garch_var_90'].values
        
        # 檢查 NaN 值
        if pd.isna(truth).any():
            print(f"Warning: NaN values found in truth for stock {code}. Skipping.")
            continue
        
        if pd.isna(garch_pred).any():
            print(f"Warning: NaN values found in GARCH predictions for stock {code}. Skipping.")
            continue
        
        # 計算指標
        garch_mse = mean_squared_error(truth, garch_pred)
        garch_mae = mean_absolute_error(truth, garch_pred)
        
        garch_mses.append(garch_mse)
        garch_maes.append(garch_mae)
        
        results.append({
            "code": code,
            "garch_mse": garch_mse,
            "garch_mae": garch_mae,
        })
    
    return results, garch_mses, garch_maes


def main():
    # 1. 載入 GARCH 資料
    final_df = load_garch_data(CSV_PATH)
    print(f"Loaded data shape: {final_df.shape}")
    print(f"Number of unique stocks: {final_df['code'].nunique()}")
    
    # 2. 評估 GARCH
    results, garch_mses, garch_maes = evaluate_garch(final_df)
    
    # 3. 生成結果 DataFrame
    results_df = pd.DataFrame(results)
    
    # 4. 輸出摘要
    print("\n" + "="*50)
    print("GARCH Evaluation Summary")
    print("="*50)
    print(f"Total Stocks Evaluated: {len(results_df)}")
    
    if len(garch_mses) > 0:
        print(f"Average GARCH MSE: {np.mean(garch_mses):.8f}")
        print(f"Average GARCH MAE: {np.mean(garch_maes):.8f}")
        print(f"Median GARCH MSE: {np.median(garch_mses):.8f}")
        print(f"Median GARCH MAE: {np.median(garch_maes):.8f}")
        print(f"Min GARCH MSE: {np.min(garch_mses):.8f}")
        print(f"Max GARCH MSE: {np.max(garch_mses):.8f}")
    
    # 5. 儲存結果
    output_path = "garch_evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")
    
    return results_df


if __name__ == "__main__":
    main()

