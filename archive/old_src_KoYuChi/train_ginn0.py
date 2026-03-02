# train_ginn0.py
"""
兩種策略共用一份腳本：
1) LSTM baseline: 用 LSTM 直接預測 var_true_90
2) GINN-0: 用 LSTM 學 garch_var_90 (student mimics teacher)，再拿來比對 var_true_90

輸入統一來自 ml_dataset_alpha101_volatility.csv
output: 一個結果 DataFrame，可被 wrapper 讀取
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import time
import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mse, mae
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocess.data_cleaning import row_cleaning, generate_stock_code_list  # 用你們原本的


# ===== 全域設定 =====
CSV_PATH = "Dataset/data/ml_dataset_alpha101_volatility.csv"
PRED_LEN = 5                # 預測 5 天
INPUT_CHUNK = 90            # 使用過去 90 天 window
N_EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_DIM = 64
N_LAYERS = 2
DROPOUT = 0.1
RANDOM_STATE = 42


# ===== 1. 資料前處理：跟 GARCH 一樣的邏輯，但抽離成共用函式 =====
def load_garch_dataset(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    載入 & 清洗資料，輸出欄位:
    [code, date, var_true_90, garch_var_90]
    """
    print(f"[INFO] Loading csv from {csv_path}")
    df = pd.read_csv(csv_path)
    stock_codes = generate_stock_code_list(df)

    cleaned_list = []
    for code in stock_codes:
        stock_data = df[df["code"] == code].copy()
        stock_data = row_cleaning(stock_data)  # 跟你們 GARCH / DeepAR preproc 一致
        cleaned_list.append(stock_data)

    final_df = pd.concat(cleaned_list, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])

    final_df = final_df[["code", "date", "var_true_90", "garch_var_90"]].copy()
    print(f"[INFO] Cleaned dataset shape: {final_df.shape}")
    return final_df

def evaluate_garch_only(
    series_true: Dict[int, TimeSeries],
    series_garch: Dict[int, TimeSeries],
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    GARCH baseline:
    - 不訓練任何模型
    - 直接拿 garch_var_90 的最後 PRED_LEN 跟 var_true_90 當天對比
    回傳 per-stock MSE / MAE
    這裡完全不訓練東西，純粹把 CSV 裡現成的 garch_var_90 當預測值去算 metric。
    
    如果提供了 df，且 TimeSeries 中有 NaN，則從 DataFrame 直接獲取數據（與 eval_garch_only.py 一致）
    """
    results = []
    for code in series_true.keys():
        ts_true = series_true[code]
        ts_garch = series_garch[code]

        if len(ts_true) <= PRED_LEN:
            continue

        ts_true_test = ts_true[-PRED_LEN:]
        ts_garch_test = ts_garch[-PRED_LEN:]

        y_true = ts_true_test.values().flatten()
        y_garch = ts_garch_test.values().flatten()
        
        # 如果 TimeSeries 中有 NaN，且提供了原始 DataFrame，則從 DataFrame 直接獲取
        if (pd.isna(y_true).any() or pd.isna(y_garch).any()) and df is not None:
            # 從 DataFrame 直接獲取數據（與 eval_garch_only.py 一致）
            stock_data = df[df['code'] == code].copy()
            stock_data = stock_data.sort_values('date').reset_index(drop=True)
            
            if len(stock_data) < PRED_LEN:
                continue
            
            test_data = stock_data.iloc[-PRED_LEN:].copy()
            y_true = test_data['var_true_90'].values
            y_garch = test_data['garch_var_90'].values
        
        # 檢查 NaN 值（與 eval_garch_only.py 保持一致）
        if pd.isna(y_true).any():
            print(f"[WARNING] NaN values found in truth for stock code {code}. Skipping.")
            continue
        
        if pd.isna(y_garch).any():
            print(f"[WARNING] NaN values found in GARCH predictions for stock code {code}. Skipping.")
            continue
        
        # 計算指標
        garch_mse_val = mean_squared_error(y_true, y_garch)
        garch_mae_val = mean_absolute_error(y_true, y_garch)

        results.append({
            "code": code,
            "strategy": "garch",
            "mse": garch_mse_val,
            "mae": garch_mae_val,
        })

    return pd.DataFrame(results)

def build_series_per_stock(df: pd.DataFrame) -> Tuple[Dict[int, TimeSeries], Dict[int, TimeSeries]]:
    """
    依照 code 分組，回傳:
    - series_true[code]: var_true_90 的 TimeSeries
    - series_garch[code]: garch_var_90 的 TimeSeries
    """
    series_true = {}
    series_garch = {}

    for code, sub in df.groupby("code"):
        sub = sub.sort_values("date")

        ts_true = TimeSeries.from_dataframe(
            sub, time_col="date", value_cols="var_true_90", fill_missing_dates=True, freq="D"
        )
        ts_garch = TimeSeries.from_dataframe(
            sub, time_col="date", value_cols="garch_var_90", fill_missing_dates=True, freq="D"
        )

        series_true[int(code)] = ts_true
        series_garch[int(code)] = ts_garch

    print(f"[INFO] Built TimeSeries for {len(series_true)} stocks")
    return series_true, series_garch


# ===== 2. 兩種策略的訓練與評估邏輯 =====

def train_lstm_baseline(series_true: Dict[int, TimeSeries]) -> RNNModel:
    """
    LSTM baseline:
    - input / target 都是 var_true_90
    - 多股票一起訓練，多序列 shared model
    """
    all_series: List[TimeSeries] = list(series_true.values())

    # 確保長度足夠
    all_series = [
        s for s in all_series if len(s) > INPUT_CHUNK + PRED_LEN
    ]

    train_series = [s[:-PRED_LEN] for s in all_series]

    model = RNNModel(
        model="LSTM",
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        n_rnn_layers=N_LAYERS,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        random_state=RANDOM_STATE,
    )

    print("[LSTM] Training LSTM baseline on var_true_90 ...")
    model.fit(train_series, verbose=True)
    return model


def train_ginn0(series_garch: Dict[int, TimeSeries]) -> RNNModel:
    """
    GINN-0:
    - input / target 都是 garch_var_90
    - LSTM 學習 GARCH 輸出的型態
    - 評估時再拿預測 vs var_true_90 比
    """
    all_series: List[TimeSeries] = list(series_garch.values())
    all_series = [
        s for s in all_series if len(s) > INPUT_CHUNK + PRED_LEN
    ]
    train_series = [s[:-PRED_LEN] for s in all_series]

    model = RNNModel(
        model="LSTM",
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        n_rnn_layers=N_LAYERS,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        random_state=RANDOM_STATE,
    )

    print("[GINN-0] Training LSTM to mimic garch_var_90 ...")
    model.fit(train_series, verbose=True)
    return model


# ===== 3. 評估 =====

def evaluate_lstm_baseline(
    model: RNNModel,
    series_true: Dict[int, TimeSeries],
) -> pd.DataFrame:
    """
    針對每檔股票:
    - 用 LSTM baseline 預測最後 PRED_LEN 天 var_true_90
    - 和真實值比較

    回傳 per-stock 指標 DataFrame
    """
    results = []
    for code, ts_true in series_true.items():
        if len(ts_true) <= INPUT_CHUNK + PRED_LEN:
            continue

        ts_train = ts_true[:-PRED_LEN]
        ts_test = ts_true[-PRED_LEN:]

        pred = model.predict(PRED_LEN, series=ts_train)

        y_true = ts_test.values().flatten()
        y_pred = pred.values().flatten()

        mse_val = mean_squared_error(y_true, y_pred)
        mae_val = mean_absolute_error(y_true, y_pred)

        results.append({
            "code": code,
            "strategy": "lstm",
            "mse": mse_val,
            "mae": mae_val,
        })

    return pd.DataFrame(results)


def evaluate_ginn0(
    model: RNNModel,
    series_true: Dict[int, TimeSeries],
    series_garch: Dict[int, TimeSeries],
) -> pd.DataFrame:
    """
    針對每檔股票:
    - ts_garch 用來給 model 預測 (input)
    - var_true_90 用來當 ground truth
    - 同時算出 GARCH 自己的表現 & GINN-0 的表現
    """
    results = []
    for code in series_true.keys():
        ts_true = series_true[code]
        ts_garch = series_garch[code]

        if len(ts_true) <= INPUT_CHUNK + PRED_LEN:
            continue

        ts_garch_train = ts_garch[:-PRED_LEN]
        ts_garch_test = ts_garch[-PRED_LEN:]
        ts_true_test = ts_true[-PRED_LEN:]

        # GINN-0 預測
        pred_ginn0 = model.predict(PRED_LEN, series=ts_garch_train)

        y_true = ts_true_test.values().flatten()
        y_garch = ts_garch_test.values().flatten()
        y_ginn0 = pred_ginn0.values().flatten()

        garch_mse_val = mean_squared_error(y_true, y_garch)
        ginn0_mse_val = mean_squared_error(y_true, y_ginn0)
        garch_mae_val = mean_absolute_error(y_true, y_garch)
        ginn0_mae_val = mean_absolute_error(y_true, y_ginn0)

        results.append({
            "code": code,
            "strategy": "ginn0",
            "garch_mse": garch_mse_val,
            "ginn0_mse": ginn0_mse_val,
            "garch_mae": garch_mae_val,
            "ginn0_mae": ginn0_mae_val,
            "mse_improvement": garch_mse_val - ginn0_mse_val,
        })

    return pd.DataFrame(results)


# ===== 4. 對外介面：run_experiment，給 wrapper 用 =====

def run_experiment(
    mode: str,
    csv_path: str = CSV_PATH,
    save_path: str = None,
) -> pd.DataFrame:
    assert mode in {"lstm", "ginn0", "garch"}

    df = load_garch_dataset(csv_path)
    series_true, series_garch = build_series_per_stock(df)
    start_time = time.time()
    if mode == "lstm":
        model = train_lstm_baseline(series_true)
        res_df = evaluate_lstm_baseline(model, series_true)

    elif mode == "ginn0":
        model = train_ginn0(series_garch)
        res_df = evaluate_ginn0(model, series_true, series_garch)

    else:  # mode == "garch"
        res_df = evaluate_garch_only(series_true, series_garch, df=df)
    end_time = time.time()
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(save_p, index=False)
        print(f"[INFO] Saved results to {save_p}")

    # summary
    if mode == "lstm":
        print("=== LSTM baseline summary ===")
        print(f"Total stocks evaluated: {len(res_df)}")
        print(f"Avg MSE: {res_df['mse'].mean():.8f}")
        print(f"Avg MAE: {res_df['mae'].mean():.8f}")

    elif mode == "ginn0":
        print("=== GINN-0 summary ===")
        print(f"Total stocks evaluated: {len(res_df)}")
        print(f"Avg GARCH MSE: {res_df['garch_mse'].mean():.8f}")
        print(f"Avg GINN-0 MSE: {res_df['ginn0_mse'].mean():.8f}")
        better = (res_df["mse_improvement"] > 0).sum()
        total = len(res_df)
        if total > 0:
            print(f"GINN-0 better than GARCH in {better}/{total} stocks ({better/total:.2%})")

    else:  # garch
        print("=== GARCH baseline summary ===")
        print(f"Total stocks evaluated: {len(res_df)}")
        print(f"Avg GARCH MSE: {res_df['mse'].mean():.8f}")
        print(f"Avg GARCH MAE: {res_df['mae'].mean():.8f}")
    print(f"Time usage: {end_time-start_time} seconds")
    return res_df


# ===== 5. CLI 入口 =====

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="ginn0",
        choices=["lstm", "ginn0", "garch"],
        help="Select training strategy: lstm | ginn0",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=CSV_PATH,
        help="Path to ml_dataset_alpha101_volatility.csv",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Where to save result csv (optional)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        mode=args.mode,
        csv_path=args.csv_path,
        save_path=args.save_path,
    )
