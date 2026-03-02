# GLoVE: GARCH-Loss Based Volatility Forecasting Experiment - Reproduction Guide

This document provides instructions on how to reproduce the experimental results for the GLoVE project.

## How to reenact the result and further comparison

### 0. Set up Python virtual environment
``` 
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Download newest dataset from **Releases**
- unzip and rename the folder to "Dataset"
- it is on https://github.com/hochienH/GLoVE -releases right now

### 2. Run following code to complete data preprocessing
```
python src/preprocess.py --input Dataset/ml_dataset_alpha101_volatility.csv --output Dataset/clean.pkl --disabled_features close log_return u_hat_90 gjrgarch_var_90 tgarch_var_90 --use_log_target --target_col var_true_90 --garch_col garch_var_90 
```
- `--use_log_target`: 將target_col及garch_col數據取ln(1+p)，後續Eval Model時設定需與此相同
- Final data is in Dataset/clean.pkl

### 3. Build dataset
```
python src/dataset_builder.py --input Dataset/clean.pkl --output Dataset/ts_data.pkl --val_frac 0.2 --test_frac 0.1 --input_chunk_length 90 --static_mode ticker --target_col var_true_90 --garch_col garch_var_90 --scale_target zscore
```
- `--scale_target`: 預設為`none`，可調整為`mean`, `zscore`，對target_col及garch_col標準化，後續Eval Model時設定需與此相同

### 4. Execute deep learning scripts
腳本會自動化進行以下步驟:
1. 以optuna搜尋最佳超參數
2. 對各種lambda分別訓練模型及預測
3. 將各模型結果整合並繪製成圖表

#### TSMixer
```
bash tsmixer_pipeline.sh --parallel 4 --use_log_target --covariate_mode alpha --invert_train_scale zscore
```
#### LSTM
```
bash lstm_pipeline.sh --parallel 4 --use_log_target --covariate_mode alpha --invert_train_scale zscore
```
- `--parallel`: 平行化幾個
- `--use_log_target`: 若Step 2.處理資料時有進行log轉換，則需要在此轉換回來
- `--covariate_mode`: 預設為`none`，不使用alpha資料；設定為`alpha`會使用alpha資料
- `--invert_train_scale`: 若Step 3.處理資料時有進行標準化，則需要在此返還波動度為真實值
- 其餘參數可在.sh檔調整

## Resources & Links
- **Presentation**: [Google Slides](https://docs.google.com/presentation/d/1RK-kOrt1dCIzXsmUMu1cpMOTFbq2gh3wWMnMSdsICV0/edit?usp=sharing)
- **Introduction Video**: [YouTube](https://www.youtube.com/watch?v=AhvbGsd1O98)
- **Trained Model and Experimental Results**: [Google Drive](https://drive.google.com/drive/u/0/folders/1xbHQe9j2vbTLrdpKLcS7qy-QH8qNLHXI)
