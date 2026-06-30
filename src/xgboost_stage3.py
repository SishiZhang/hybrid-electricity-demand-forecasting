import pandas as pd
from xgboost import XGBRegressor
from config import LAG_LIST

def build_lag_features(resid2_series):
    X = pd.DataFrame()
    for lag in LAG_LIST:
        X[f"lag{lag}"] = resid2_series.shift(lag)
    y = resid2_series
    X = X.dropna()
    y = y.loc[X.index]
    return X, y

def train_xgb(X, y):
    xgb_model = XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X, y)
    pred_resid = xgb_model.predict(X)
    return xgb_model, pred_resid

def run_stage3(resid2):
    X, y = build_lag_features(resid2)
    xgb_model, pred_residual = train_xgb(X, y)
    print("====XGBoost残差模型训练完成====")
    return X, y, xgb_model, pred_residual

if __name__ == "__main__":
    from src.data_process import run_data_pipeline
    from src.prophet_stage1 import run_stage1
    from src.sarima_stage2 import run_stage2
    _, train, test = run_data_pipeline()
    _, _, train_sar, _ = run_stage1(train, test)
    _, _, resid2 = run_stage2(train_sar, None)
    run_stage3(resid2)
