import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from utils import save_figure

def hybrid_predict(train_trend_df, sarima_fit, xgb_pred_resid, train_df):
    # 三部分相加：趋势+季节+XGB残差
    trend_arr = train_trend_df['trend'].values
    sarima_arr = sarima_fit.values[:len(trend_arr)]
    # 对齐长度
    min_len = min(len(trend_arr), len(sarima_arr), len(xgb_pred_resid))
    final_pred = trend_arr[:min_len] + sarima_arr[:min_len] + xgb_pred_resid[:min_len]
    true_y = train_df['y'].iloc[-min_len:].values
    # 计算最终指标
    final_rmse = np.sqrt(mean_squared_error(true_y, final_pred))
    final_mape = mean_absolute_percentage_error(true_y, final_pred)
    print("====Final Hybrid Model 最终指标====")
    print(f"Final RMSE: {final_rmse:.2f}, MAPE: {final_mape:.4f}")
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(true_y)), true_y, label='Actual')
    ax.plot(range(len(final_pred)), final_pred, label='Hybrid Prediction')
    ax.set_title('Final Hybrid Model on Training Set')
    ax.legend()
    ax.grid(True)
    save_figure(fig, "hybrid_final_fit")
    return final_pred, true_y, final_rmse, final_mape

def run_hybrid(train_trend_df, sarima_fit, xgb_pred_resid, train_df):
    pred, true, rmse, mape = hybrid_predict(train_trend_df, sarima_fit, xgb_pred_resid, train_df)
    return pred, true, rmse, mape

if __name__ == "__main__":
    from src.data_process import run_data_pipeline
    from src.prophet_stage1 import run_stage1
    from src.sarima_stage2 import run_stage2
    from src.xgboost_stage3 import run_stage3
    full_data, train, test = run_data_pipeline()
    train_trend, test_trend, train_sar, test_sar = run_stage1(train, test)
    sarima_res, sarima_fit, resid2 = run_stage2(train_sar, test_sar)
    X, y, xgb_model, xgb_pred = run_stage3(resid2)
    run_hybrid(train_trend, sarima_fit, xgb_pred, train)
