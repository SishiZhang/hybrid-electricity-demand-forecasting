from src.data_process import run_data_pipeline
from src.eda_plot import run_eda
from src.prophet_stage1 import run_stage1
from src.sarima_stage2 import run_stage2
from src.xgboost_stage3 import run_stage3
from src.hybrid_final import run_hybrid

if __name__ == "__main__":
    print("=====开始完整混合模型预测流程=====")
    # 1.数据预处理
    full_dataset, train_data, test_data = run_data_pipeline()
    # 2.EDA绘图
    run_eda(full_dataset)
    # 3.Stage1 Prophet趋势提取
    train_trend_df, test_trend_df, sarima_train_df, sarima_test_df = run_stage1(train_data, test_data)
    # 4.Stage2 SARIMA季节建模
    sarima_model, sarima_fitted, residual2 = run_stage2(sarima_train_df, sarima_test_df)
    # 5.Stage3 XGBoost残差拟合
    X_lag, y_resid, xgb_model, xgb_pred_residual = run_stage3(residual2)
    # 6.混合模型融合+最终评估
    hybrid_pred, true_y, final_rmse, final_mape = run_hybrid(train_trend_df, sarima_fitted, xgb_pred_residual, train_data)
    print("\n====全部流程执行完毕，图片输出至output/figures====")
