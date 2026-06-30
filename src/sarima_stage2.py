import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import SARIMA_ORDER, SARIMA_SEASONAL_ORDER
from utils import save_figure

def adf_test(resid_series):
    res = adfuller(resid_series.dropna())
    print("\n====ADF平稳性检验====")
    print(f"ADF Statistic: {res[0]:.3f}")
    print(f"p-value: {res[1]:.3f}")
    print("Stationary" if res[1] < 0.05 else "Non-stationary")
    return res

def plot_residual_series(sarima_train_df):
    # 原始残差时序
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(sarima_train_df['ds'], sarima_train_df['y_seasonal'], color='skyblue')
    ax1.set_title('Residual Series from Prophet (Training Set)')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Residual (y - trend)')
    ax1.grid(True)
    save_figure(fig1, "prophet_residual_ts")
    # 滚动均值标准差
    df = sarima_train_df.copy()
    df['roll_mean'] = df['y_seasonal'].rolling(12).mean()
    df['roll_sd'] = df['y_seasonal'].rolling(12).std()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df['ds'], df['y_seasonal'], label='Residual', alpha=0.6)
    ax2.plot(df['ds'], df['roll_mean'], label='Rolling Mean', linestyle='--')
    ax2.plot(df['ds'], df['roll_sd'], label='Rolling SD', linestyle=':')
    ax2.set_title('ADF Visual Inspection: Residuals, Rolling Mean & SD')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Residual')
    ax2.legend()
    ax2.grid(True)
    save_figure(fig2, "residual_rolling_mean_sd")

def train_sarima(sarima_train_df):
    model = SARIMAX(sarima_train_df['y_seasonal'], order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
    result = model.fit(disp=False)
    fitted_vals = result.fittedvalues
    # 二阶残差 = 一阶残差 - SARIMA拟合季节分量
    residual2 = sarima_train_df['y_seasonal'] - fitted_vals
    return result, fitted_vals, residual2

def run_stage2(sarima_train_df, sarima_test_df):
    # ADF检验+绘图
    adf_test(sarima_train_df['y_seasonal'])
    plot_residual_series(sarima_train_df)
    # SARIMA训练
    sarima_res, sarima_fit, resid2 = train_sarima(sarima_train_df)
    print("====SARIMA模型训练完成====")
    return sarima_res, sarima_fit, resid2

if __name__ == "__main__":
    from src.data_process import run_data_pipeline
    from src.prophet_stage1 import run_stage1
    _, train, test = run_data_pipeline()
    _, _, train_sar, _ = run_stage1(train, test)
    run_stage2(train_sar, None)
