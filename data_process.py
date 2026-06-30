import os
import pandas as pd
from functools import reduce
from config import RAW_DATA_DIR, FILL_ZERO_COLS, FILL_MEAN_COLS, TRAIN_CUT, TEST_END
from utils import create_dir

def read_all_excel():
    """读取全部原始Excel文件"""
    file_list = [
        "EV_registration(monthly).xlsx",
        "GasConsumption(monthly).xlsx",
        "GSP(Monthly).xlsx",
        "Humidity新(monthly).xlsx",
        "NewenergyDevice(monthly).xlsx",
        "Population(monthly).xlsx",
        "Rainfall(monthly).xlsx",
        "Sunshine(monthly).xlsx",
        "Temperature(monthly).xlsx",
        "Totaldemand(monthly).xlsx"
    ]
    df_dict = {}
    for fname in file_list:
        path = os.path.join(RAW_DATA_DIR, fname)
        df_dict[fname] = pd.read_excel(path)
    ev = df_dict["EV_registration(monthly).xlsx"]
    gas = df_dict["GasConsumption(monthly).xlsx"]
    gsp = df_dict["GSP(Monthly).xlsx"]
    humidity = df_dict["Humidity新(monthly).xlsx"]
    device = df_dict["NewenergyDevice(monthly).xlsx"]
    population = df_dict["Population(monthly).xlsx"]
    rain = df_dict["Rainfall(monthly).xlsx"]
    sunshine = df_dict["Sunshine(monthly).xlsx"]
    temp = df_dict["Temperature(monthly).xlsx"]
    demand = df_dict["Totaldemand(monthly).xlsx"]
    return demand, temp, sunshine, rain, population, device, humidity, gsp, gas, ev

def format_month(df):
    """统一Month列为%Y-%m字符串"""
    if pd.api.types.is_integer_dtype(df['Month']):
        df['Month'] = pd.to_datetime(df['Month'], origin='1899-12-30', unit='D')
    elif not pd.api.types.is_datetime64_any_dtype(df['Month']):
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    df['Month'] = df['Month'].dt.strftime('%Y-%m')
    return df

def merge_all_data(df_list):
    """按Month左连接合并所有表"""
    def merge_monthly(left, right):
        return pd.merge(left, right, on='Month', how='left')
    dataset = reduce(merge_monthly, df_list)
    return dataset

def check_data_problem(df):
    """检查非数值、缺失值"""
    issues = []
    for col in df.columns:
        if col == "Month":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append((col, "非数值类型"))
            continue
        miss_cnt = df[col].isna().sum()
        if miss_cnt > 0:
            issues.append((col, f"缺失值{miss_cnt}个"))
    for col, msg in issues:
        print(f"{col}: {msg}")
    return issues

def fill_missing_data(df):
    """缺失值填充"""
    for col in FILL_ZERO_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    for col in FILL_MEAN_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
    return df

def split_train_test(df):
    """生成ds、y，划分训练测试集"""
    df['ds'] = pd.to_datetime(df['Month'] + '-01')
    df['y'] = df['Total_Demand(MWh)']
    train_data = df.iloc[:TRAIN_CUT][['ds', 'y']].copy()
    test_data = df.iloc[TRAIN_CUT:TEST_END][['ds', 'y']].copy()
    return df, train_data, test_data

def run_data_pipeline():
    # 1.读取数据
    demand, temp, sunshine, rain, population, device, humidity, gsp, gas, ev = read_all_excel()
    all_dfs = [demand, temp, sunshine, rain, population, device, humidity, gsp, gas, ev]
    # 2.时间标准化
    new_dfs = [format_month(d) for d in all_dfs]
    # 3.合并
    full_dataset = merge_all_data(new_dfs)
    # 4.数据校验
    check_data_problem(full_dataset)
    # 5.填充缺失值
    full_dataset = fill_missing_data(full_dataset)
    # 6.划分训练测试
    full_dataset, train_df, test_df = split_train_test(full_dataset)
    print("====数据预处理完成====")
    print(f"训练集长度：{len(train_df)}，测试集长度：{len(test_df)}")
    return full_dataset, train_df, test_df

if __name__ == "__main__":
    run_data_pipeline()
