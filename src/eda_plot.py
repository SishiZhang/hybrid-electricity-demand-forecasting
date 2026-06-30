import math
import seaborn as sns
import matplotlib.pyplot as plt
from utils import save_figure

def plot_all_variable_timeseries(dataset):
    # 全变量大图
    df_long = dataset.drop(columns=['ds', 'y']).melt(id_vars='Month', var_name='Variable', value_name='Value')
    df_long['Month'] = pd.to_datetime(df_long['Month'] + '-01')
    g = sns.FacetGrid(df_long, col='Variable', col_wrap=4, sharey=False, height=3.5, aspect=1.8)
    g.map_dataframe(sns.lineplot, x='Month', y='Value', color='skyblue')
    g.set_titles("{col_name}")
    g.set_axis_labels("Time", "Value")
    g.fig.suptitle("Monthly Time Series Trends of All Variables", fontsize=16)
    plt.subplots_adjust(top=0.92)
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    save_figure(g.fig, "all_variables_highres")

def plot_batch_timeseries(dataset):
    # 分批输出子图
    df_long = dataset.drop(columns=['ds', 'y']).melt(id_vars='Month', var_name='Variable', value_name='Value')
    df_long['Month'] = pd.to_datetime(df_long['Month'] + '-01')
    variables = df_long['Variable'].unique().tolist()
    batch_size = 3
    n_batches = math.ceil(len(variables) / batch_size)
    for i in range(n_batches):
        batch_vars = variables[i * batch_size:(i + 1) * batch_size]
        sub_df = df_long[df_long['Variable'].isin(batch_vars)]
        g = sns.FacetGrid(sub_df, col='Variable', col_wrap=3, sharey=False, height=5, aspect=1.8)
        g.map_dataframe(sns.lineplot, x='Month', y='Value', color='skyblue')
        g.set_titles("{col_name}", size=13)
        g.set_axis_labels("Time", "Value")
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
        plt.subplots_adjust(top=0.88)
        save_figure(g.fig, f"variables_batch_{i+1}")

def run_eda(dataset):
    plot_all_variable_timeseries(dataset)
    plot_batch_timeseries(dataset)
    print("EDA时序绘图全部完成，图片保存在output/figures")

if __name__ == "__main__":
    from src.data_process import run_data_pipeline
    full_data, _, _ = run_data_pipeline()
    run_eda(full_data)
