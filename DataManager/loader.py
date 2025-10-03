# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Filename: loader.py
# Description: 根据《因子投资：方法与实践》的数据处理原则，加载并清洗由
#              SampleDownload.py 生成的CSV数据。
# Author: Quant (Your Mentor)
# Date: 2025-10-03
# -----------------------------------------------------------------------------

import pandas as pd
import os

# 将 financial_fields 定义在函数外部，使其成为全局变量
financial_fields = ["roe_avg", "roa2", "catoassets", "assetstoequity", "current"]


def load_and_clean_data(data_dir):
    """
    加载并清洗所有样例数据。

    Args:
        data_dir (str): 存放CSV文件的文件夹路径。

    Returns:
        pandas.DataFrame: 清洗和合并后的主数据框。
        pandas.DataFrame: 清洗后的指数数据框。
    """
    print("=" * 50)
    print("开始执行数据加载与清洗任务...")
    print("=" * 50)

    # --- 1. 定义文件路径 ---
    market_data_path = os.path.join(data_dir, "daily_market_data.csv")
    financial_data_path = os.path.join(data_dir, "quarterly_financial_data.csv")
    index_data_path = os.path.join(data_dir, "daily_index_data.csv")

    # --- 2. 加载数据 ---
    print("\n[步骤 1/4] 正在加载CSV文件...")
    try:
        market_df = pd.read_csv(market_data_path, parse_dates=['date'])

        if os.path.exists(financial_data_path):
            financial_df = pd.read_csv(financial_data_path, parse_dates=['report_date'])
        else:
            financial_df = None

        index_df = pd.read_csv(index_data_path, parse_dates=['date'])
        print("所有CSV文件加载成功！")
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}。请先运行 SampleDownload.py 生成数据。")
        return None, None

    # --- 3. 数据清洗与预处理 ---
    print("\n[步骤 2/4] 正在进行数据预处理...")

    market_df['adj_close'] = market_df['close'] * market_df['adjfactor']
    market_df['adj_open'] = market_df['open'] * market_df['adjfactor']
    market_df.set_index(['date', 'stock_code'], inplace=True)
    print(" -> 行情数据处理完成，已计算后复权价。")

    if financial_df is not None:
        financial_df.set_index(['report_date', 'stock_code'], inplace=True)
        financial_df.sort_index(inplace=True)
        print(" -> 财务数据处理完成。")
    else:
        print(" -> 警告：未找到财务数据文件，将跳过财务数据相关处理。")

    index_df.set_index(['date', 'index_code'], inplace=True)
    print(" -> 指数数据处理完成。")

    # --- 4. 合并数据 ---
    print("\n[步骤 3/4] 正在合并行情与财务数据...")
    market_df_reset = market_df.reset_index()

    if financial_df is not None:
        financial_df_reset = financial_df.reset_index()
        merged_df = pd.merge_asof(
            market_df_reset,
            financial_df_reset,
            left_on='date',
            right_on='report_date',
            by='stock_code',
            direction='backward'
        )
        print("行情与财务数据合并成功！")
    else:
        merged_df = market_df_reset
        print("仅使用行情数据，未进行财务数据合并。")

    # --- 5. 最终清洗 ---
    print("\n[步骤 4/4] 正在进行最终清洗...")

    # ******** 这是核心修改 ********
    # 这里的逻辑是基于股票在【样本内】出现的天数，而非真实的上市天数。
    # 由于我们的样本只有6个月（约180天），>252的过滤条件会移除所有数据。
    # 在真实研究中，应获取真实的IPO日期来计算上市天数。
    # 为了当前学习的顺利进行，我们暂时注释掉这一行。
    # merged_df['days_on_market'] = merged_df.groupby('stock_code')['date'].transform(lambda x: (x - x.min()).dt.days)
    # initial_rows = len(merged_df)
    # merged_df = merged_df[merged_df['days_on_market'] > 252]
    # print(f" -> 已剔除上市不足一年的次新股，移除 {initial_rows - len(merged_df)} 条记录。")
    # **************************

    if financial_df is not None:
        def winsorize(series, lower_quantile=0.01, upper_quantile=0.99):
            lower_bound = series.quantile(lower_quantile)
            upper_bound = series.quantile(upper_quantile)
            return series.clip(lower=lower_bound, upper=upper_bound)

        for col in financial_fields:
            if col in merged_df.columns:
                merged_df[col] = merged_df.groupby('date')[col].transform(winsorize)
        print(" -> 已对财务数据进行缩尾处理。")

    merged_df.set_index(['date', 'stock_code'], inplace=True)

    print("\n" + "=" * 50)
    print("数据加载与清洗任务完成！")
    print("=" * 50)

    return merged_df, index_df


if __name__ == '__main__':
    SAMPLE_DATA_DIR = "DemoData"

    if os.path.exists(SAMPLE_DATA_DIR):
        master_data, index_data = load_and_clean_data(SAMPLE_DATA_DIR)

        if master_data is not None:
            print("\n清洗后的股票主数据预览:")
            print(master_data.head())
            print(f"\n数据形状: {master_data.shape}")

        if index_data is not None:
            print("\n清洗后的指数数据预览:")
            print(index_data.head())
            print(f"\n数据形状: {index_data.shape}")
    else:
        print(f"错误: 未找到数据文件夹 '{SAMPLE_DATA_DIR}'。")
        print("请先运行 SampleDownload.py 来下载样例数据。")