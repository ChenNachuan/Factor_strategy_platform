# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Filename: download_to_csv_with_debug.py
# Description: 下载样例数据并保存为CSV格式，增加了详细的错误诊断信息。
# Author: Gemini AI
# Date: 2025-10-02
# -----------------------------------------------------------------------------

import pandas as pd
from WindPy import w
import os
import datetime
from functools import reduce


def download_mini_sample_data(start_date="2024-01-01", end_date="2024-06-30", num_stocks=200):  # 默认改为50
    """
    下载一个小型的、指定股票数量和时间范围的A股数据样例。
    """
    # --- 1. 初始化与连接 ---
    print("=" * 50)
    print("开始执行数据下载任务 (输出格式: CSV, 带诊断)...")
    print("=" * 50)

    if not w.isconnected():
        print("正在连接到Wind...")
        w.start()

    if not w.isconnected():
        print("错误: Wind连接失败，请检查Wind终端是否已打开并登录。")
        return

    print("Wind连接成功！")

    # --- 2. 定义下载参数 ---
    output_dir = f"sample_data_{start_date}_to_{end_date}_{num_stocks}stocks_csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建文件夹: '{output_dir}'")

    # --- 3. 获取股票列表 ---
    first_trade_day_obj = w.tdaysoffset(0, start_date, "").Data[0][0]
    first_trade_day_str = first_trade_day_obj.strftime('%Y%m%d')
    print(f"\n[步骤 1/4] 正在获取 {first_trade_day_str} 的前 {num_stocks} 只A股列表...")

    all_a_shares_data = w.wset("sectorconstituent", f"date={first_trade_day_str};sectorid=a001010100000000")
    if all_a_shares_data.ErrorCode != 0:
        print(f"错误: 获取股票列表失败: {all_a_shares_data.Data}")
        return
    stock_codes = all_a_shares_data.Data[1][:num_stocks]
    print(f"成功获取 {len(stock_codes)} 只A股代码作为样例。")

    # --- 4. 下载日频行情数据 ---
    print("\n[步骤 2/4] 正在下载日频行情数据...")
    market_fields = ["open","high", "low", "close", "adjfactor", "volume", "amt", "mkt_cap_total", "mkt_cap_ard", "turn"]
    market_data_path = os.path.join(output_dir, "daily_market_data.csv")

    # [逻辑同上]...
    all_fields_data = []
    for field in market_fields:
        print(f"  -> 正在下载指标: {field}...")
        wsd_data = w.wsd(stock_codes, field, start_date, end_date, "priceAdj=U;fill=Previous")
        if wsd_data.ErrorCode == 0:
            df = pd.DataFrame(wsd_data.Data, index=wsd_data.Codes, columns=wsd_data.Times).T
            df_melted = df.reset_index().melt(id_vars='index', var_name='stock_code', value_name=field)
            df_melted.rename(columns={'index': 'date'}, inplace=True)
            all_fields_data.append(df_melted)
        else:
            print(f"  -> 警告: 下载指标 {field} 失败: {wsd_data.Data}")  # 增加了行情下载的错误提示
            continue

    if not all_fields_data:
        print("错误: 未能下载到任何行情数据，任务终止。")
        return

    final_market_df = reduce(lambda left, right: pd.merge(left, right, on=['date', 'stock_code'], how='outer'),
                             all_fields_data)
    final_market_df.to_csv(market_data_path, index=False, encoding='utf-8-sig')
    print(f"日频行情数据已成功保存到: {market_data_path}")

    # --- 5. 下载财务与基础数据 ---
    print("\n[步骤 3/4] 正在下载财务与基础数据...")
    financial_fields = ["roe_avg","roa2","catoassets","assetstoequity","current"]
    report_dates = ["2023-12-31", "2024-03-31"]
    financial_data_list = []

    # ******** 这是修改的核心逻辑 ********
    for report_date in report_dates:
        report_date_str = report_date.replace('-', '')
        print(f"  -> 正在尝试获取报告期 {report_date} 的财务数据...")

        wss_data = w.wss(stock_codes, financial_fields, f"unit=1;rptDate={report_date_str}")

        if wss_data.ErrorCode == 0:
            print(f"  -> 成功获取报告期 {report_date} 的数据。")
            df = pd.DataFrame(wss_data.Data, index=wss_data.Fields, columns=wss_data.Codes).T
            df['report_date'] = pd.to_datetime(report_date)
            financial_data_list.append(df)
        else:
            # 明确打印出错误信息
            print(f"  -> 警告: 获取报告期 {report_date} 的财务数据失败。Wind返回错误: {wss_data.Data}")

    if financial_data_list:
        final_financial_df = pd.concat(financial_data_list)
        final_financial_df.index.name = 'stock_code'
        financial_data_path = os.path.join(output_dir, "quarterly_financial_data.csv")
        final_financial_df.to_csv(financial_data_path, encoding='utf-8-sig')
        print(f"财务数据已成功保存到: {financial_data_path}")
    else:
        # 如果列表为空，明确告知用户
        print("  -> 未能下载到任何有效的财务数据，将跳过生成 quarterly_financial_data.csv 文件。")
    # **********************************

    # --- 6. 下载指数行情数据 ---
    print("\n[步骤 4/4] 正在下载指数行情数据...")
    index_codes = ["000300.SH", "000905.SH"]
    index_fields = ["open", "high", "low", "close", "volume", "amt"]
    index_data_path = os.path.join(output_dir, "daily_index_data.csv")

    # ******** 这是修改的核心逻辑 ********
    # 同样采用循环方式下载指数数据
    all_index_fields_data = []
    for field in index_fields:
        print(f"  -> 正在下载指数指标: {field}...")
        wsd_index_data = w.wsd(index_codes, field, start_date, end_date, "")
        if wsd_index_data.ErrorCode == 0:
            df = pd.DataFrame(wsd_index_data.Data, index=wsd_index_data.Codes, columns=wsd_index_data.Times).T
            df_melted = df.reset_index().melt(id_vars='index', var_name='index_code', value_name=field)
            df_melted.rename(columns={'index': 'date'}, inplace=True)
            all_index_fields_data.append(df_melted)
        else:
            print(f"  -> 警告: 下载指数指标 {field} 失败: {wsd_index_data.Data}")
            continue

    if all_index_fields_data:
        final_index_df = reduce(lambda left, right: pd.merge(left, right, on=['date', 'index_code'], how='outer'),
                                all_index_fields_data)
        final_index_df.to_csv(index_data_path, index=False, encoding='utf-8-sig')
        print(f"指数行情数据已成功保存到: {index_data_path}")
    else:
        print("  -> 未能下载到任何有效的指数数据。")

    print("\n" + "=" * 50)
    print(f"所有样例数据下载完毕！请将 '{output_dir}' 文件夹压缩后发给队友。")
    print("=" * 50)


if __name__ == '__main__':
    download_mini_sample_data()