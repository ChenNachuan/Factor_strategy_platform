# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Filename: engine.py (完整优化版)
# Description: 向量化回测引擎，支持灵活的多头方向选择，并调用升级后的
#              PerformanceAnalyzer进行全面的业绩与因子分析。
# Author: Quant (Your Mentor)
# Date: 2025-10-05
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import sys

# 动态添加项目根目录到Python路径，以便导入其他模块
# 假设engine.py位于 a_project_folder/backtest_engine/ 目录下
try:
    from backtest_engine.performance import PerformanceAnalyzer
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from backtest_engine.performance import PerformanceAnalyzer


class BacktestEngine:
    """
    向量化回测引擎 (修改版)
    增加了对多头方向的灵活选择，以适应不同类型的因子。
    """

    def __init__(self, master_data, factor_data, n_groups=5, fee=0.000, long_direction='high'):
        """
        初始化回测引擎。

        Args:
            master_data (pd.DataFrame): 包含价格和复权因子的主数据。
            factor_data (pd.DataFrame): 包含因子值的DataFrame。
            n_groups (int): 分组数量，默认为5。
            fee (float): 交易手续费，默认为0。
            long_direction (str): 多头方向。
                                'high'表示做多因子值最高的组 (适用于正向因子，如动量)。
                                'low'表示做多因子值最低的组 (适用于负向因子，如市盈率)。
        """
        self.master_data = master_data
        self.factor_data = factor_data
        self.n_groups = n_groups
        self.fee = fee
        self.portfolio_returns = None

        if long_direction not in ['high', 'low']:
            raise ValueError("参数 long_direction 必须是 'high' 或 'low'。")
        self.long_direction = long_direction

        print(f"Backtest Engine initialized. Long direction set to: '{self.long_direction}'")

    def run(self):
        """
        执行向量化回测。
        """
        print("\n[Backtest] Starting vectorized backtest...")

        # --- 步骤 1: 数据对齐与合并 ---
        # 为主数据计算未来一日收益率，用于回测
        self.master_data['next_day_return'] = self.master_data.groupby(level='stock_code')['adj_close'].transform(
            lambda x: x.pct_change().shift(-1)
        )
        # 将因子数据与未来收益率数据合并
        factor_name = self.factor_data.columns[0]
        combined_data = pd.merge(self.factor_data, self.master_data[['next_day_return']], on=['date', 'stock_code'])
        combined_data.dropna(subset=[factor_name, 'next_day_return'], inplace=True)
        print(" -> Step 1: Data alignment and merge complete.")

        # --- 步骤 2: 股票分组 ---
        # 根据每日因子值，将股票分为 n 组
        combined_data['group'] = combined_data.groupby('date')[factor_name].transform(
            lambda x: pd.qcut(x, self.n_groups, labels=False, duplicates='drop') + 1
        )
        print(f" -> Step 2: Stocks grouped into {self.n_groups} portfolios based on '{factor_name}'.")

        # --- 步骤 3: 计算分组每日收益率 ---
        # 假设组内等权，计算每个组的平均日收益
        group_returns = combined_data.groupby(['date', 'group'])['next_day_return'].mean().unstack()
        group_returns.columns = [f'Group_{int(g)}' for g in group_returns.columns]
        print(" -> Step 3: Daily group returns calculated.")

        # --- 步骤 4: 构建多空与多头组合 ---
        # 根据 long_direction 参数，灵活构建投资组合
        top_group_name = f'Group_{self.n_groups}'  # 因子值最高的组
        bottom_group_name = 'Group_1'  # 因子值最低的组

        if self.long_direction == 'high':
            # 做多因子值最高的组，做空因子值最低的组
            long_portfolio = group_returns[top_group_name]
            short_portfolio = group_returns[bottom_group_name]
            print(f" -> Strategy: Long {top_group_name}, Short {bottom_group_name}")
        else:  # self.long_direction == 'low'
            # 做多因子值最低的组，做空因子值最高的组
            long_portfolio = group_returns[bottom_group_name]
            short_portfolio = group_returns[top_group_name]
            print(f" -> Strategy: Long {bottom_group_name}, Short {top_group_name}")

        group_returns['Long_Short'] = long_portfolio - short_portfolio
        group_returns['Long_Only'] = long_portfolio
        print(" -> Step 4: Long-Short portfolio constructed correctly.")

        # 考虑交易成本（此处为简化版，可根据需要扩展）
        if self.fee > 0:
            # 假设每日调仓，对多头和多空组合收取手续费
            group_returns['Long_Only'] = group_returns['Long_Only'] - self.fee
            group_returns['Long_Short'] = group_returns['Long_Short'] - (self.fee * 2)  # 多头和空头双边成本

        self.portfolio_returns = group_returns
        print("[Backtest] Backtest execution finished!")
        return self.portfolio_returns


if __name__ == '__main__':
    # 这是一个完整的演示流程，展示如何端到端地运行回测

    # 动态导入项目中的其他模块
    from data_manager.loader import load_and_clean_data
    from factor_library.fundamental.size_factor import SizeFactor

    # 构建数据文件夹的绝对路径
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR_PATH = os.path.join(project_root_path, "data_manager", "DemoData")

    if os.path.exists(DATA_DIR_PATH):
        # 1. 加载并清洗数据
        master_data, _ = load_and_clean_data(DATA_DIR_PATH)

        if master_data is not None and not master_data.empty:
            # 2. 计算因子
            size_calculator = SizeFactor(master_data)
            size_factor = size_calculator.calculate_factor()

            # 3. 运行回测
            # 关键：根据你的因子假设，设置 long_direction
            # 对于之前的市值因子，我们发现其IC为正，说明市值越大收益越高，
            # 因此我们应该做多市值最高的组（Group 5），设置 long_direction='high'
            backtest = BacktestEngine(master_data, size_factor, n_groups=5, long_direction='high')
            portfolio_returns = backtest.run()

            # 4. 分析并可视化结果
            if portfolio_returns is not None:
                # 将所有需要的数据传入分析器
                analyzer = PerformanceAnalyzer(portfolio_returns, size_factor, master_data)
                analyzer.calculate_metrics()
                analyzer.plot_results()
    else:
        print(f"错误: 主程序无法找到数据文件夹 '{DATA_DIR_PATH}'。")