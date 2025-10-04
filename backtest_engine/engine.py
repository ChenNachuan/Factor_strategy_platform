# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Filename: engine.py (升级版)
# Description: 向量化回测引擎，调用升级后的PerformanceAnalyzer进行业绩分析。
# Author: Quant (Your Mentor)
# Date: 2025-10-05
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
from backtest_engine.performance import PerformanceAnalyzer


class BacktestEngine:
    # ... (类的代码和之前一样，此处省略以保持简洁)
    def __init__(self, master_data, factor_data, n_groups=5, fee=0.000):
        self.master_data = master_data
        self.factor_data = factor_data
        self.n_groups = n_groups
        self.fee = fee
        self.portfolio_returns = None
        print("Backtest Engine initialized.")

    def run(self):
        print("\n[Backtest] Starting vectorized backtest...")
        self.master_data['next_day_return'] = self.master_data.groupby(level='stock_code')['adj_close'].transform(
            lambda x: x.pct_change().shift(-1)
        )
        combined_data = pd.merge(self.factor_data, self.master_data[['next_day_return']], on=['date', 'stock_code'])
        combined_data.dropna(subset=[self.factor_data.columns[0], 'next_day_return'], inplace=True)
        print(" -> Step 1: Data alignment and merge complete.")
        factor_name = self.factor_data.columns[0]
        combined_data['group'] = combined_data.groupby('date')[factor_name].transform(
            lambda x: pd.qcut(x, self.n_groups, labels=False, duplicates='drop') + 1
        )
        print(f" -> Step 2: Stocks grouped into {self.n_groups} portfolios based on '{factor_name}'.")
        group_returns = combined_data.groupby(['date', 'group'])['next_day_return'].mean().unstack()
        group_returns.columns = [f'Group_{int(g)}' for g in group_returns.columns]
        print(" -> Step 3: Daily group returns calculated.")
        long_portfolio = group_returns['Group_1']
        short_portfolio = group_returns[f'Group_{self.n_groups}']
        group_returns['Long_Short'] = long_portfolio - short_portfolio
        group_returns['Long_Only'] = long_portfolio
        print(" -> Step 4: Long-Short portfolio constructed.")
        self.portfolio_returns = group_returns
        print("[Backtest] Backtest execution finished!")
        return self.portfolio_returns


if __name__ == '__main__':
    import sys

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from data_manager.loader import load_and_clean_data
    from factor_library.fundamental.size_factor import SizeFactor

    DATA_DIR_PATH = os.path.join(project_root, "data_manager", "DemoData")

    if os.path.exists(DATA_DIR_PATH):
        master_data, _ = load_and_clean_data(DATA_DIR_PATH)

        if master_data is not None and not master_data.empty:
            size_calculator = SizeFactor(master_data)
            size_factor = size_calculator.calculate_factor()

            backtest = BacktestEngine(master_data, size_factor, n_groups=5)
            portfolio_returns = backtest.run()

            if portfolio_returns is not None:
                # ******** 这是核心修改：将更多数据传入分析器 ********
                # 实例化业绩分析器时，传入因子数据和主数据
                analyzer = PerformanceAnalyzer(portfolio_returns, size_factor, master_data)
                analyzer.calculate_metrics()
                analyzer.plot_results()
                # ***************************************************

    else:
        print(f"错误: 主程序无法找到数据文件夹 '{DATA_DIR_PATH}'。")