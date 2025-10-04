# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Filename: engine.py
# Description: 实现一个基于投资组合排序法的向量化回测引擎。
# Author: Quant (Your Mentor)
# Date: 2025-10-04
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class BacktestEngine:
    """
    向量化单因子回测引擎。

    核心回测逻辑严格遵循《因子投资：方法与实践》第2.1节介绍的投资组合排序法。
    """

    def __init__(self, master_data, factor_data, n_groups=5, fee=0.000):
        """
        初始化回测引擎。

        Args:
            master_data (pd.DataFrame): 包含每日行情数据的主数据框。
            factor_data (pd.DataFrame): 包含因子值的DataFrame。
            n_groups (int): 投资组合的分组数量，默认为5（五分位）。
            fee (float): 单边交易成本费率，默认为0。
        """
        self.master_data = master_data
        self.factor_data = factor_data
        self.n_groups = n_groups
        self.fee = fee
        self.portfolio_returns = None
        print("Backtest Engine initialized.")

    def run(self):
        """
        执行回测流程。
        """
        print("\n[Backtest] Starting vectorized backtest...")

        # 1. Align and merge data
        self.master_data['next_day_return'] = self.master_data.groupby(level='stock_code')['adj_close'].transform(
            lambda x: x.pct_change().shift(-1)
        )

        combined_data = pd.merge(self.factor_data, self.master_data[['next_day_return']], on=['date', 'stock_code'])
        combined_data.dropna(subset=[self.factor_data.columns[0], 'next_day_return'], inplace=True)
        print(" -> Step 1: Data alignment and merge complete.")

        # 2. Portfolio Grouping
        factor_name = self.factor_data.columns[0]
        combined_data['group'] = combined_data.groupby('date')[factor_name].transform(
            lambda x: pd.qcut(x, self.n_groups, labels=False, duplicates='drop') + 1
        )
        print(f" -> Step 2: Stocks grouped into {self.n_groups} portfolios based on '{factor_name}'.")

        # 3. Calculate daily returns for each group (equal weight)
        group_returns = combined_data.groupby(['date', 'group'])['next_day_return'].mean().unstack()
        group_returns.columns = [f'Group_{int(g)}' for g in group_returns.columns]
        print(" -> Step 3: Daily group returns calculated.")

        # 4. Construct Long-Short Portfolio
        # For the size factor, a smaller factor value (log market cap) is better.
        # So we long Group 1 (smallest) and short Group N (largest).
        long_portfolio = group_returns['Group_1']
        short_portfolio = group_returns[f'Group_{self.n_groups}']

        group_returns['Long_Short'] = long_portfolio - short_portfolio
        group_returns['Long_Only'] = long_portfolio
        print(" -> Step 4: Long-Short portfolio constructed.")

        self.portfolio_returns = group_returns
        print("[Backtest] Backtest execution finished!")

    def performance_analysis(self):
        """
        Calculate and display key performance indicators.
        """
        if self.portfolio_returns is None:
            print("Error: Please run the .run() method first to perform a backtest.")
            return

        print("\n[Performance Analysis] Calculating key performance metrics...")

        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        days = len(cumulative_returns)
        annualized_return = cumulative_returns.iloc[-1] ** (252 / days) - 1
        annualized_volatility = self.portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        performance_metrics = pd.DataFrame({
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        })

        print("Performance Metrics:")
        print(performance_metrics.round(4))

        self.cumulative_returns = cumulative_returns
        return performance_metrics

    def plot_results(self):
        """
        Plot the cumulative return curves.
        """
        if not hasattr(self, 'cumulative_returns'):
            print("Error: Please run .performance_analysis() first to calculate metrics.")
            return

        print("\n[Visualization] Plotting equity curves...")

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        # ******** 这是核心修改 ********
        # 将图表标签换成英文
        self.cumulative_returns.plot(ax=ax)
        ax.set_title('Factor Group Backtest Cumulative Returns', fontsize=16)
        ax.set_ylabel('Cumulative Value')
        ax.set_xlabel('Date')
        ax.legend(title='Portfolio')
        # *******************************

        ax.grid(True)
        plt.tight_layout()
        plt.show()
        print("Plotting complete.")


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
            backtest.run()

            backtest.performance_analysis()
            backtest.plot_results()

    else:
        print(f"Error: Data directory not found at '{DATA_DIR_PATH}'.")