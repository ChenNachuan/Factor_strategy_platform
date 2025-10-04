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
        print("回测引擎初始化完成。")

    def run(self):
        """
        执行回测流程。
        """
        print("\n[回测流程] 开始执行向量化回测...")

        # 1. 数据对齐与合并
        # 我们需要计算下个周期的收益，所以使用shift(-1)
        self.master_data['next_day_return'] = self.master_data.groupby(level='stock_code')['adj_close'].transform(
            lambda x: x.pct_change().shift(-1)
        )

        combined_data = pd.merge(self.factor_data, self.master_data[['next_day_return']], on=['date', 'stock_code'])
        combined_data.dropna(subset=[self.factor_data.columns[0], 'next_day_return'], inplace=True)
        print(" -> 步骤1: 数据对齐与合并完成。")

        # 2. 投资组合分组
        factor_name = self.factor_data.columns[0]
        combined_data['group'] = combined_data.groupby('date')[factor_name].transform(
            lambda x: pd.qcut(x, self.n_groups, labels=False, duplicates='drop') + 1
        )
        print(f" -> 步骤2: 已按 '{factor_name}' 因子值将股票分为 {self.n_groups} 组。")

        # 3. 计算每个分组的日收益率（等权重）
        group_returns = combined_data.groupby(['date', 'group'])['next_day_return'].mean().unstack()
        group_returns.columns = [f'Group_{int(g)}' for g in group_returns.columns]
        print(" -> 步骤3: 计算各分组每日收益率完成。")

        # 4. 构建多空组合 (Long-Short Portfolio)
        long_portfolio = group_returns['Group_1']
        short_portfolio = group_returns[f'Group_{self.n_groups}']

        group_returns['Long_Short'] = long_portfolio - short_portfolio
        group_returns['Long_Only'] = long_portfolio
        print(" -> 步骤4: 构建多空对冲组合完成。")

        self.portfolio_returns = group_returns
        print("[回测流程] 回测执行完毕！")

    def performance_analysis(self):
        """
        计算并展示关键业绩指标。
        """
        if self.portfolio_returns is None:
            print("错误：请先运行 .run() 方法进行回测。")
            return

        print("\n[业绩分析] 正在计算关键业绩指标...")

        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        days = len(cumulative_returns)
        annualized_return = cumulative_returns.iloc[-1] ** (252 / days) - 1
        annualized_volatility = self.portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        performance_metrics = pd.DataFrame({
            '年化收益率': annualized_return,
            '年化波动率': annualized_volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown
        })

        print("业绩指标计算完成：")
        print(performance_metrics.round(4))

        self.cumulative_returns = cumulative_returns
        return performance_metrics

    def plot_results(self):
        """
        绘制累计收益率曲线图。
        """
        if not hasattr(self, 'cumulative_returns'):
            print("错误：请先运行 .performance_analysis() 方法计算业绩。")
            return

        print("\n[结果可视化] 正在绘制净值曲线...")

        # 解决matplotlib中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        self.cumulative_returns.plot(ax=ax)

        ax.set_title('因子分组回测累计收益率', fontsize=16)
        ax.set_ylabel('累计净值')
        ax.set_xlabel('日期')
        ax.legend(title='投资组合')
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        print("绘图完成。")


if __name__ == '__main__':
    # 这是一个演示如何使用回测引擎的完整流程

    # ******** 这是核心修改 1 ********
    # 动态添加项目根目录到Python路径
    import sys

    # __file__ 是当前文件: .../backtest_engine/engine.py
    # os.path.dirname(__file__) 是 .../backtest_engine
    # os.path.join(..., '..') 是 .../factor_strategy_platform
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    # *******************************

    # 1. 加载数据
    from data_manager.loader import load_and_clean_data

    # 2. 计算因子
    from factor_library.fundamental.size_factor import SizeFactor

    # ******** 这是核心修改 2 ********
    # 构建到数据文件夹的正确路径
    # 从项目根目录 project_root 出发，进入 data_manager/DemoData
    DATA_DIR_PATH = os.path.join(project_root, "data_manager", "DemoData")
    # **************************

    if os.path.exists(DATA_DIR_PATH):
        master_data, _ = load_and_clean_data(DATA_DIR_PATH)

        if master_data is not None and not master_data.empty:
            # 计算规模因子
            size_calculator = SizeFactor(master_data)
            size_factor = size_calculator.calculate_factor()

            # 3. 运行回测
            backtest = BacktestEngine(master_data, size_factor, n_groups=5)
            backtest.run()

            # 4. 分析和展示结果
            backtest.performance_analysis()
            backtest.plot_results()

    else:
        print(f"错误: 主程序无法找到数据文件夹 '{DATA_DIR_PATH}'。")