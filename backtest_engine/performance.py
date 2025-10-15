import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


class PerformanceAnalyzer:
    """
    回测业绩分析器，增加了对因子本身的评估指标（如IC）和更丰富的可视化。
    """

    def __init__(self, portfolio_returns, factor_data, master_data, benchmark_returns=None):
        """
        初始化业绩分析器。

        Args:
            portfolio_returns (pd.DataFrame): 包含策略/分组日收益率的DataFrame。
            factor_data (pd.DataFrame): 包含原始因子值的DataFrame。
            master_data (pd.DataFrame): 包含'next_day_return'的主数据框。
            benchmark_returns (pd.Series, optional): 基准组合的日收益率序列。
        """
        if not isinstance(portfolio_returns, pd.DataFrame):
            raise TypeError("portfolio_returns must be a pandas DataFrame.")

        self.returns = portfolio_returns
        self.factor_data = factor_data
        self.master_data = master_data
        self.benchmark = benchmark_returns
        self.metrics = None
        self.cumulative_returns = None
        self.ic_series = None
        print("Performance Analyzer initialized.")

    def calculate_metrics(self, annual_trading_days=252):
        """
        计算所有关键业绩指标。
        """
        print("\n[Performance Analysis] Calculating portfolio performance metrics...")
        self.cumulative_returns = (1 + self.returns).cumprod()

        days = len(self.cumulative_returns)
        annualized_return = self.cumulative_returns.iloc[-1] ** (annual_trading_days / days) - 1
        annualized_volatility = self.returns.std() * np.sqrt(annual_trading_days)
        sharpe_ratio = annualized_return / annualized_volatility

        running_max = self.cumulative_returns.cummax()
        drawdown = (self.cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        if self.benchmark is not None:
            active_return = self.returns.subtract(self.benchmark, axis=0)
            tracking_error = active_return.std() * np.sqrt(annual_trading_days)
            if not self.benchmark.empty:
                benchmark_annual_return = (1 + self.benchmark).prod() ** (annual_trading_days / days) - 1
                information_ratio = (annualized_return - benchmark_annual_return) / tracking_error
            else:
                information_ratio = np.nan
        else:
            information_ratio = np.nan

        self.metrics = pd.DataFrame({
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Information Ratio (vs Benchmark)': information_ratio
        })

        print("Portfolio Performance Metrics:")
        print(self.metrics.round(4))

        self.calculate_ic_analysis()

        return self.metrics

    def calculate_ic_analysis(self):
        """
        计算信息系数 (IC) 相关指标。
        """
        print("\n[Factor Analysis] Calculating Information Coefficient (IC)...")

        # 检查数据列名并适配不同的格式
        factor_columns = self.factor_data.columns.tolist()
        master_columns = self.master_data.columns.tolist()
        
        # 获取第一个数值列作为因子列（跳过日期和股票代码列）
        factor_name = None
        for col in factor_columns:
            if col not in ['date', 'trade_date', 'stock_code', 'ts_code']:
                factor_name = col
                break
        
        if factor_name is None:
            raise ValueError("无法找到有效的因子列")
        
        date_col = 'date' if 'date' in self.factor_data.columns else 'trade_date'
        stock_col = 'stock_code' if 'stock_code' in self.factor_data.columns else 'ts_code'
        return_col = 'next_day_return' if 'next_day_return' in self.master_data.columns else 'next_return'
        
        # 准备合并用的数据 - 只选择需要的列避免重复
        factor_data_clean = self.factor_data[[date_col, stock_col, factor_name]].copy()
        factor_data_clean.columns = ['date', 'stock_code', factor_name]
        
        master_date_col = 'date' if 'date' in self.master_data.columns else 'trade_date'
        master_stock_col = 'stock_code' if 'stock_code' in self.master_data.columns else 'ts_code'
        
        master_data_clean = self.master_data[[master_date_col, master_stock_col, return_col]].copy()
        master_data_clean.columns = ['date', 'stock_code', 'next_day_return']
        
        merged_data = pd.merge(
            factor_data_clean, 
            master_data_clean, 
            on=['date', 'stock_code']
        )
        merged_data.dropna(inplace=True)

        if len(merged_data) == 0:
            print("警告: 合并后数据为空，无法计算IC")
            self.ic_series = pd.Series([], name='Daily_IC')
            return

        # 计算每日IC
        ic_list = []
        dates = []
        
        for date, group in merged_data.groupby('date'):
            if len(group) > 1:
                # 确保数据类型正确
                factor_values = group[factor_name].astype(float)
                return_values = group['next_day_return'].astype(float)
                
                # 去除NaN值
                valid_mask = ~(pd.isna(factor_values) | pd.isna(return_values))
                if valid_mask.sum() > 1:
                    try:
                        correlation, p_value = spearmanr(factor_values[valid_mask], return_values[valid_mask])
                        ic_list.append(correlation)
                        dates.append(date)
                    except (ValueError, TypeError):
                        continue
        
        self.ic_series = pd.Series(ic_list, index=dates, name='Daily_IC')

        ic_mean = self.ic_series.mean()
        ic_std = self.ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std != 0 else 0
        ic_gt_2_pct = (self.ic_series.abs() > 0.02).sum() / len(self.ic_series) * 100

        ic_metrics = pd.DataFrame({
            'IC Mean': [ic_mean],
            'IC Std.': [ic_std],
            'IC IR': [ic_ir],
            '|IC|>2% Ratio (%)': [ic_gt_2_pct]
        }, index=['Factor'])

        print("\nFactor Predictive Power Analysis (IC Metrics):")
        print(ic_metrics.round(4))

    def plot_results(self, rolling_window=60):
        """
        绘制包含累计收益率和IC分析的2x2图表。

        Args:
            rolling_window (int): 计算IC滚动均值时使用的窗口期。
        """
        if self.cumulative_returns is None or self.ic_series is None:
            print("Error: Please run .calculate_metrics() first.")
            return

        print("\n[Visualization] Plotting 2x2 analysis charts...")

        plt.style.use('seaborn-v0_8-whitegrid')

        # ******** 这是核心修改：创建一个2x2的子图画布 ********
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle('Factor Performance and IC Analysis', fontsize=20)
        # ******************************************************

        # --- 子图 1: 累计收益率曲线 (Top-Left) ---
        self.cumulative_returns.plot(ax=axes[0, 0])
        axes[0, 0].set_title('Cumulative Returns by Factor Group', fontsize=14)
        axes[0, 0].set_ylabel('Cumulative Value')
        axes[0, 0].legend(title='Portfolio')

        # --- 子图 2: 每日IC时间序列图 (Bottom-Left) ---
        self.ic_series.plot(ax=axes[1, 0], color='steelblue', alpha=0.7)
        axes[1, 0].axhline(self.ic_series.mean(), color='red', linestyle='--',
                           label=f'Mean IC: {self.ic_series.mean():.4f}')
        axes[1, 0].set_title('Daily Information Coefficient (IC)', fontsize=14)
        axes[1, 0].set_ylabel('IC Value')
        axes[1, 0].legend()

        # --- 子图 3: IC分布直方图 (Top-Right) ---
        self.ic_series.hist(bins=50, ax=axes[0, 1], color='skyblue', edgecolor='black')
        axes[0, 1].axvline(self.ic_series.mean(), color='red', linestyle='--', label='Mean IC')
        axes[0, 1].set_title('Distribution of Daily IC', fontsize=14)
        axes[0, 1].set_xlabel('IC Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # --- 子图 4: IC滚动均值 (Bottom-Right) ---
        rolling_ic = self.ic_series.rolling(window=rolling_window).mean()
        rolling_ic.plot(ax=axes[1, 1], color='darkorange')
        axes[1, 1].axhline(self.ic_series.mean(), color='red', linestyle='--', label='Overall Mean IC')
        axes[1, 1].set_title(f'{rolling_window}-Day Rolling IC Mean', fontsize=14)
        axes[1, 1].set_ylabel('Rolling IC Mean')
        axes[1, 1].legend()

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust layout to make room for suptitle
        plt.show()
        print("Plotting complete.")