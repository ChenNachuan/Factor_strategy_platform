import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, List

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'data_manager'))
sys.path.append(str(project_root / 'factor_library' / 'fundamental'))

try:
    from backtest_engine.performance import PerformanceAnalyzer
    from data_manager.data import DataManager
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保相关模块存在且可访问")
    raise

class BacktestEngine:
    """
    向量化回测引擎 (重构版)
    
    适配新的 SizeFactor 数据结构，支持：
    - 周度调仓功能
    - 灵活的多头方向选择
    - 新的数据管理系统
    - 标准化的因子数据格式
    """

    def __init__(self, 
                 data_manager: Optional[DataManager] = None,
                 n_groups: int = 5, 
                 fee: float = 0.001, 
                 long_direction: str = 'high',
                 rebalance_freq: str = 'weekly',
                 factor_name: str = 'factor'):
        """
        初始化回测引擎

        Args:
            data_manager: DataManager实例，用于数据加载
            n_groups: 分组数量，默认为5
            fee: 交易手续费，默认为0.1%
            long_direction: 多头方向
                'high' - 做多因子值最高的组 (适用于正向因子)
                'low' - 做多因子值最低的组 (适用于负向因子)
            rebalance_freq: 调仓频率 ['daily', 'weekly', 'monthly']
            factor_name: 因子列名，默认为 'factor'
        """
        self.data_manager = data_manager or DataManager()
        self.n_groups = n_groups
        self.fee = fee
        self.rebalance_freq = rebalance_freq
        self.factor_name = factor_name
        
        if long_direction not in ['high', 'low']:
            raise ValueError("参数 long_direction 必须是 'high' 或 'low'")
        self.long_direction = long_direction
        
        if rebalance_freq not in ['daily', 'weekly', 'monthly']:
            raise ValueError("参数 rebalance_freq 必须是 'daily', 'weekly', 或 'monthly'")
        
        # 回测结果存储
        self.factor_data = None
        self.stock_data = None
        self.portfolio_returns = None
        self.combined_data = None
        
        print(f"🔧 回测引擎初始化完成")
        print(f"   多头方向: {self.long_direction}")
        print(f"   调仓频率: {self.rebalance_freq}")
        print(f"   交易费用: {self.fee:.3%}")
        print(f"   因子名称: {self.factor_name}")

    def prepare_data(self, 
                    start_date: str,
                    end_date: str,
                    stock_codes: Optional[List[str]] = None,
                    factor_method: str = 'log_market_cap') -> None:
        """
        准备回测所需的数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            stock_codes: 股票代码列表
            factor_method: 因子计算方法
        """
        print(f"\n📊 准备回测数据...")
        print(f"   时间范围: {start_date} ~ {end_date}")
        
        # 1. 计算因子数据
        print("🔄 计算规模因子...")
        # 延迟导入，避免循环依赖
        from factor_library.fundamental.size_factor import calculate_size_factor
        self.factor_data = calculate_size_factor(
            data_manager=self.data_manager,
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
        )
        
        # 2. 加载股票价格数据
        print("📈 加载股票价格数据...")
        self.stock_data = self.data_manager.load_data(
            'daily',
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            verbose=False
        )
        
        if self.stock_data is None or self.stock_data.empty:
            raise ValueError("❌ 无法加载股票价格数据")
        
        # 3. 计算未来收益率
        print("📊 计算股票收益率...")
        self.stock_data = self.stock_data.sort_values(['ts_code', 'trade_date'])
        self.stock_data['next_return'] = self.stock_data.groupby('ts_code')['close'].pct_change().shift(-1)
        
        # 4. 合并因子和收益率数据
        print("🔗 合并因子和价格数据...")
        factor_reset = self.factor_data.reset_index()
        stock_subset = self.stock_data[['ts_code', 'trade_date', 'next_return']].copy()
        
        self.combined_data = pd.merge(
            factor_reset, 
            stock_subset,
            left_on=['ts_code', 'trade_date'],
            right_on=['ts_code', 'trade_date'],
            how='inner'
        )
        
        # 移除缺失值（使用显式的因子列名）
        if self.factor_name not in self.combined_data.columns:
            # 如果指定的因子名不存在，尝试使用第一个非标准列
            standard_cols = {'ts_code', 'trade_date', 'next_return'}
            available_factors = [c for c in self.combined_data.columns if c not in standard_cols]
            if available_factors:
                self.factor_name = available_factors[0]
                print(f"   因子列 '{self.factor_name}' 自动识别")
            else:
                raise ValueError(f"❌ 无法找到因子列 '{self.factor_name}'")
        
        self.combined_data.dropna(subset=[self.factor_name, 'next_return'], inplace=True)
        
        print(f"✅ 数据准备完成:")
        print(f"   因子数据: {len(self.factor_data):,} 条")
        print(f"   价格数据: {len(self.stock_data):,} 条")
        print(f"   合并数据: {len(self.combined_data):,} 条")
        
    def _get_rebalance_dates(self) -> List[str]:
        """获取调仓日期列表"""
        if self.combined_data is None:
            raise ValueError("请先调用 prepare_data 方法")
            
        all_dates = sorted(self.combined_data['trade_date'].unique())
        
        if self.rebalance_freq == 'daily':
            return all_dates
        elif self.rebalance_freq == 'weekly':
            # 每周五调仓（或最后一个交易日）
            df_dates = pd.DataFrame({'trade_date': all_dates})
            df_dates['trade_date'] = pd.to_datetime(df_dates['trade_date'])
            df_dates['week'] = df_dates['trade_date'].dt.to_period('W')
            weekly_dates = df_dates.groupby('week')['trade_date'].max().dt.strftime('%Y-%m-%d').tolist()
            return weekly_dates
        elif self.rebalance_freq == 'monthly':
            # 每月最后一个交易日调仓
            df_dates = pd.DataFrame({'trade_date': all_dates})
            df_dates['trade_date'] = pd.to_datetime(df_dates['trade_date'])
            df_dates['month'] = df_dates['trade_date'].dt.to_period('M')
            monthly_dates = df_dates.groupby('month')['trade_date'].max().dt.strftime('%Y-%m-%d').tolist()
            return monthly_dates
        else:
            return all_dates

    def run(self) -> pd.DataFrame:
        """
        执行向量化回测
        
        Returns:
            pd.DataFrame: 投资组合收益率数据
        """
        if self.combined_data is None:
            raise ValueError("❌ 请先调用 prepare_data 方法准备数据")
            
        print(f"\n🚀 开始执行回测...")
        print(f"   调仓频率: {self.rebalance_freq}")
        
        # 1. 获取调仓日期
        rebalance_dates = self._get_rebalance_dates()
        print(f"   调仓次数: {len(rebalance_dates)} 次")
        
        # 2. 按调仓日期分组并计算收益率（使用显式的因子名）
        if self.factor_data is None:
            raise ValueError("❌ 缺少因子数据，请先调用 prepare_data 方法")
        
        all_returns = []
        last_positions = None  # 记录上期等权持仓集合
        turnover_cost_series = []  # 记录仅在调仓日扣除的成本（对组合收益的冲击）
        
        for i, rebal_date in enumerate(rebalance_dates):
            if i == len(rebalance_dates) - 1:
                continue  # 最后一期无法计算收益率
                
            next_rebal_date = rebalance_dates[i + 1]
            
            # 获取调仓日的因子数据进行分组
            rebal_data = self.combined_data[
                self.combined_data['trade_date'] == rebal_date
            ].copy()
            
            if len(rebal_data) == 0:
                continue
                
            # 分组（使用显式的因子名）
            try:
                rebal_data['group'] = pd.qcut(
                    rebal_data[self.factor_name], 
                    self.n_groups, 
                    labels=False, 
                    duplicates='drop'
                ) + 1
            except ValueError:
                # 处理分位数相同的情况
                rebal_data['group'] = pd.cut(
                    rebal_data[self.factor_name], 
                    self.n_groups, 
                    labels=False
                ) + 1
            
            # 获取持有期内的所有收益率
            period_data = self.combined_data[
                (self.combined_data['trade_date'] > rebal_date) & 
                (self.combined_data['trade_date'] <= next_rebal_date)
            ].copy()
            
            # 合并分组信息
            period_data = pd.merge(
                period_data[['ts_code', 'trade_date', 'next_return']],
                rebal_data[['ts_code', 'group']],
                on='ts_code',
                how='inner'
            )
            
            # 计算各组日收益率
            group_returns = period_data.groupby(['trade_date', 'group'])['next_return'].mean().unstack()
            group_returns.columns = [f'Group_{int(g)}' for g in group_returns.columns]
            
            # 计算本期等权持仓（以组为单位，后续组合构建再取top/bottom）
            # 这里记录股票层面的持仓集合用于换手率估计
            current_positions = set(rebal_data['ts_code'])
            if self.fee > 0:
                if last_positions is None:
                    est_turnover = 1.0  # 首期建仓视为100%换手
                else:
                    # 近似换手率 = (新旧持仓的对称差集规模) / 当前持仓规模
                    diff_count = len(current_positions.symmetric_difference(last_positions))
                    denom = max(len(current_positions | last_positions), 1)
                    est_turnover = diff_count / denom
                # 将该期的成本分配到期首的第一个交易日作为一次性冲击
                if not group_returns.empty:
                    first_day = group_returns.index.min()
                    turnover_cost_series.append((first_day, est_turnover * self.fee))
            last_positions = current_positions

            all_returns.append(group_returns)
        
        # 3. 合并所有期间的收益率
        if not all_returns:
            raise ValueError("❌ 无法计算投资组合收益率")
            
        portfolio_returns = pd.concat(all_returns, axis=0).sort_index()
        
        # 4. 构建多空组合
        top_group = f'Group_{self.n_groups}'
        bottom_group = 'Group_1'
        
        if self.long_direction == 'high':
            long_portfolio = portfolio_returns[top_group]
            short_portfolio = portfolio_returns[bottom_group]
            print(f"📈 策略: 做多 {top_group}, 做空 {bottom_group}")
        else:
            long_portfolio = portfolio_returns[bottom_group]
            short_portfolio = portfolio_returns[top_group]
            print(f"📈 策略: 做多 {bottom_group}, 做空 {top_group}")
        
        # 5. 计算最终组合收益率
        portfolio_returns['Long_Short'] = long_portfolio - short_portfolio
        portfolio_returns['Long_Only'] = long_portfolio
        
        # 6. 考虑交易成本（仅在调仓日按估计换手率一次性扣除）
        if self.fee > 0 and turnover_cost_series:
            cost_df = (
                pd.DataFrame(turnover_cost_series, columns=['trade_date', 'cost'])
                .groupby('trade_date')['cost']
                .sum()
            )
            # 构建与组合收益对齐的成本序列
            cost_series = pd.Series(0.0, index=portfolio_returns.index)
            common_idx = cost_series.index.intersection(cost_df.index)
            cost_series.loc[common_idx] = cost_df.loc[common_idx].values
            # 对 Long_Only 视作单边成本；Long_Short 视作双边成本近似
            portfolio_returns['Long_Only'] = portfolio_returns['Long_Only'] - cost_series
            portfolio_returns['Long_Short'] = portfolio_returns['Long_Short'] - 2 * cost_series
            print(f"💰 交易成本: 在 {len(cost_df)} 次调仓日按估计换手率扣除，单边费率 {self.fee:.3%}")
        
        self.portfolio_returns = portfolio_returns
        
        print(f"✅ 回测完成!")
        print(f"   回测期间: {portfolio_returns.index.min()} ~ {portfolio_returns.index.max()}")
        print(f"   数据点数: {len(portfolio_returns):,} 条")
        
        return portfolio_returns
    
    def get_performance_analysis(self) -> 'PerformanceAnalyzer':
        """
        获取性能分析器
        
        Returns:
            PerformanceAnalyzer: 配置好的性能分析器实例
        """
        if self.portfolio_returns is None:
            raise ValueError("❌ 请先运行回测 (调用 run 方法)")
            
        if self.factor_data is None or self.combined_data is None:
            raise ValueError("❌ 缺少因子数据，请先调用 prepare_data 方法")
        
        # 准备 PerformanceAnalyzer 需要的数据格式
        # 重新格式化 combined_data 以匹配 PerformanceAnalyzer 的期望
        master_data_formatted = self.combined_data.copy()
        master_data_formatted['date'] = master_data_formatted['trade_date']
        master_data_formatted['stock_code'] = master_data_formatted['ts_code']
        master_data_formatted['next_day_return'] = master_data_formatted['next_return']
        
        # 重新格式化 factor_data，确保只包含因子列
        factor_data_formatted = self.factor_data.reset_index()
        factor_data_formatted['date'] = factor_data_formatted['trade_date']
        factor_data_formatted['stock_code'] = factor_data_formatted['ts_code']
        
        # 使用显式的因子名，避免索引硬编码
        factor_data_final = factor_data_formatted[['date', 'stock_code', self.factor_name]].copy()
        
        analyzer = PerformanceAnalyzer(
            portfolio_returns=self.portfolio_returns,
            factor_data=factor_data_final,
            master_data=master_data_formatted
        )
        
        return analyzer


def main():
    """
    主函数：演示新的回测引擎使用方法
    """
    print("=" * 60)
    print("规模因子回测演示 (重构版)")
    print("=" * 60)
    
    # 注意：这只是一个演示函数，实际使用中应该通过 size_factor.py 来运行回测
    print("此 main 函数仅用于演示，请使用 size_factor.py 中的 run_size_factor_backtest 函数")
    print("例如：")
    print("from factor_library.fundamental.size_factor import run_size_factor_backtest")
    print("result = run_size_factor_backtest(start_date='2024-01-01', end_date='2024-03-31')")
    
    try:
        # 这里可以添加一个简单的测试，但主要逻辑应该在 size_factor.py 中
        from factor_library.fundamental.size_factor import run_size_factor_backtest
        
        result = run_size_factor_backtest(
            start_date='2024-01-01',
            end_date='2024-01-31',
            long_direction='low'
        )
        
        print(f"\n测试运行成功！策略总收益率: {result['performance_metrics']['total_return']:.4f}")
        
    except Exception as e:
        print(f"测试运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()