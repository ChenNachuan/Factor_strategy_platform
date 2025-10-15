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
    from data import DataManager
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
                 rebalance_freq: str = 'weekly'):
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
        """
        self.data_manager = data_manager or DataManager()
        self.n_groups = n_groups
        self.fee = fee
        self.rebalance_freq = rebalance_freq
        
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
        size_calculator = SizeFactor(self.data_manager)
        self.factor_data = size_calculator.calculate_factor(
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            method=factor_method
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
        
        # 移除缺失值
        factor_col = self.factor_data.columns[0]
        self.combined_data.dropna(subset=[factor_col, 'next_return'], inplace=True)
        
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
        
        # 2. 按调仓日期分组并计算收益率
        factor_col = self.factor_data.columns[0]
        all_returns = []
        
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
                
            # 分组
            try:
                rebal_data['group'] = pd.qcut(
                    rebal_data[factor_col], 
                    self.n_groups, 
                    labels=False, 
                    duplicates='drop'
                ) + 1
            except ValueError:
                # 处理分位数相同的情况
                rebal_data['group'] = pd.cut(
                    rebal_data[factor_col], 
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
        
        # 6. 考虑交易成本
        if self.fee > 0:
            # 计算调仓次数对应的成本
            trading_cost_long = self.fee / len(rebalance_dates) * len(portfolio_returns)
            trading_cost_ls = self.fee * 2 / len(rebalance_dates) * len(portfolio_returns)
            
            portfolio_returns['Long_Only'] = portfolio_returns['Long_Only'] - trading_cost_long
            portfolio_returns['Long_Short'] = portfolio_returns['Long_Short'] - trading_cost_ls
            
            print(f"💰 交易成本: 单边 {self.fee:.3%}, 双边 {self.fee*2:.3%}")
        
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
        
        # 只保留因子列，避免日期列被误识别
        factor_col = self.factor_data.columns[0]
        factor_data_final = factor_data_formatted[['date', 'stock_code', factor_col]].copy()
        
        analyzer = PerformanceAnalyzer(
            portfolio_returns=self.portfolio_returns,
            factor_data=factor_data_final,
            master_data=master_data_formatted
        )
        
        return analyzer


def run_backtest(factor_data: pd.DataFrame,
                data_manager: DataManager,
                start_date: str,
                end_date: str,
                rebalance_freq: str = 'weekly',
                transaction_cost: float = 0.0) -> tuple:
    """
    简化的回测函数，接受预计算的因子数据
    
    参数:
        factor_data: 预计算的因子数据 (MultiIndex: trade_date, stock_code)
        data_manager: 数据管理器实例
        start_date: 开始日期
        end_date: 结束日期
        rebalance_freq: 调仓频率
        transaction_cost: 交易费用
        
    返回:
        tuple: (portfolio_returns, positions)
    """
    print(f"🎯 开始简化回测流程...")
    
    # 创建回测引擎实例
    engine = BacktestEngine(
        data_manager=data_manager,
        transaction_cost=transaction_cost
    )
    
    # 获取股票代码列表
    stock_codes = factor_data.index.get_level_values('stock_code').unique().tolist()
    
    # 加载股票价格数据
    stock_data = data_manager.load_data(
        'daily',
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes
    )
    
    if stock_data is None or stock_data.empty:
        raise ValueError("无法加载股票价格数据")
    
    # 计算股票收益率
    stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
    stock_data['next_return'] = stock_data.groupby('ts_code')['close'].pct_change().shift(-1)
    
    # 执行简化的回测逻辑
    return _simple_backtest(factor_data, stock_data, rebalance_freq, transaction_cost)


def _simple_backtest(factor_data, stock_data, rebalance_freq, transaction_cost):
    """简化的回测实现"""
    # 合并因子和收益率数据
    factor_reset = factor_data.reset_index()
    stock_subset = stock_data[['ts_code', 'trade_date', 'next_return']].copy()
    
    combined_data = pd.merge(
        factor_reset, 
        stock_subset,
        left_on=['stock_code', 'date'],
        right_on=['ts_code', 'trade_date'],
        how='inner'
    )
    
    if combined_data.empty:
        raise ValueError("因子数据与价格数据合并后为空")
    
    # 按日期分组计算收益率
    daily_returns = []
    positions_records = []
    
    # 获取交易日期
    dates = sorted(combined_data['trade_date'].unique())
    
    # 设置调仓频率
    rebalance_interval = {'daily': 1, 'weekly': 5, 'monthly': 20}.get(rebalance_freq, 5)
    
    for i, date in enumerate(dates[:-1]):  # 排除最后一天，因为没有下一天收益
        # 检查是否需要调仓
        if i % rebalance_interval == 0:
            # 获取当日因子数据
            today_data = combined_data[combined_data['trade_date'] == date]
            
            if len(today_data) > 0:
                # 简单策略：按因子值分组，做多因子值最高的50%
                n_stocks = len(today_data)
                top_n = max(1, n_stocks // 2)
                
                # 按因子值排序
                today_data = today_data.sort_values('factor', ascending=False)
                selected_stocks = today_data.head(top_n)
                
                # 等权重配置
                weights = {stock: 1.0/len(selected_stocks) for stock in selected_stocks['ts_code']}
                
                positions_records.append({
                    'date': date,
                    'positions': weights
                })
                
                # 计算组合收益率
                portfolio_return = selected_stocks['next_return'].mean()
                
                # 减去交易费用（简化处理）
                if i > 0:  # 第一次建仓不收费
                    portfolio_return -= transaction_cost
                
                daily_returns.append(portfolio_return)
            else:
                daily_returns.append(0.0)
        else:
            # 非调仓日，使用上次的持仓
            if positions_records:
                last_positions = positions_records[-1]['positions']
                today_data = combined_data[combined_data['trade_date'] == date]
                
                if len(today_data) > 0:
                    held_stocks = today_data[today_data['ts_code'].isin(last_positions.keys())]
                    if len(held_stocks) > 0:
                        portfolio_return = held_stocks['next_return'].mean()
                        daily_returns.append(portfolio_return)
                    else:
                        daily_returns.append(0.0)
                else:
                    daily_returns.append(0.0)
            else:
                daily_returns.append(0.0)
    
    # 转换为pandas Series
    portfolio_returns = pd.Series(daily_returns, index=dates[:-1])
    positions_df = pd.DataFrame(positions_records)
    
    print(f"✅ 简化回测完成!")
    print(f"  收益序列长度: {len(portfolio_returns)}")
    print(f"  调仓记录: {len(positions_records)}")
    
    return portfolio_returns, positions_df
    """
    便捷的回测运行函数
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        stock_codes: 股票代码列表
        factor_method: 因子计算方法
        n_groups: 分组数量
        long_direction: 多头方向
        rebalance_freq: 调仓频率
        fee: 交易费用
        show_analysis: 是否显示分析结果
        
    Returns:
        Dict: 包含回测结果的字典
    """
    print("🎯 开始运行便捷回测流程...")
    
    # 1. 创建回测引擎
    engine = BacktestEngine(
        n_groups=n_groups,
        long_direction=long_direction,
        rebalance_freq=rebalance_freq,
        fee=fee
    )
    
    # 2. 准备数据
    engine.prepare_data(
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        factor_method=factor_method
    )
    
    # 3. 运行回测
    returns = engine.run()
    
    # 4. 性能分析
    analyzer = engine.get_performance_analysis()
    metrics = analyzer.calculate_metrics()
    
    if show_analysis:
        analyzer.plot_results()
    
    result = {
        'returns': returns,
        'metrics': metrics,
        'analyzer': analyzer,
        'engine': engine
    }
    
    print("🎉 便捷回测流程完成！")
    return result


def main():
    """
    主函数：演示新的回测引擎使用方法
    """
    print("=" * 60)
    print("📊 规模因子回测演示 (重构版)")
    print("=" * 60)
    
    try:
        # 使用便捷函数运行回测
        result = run_backtest(
            start_date='2024-01-01',
            end_date='2024-06-30',
            stock_codes=['000001.SZ', '000002.SZ', '600000.SH', '600036.SH'],
            factor_method='log_market_cap',
            n_groups=3,  # 样本较小，使用3组
            long_direction='low',  # 规模因子通常小市值表现更好
            rebalance_freq='weekly',
            fee=0.001,
            show_analysis=True
        )
        
        print("\n📈 回测结果概览:")
        print(f"策略收益率统计:")
        print(result['returns'][['Long_Only', 'Long_Short']].describe())
        
        print(f"\n📊 性能指标:")
        print(result['metrics'])
        
    except Exception as e:
        print(f"❌ 回测执行失败: {e}")
        import traceback
        traceback.print_exc()


def run_backtest(factor_data: pd.DataFrame,
                data_manager: DataManager,
                start_date: str,
                end_date: str,
                rebalance_freq: str = 'weekly',
                transaction_cost: float = 0.0) -> tuple:
    """
    简化的回测函数，接受预计算的因子数据
    
    参数:
        factor_data: 预计算的因子数据 (MultiIndex: date, stock_code)
        data_manager: 数据管理器实例
        start_date: 开始日期
        end_date: 结束日期
        rebalance_freq: 调仓频率
        transaction_cost: 交易费用
        
    返回:
        tuple: (portfolio_returns, positions)
    """
    print(f"🎯 开始简化回测流程...")
    
    # 创建回测引擎实例
    engine = BacktestEngine(
        data_manager=data_manager,
        fee=transaction_cost
    )
    
    # 直接设置因子数据
    engine.factor_data = factor_data
    
    # 获取股票代码列表
    stock_codes = factor_data.index.get_level_values('stock_code').unique().tolist()
    
    # 加载股票价格数据
    stock_data = data_manager.load_data(
        'daily',
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes
    )
    
    if stock_data is None or stock_data.empty:
        raise ValueError("无法加载股票价格数据")
    
    # 计算股票收益率
    stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
    stock_data['next_return'] = stock_data.groupby('ts_code')['close'].pct_change().shift(-1)
    
    # 合并因子和收益率数据
    factor_reset = factor_data.reset_index()
    stock_subset = stock_data[['ts_code', 'trade_date', 'next_return']].copy()
    
    combined_data = pd.merge(
        factor_reset, 
        stock_subset,
        left_on=['stock_code', 'date'],
        right_on=['ts_code', 'trade_date'],
        how='inner'
    )
    
    if combined_data.empty:
        raise ValueError("因子数据与价格数据合并后为空")
    
    # 设置合并后的数据
    engine.combined_data = combined_data
    
    # 执行简化的回测逻辑
    return _simple_backtest(factor_data, stock_data, rebalance_freq, transaction_cost)


if __name__ == '__main__':
    main()