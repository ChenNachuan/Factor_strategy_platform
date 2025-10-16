"""
季度盈利惊喜 (Quarterly Earnings Surprise) 因子

因子逻辑：
-----------
通过对比本季度EPS与去年同期EPS的差异，捕捉公司盈利的季度改善信号。
使用公告日前一天的股价进行标准化，使得因子值在不同股票间可比。

因子公式：
---------
Factor_Value = (EPS_current - EPS_last_year_same_quarter) / Price_announcement_date-1

其中：
- EPS_current: 本季度每股收益
- EPS_last_year_same_quarter: 去年同期每股收益
- Price_announcement_date-1: 财报公告日前一天的收盘价

数据来源：
---------
1. 财务数据: income表（利润表）
   - ts_code: 股票代码
   - ann_date: 财报公告日
   - end_date: 报告期结束日
   - eps: 基本每股收益
   - report_type: 报告类型（1=年报,2=中报,3=季报）

2. 行情数据: daily表
   - close: 收盘价

因子特征：
---------
- 因子类型: 基本面 - 盈利质量因子
- 因子方向: 做多高因子值（盈利改善的股票）
- 更新频率: 每日（在财报公告日更新，其他日期延续上次值）
- 适用范围: 全市场A股

应用策略：
---------
1. 在每个财报公告日(ann_date)，计算该股票的因子值
2. 该因子值在股票上保留，直到下一个财报公告日
3. 这样形成每日更新的因子序列，可用于回测
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

# 路径：把项目根目录加入 sys.path，便于使用绝对包导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager


def calculate_earnings_surprise_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算季度盈利惊喜因子
    
    因子计算步骤：
    1. 加载利润表数据，获取每个季度的EPS和公告日
    2. 对每个季度财报，找到其去年同期的季报
    3. 计算EPS同比差值 (EPS_diff)
    4. 获取公告日前一天的股价
    5. 标准化: Factor = EPS_diff / Price
    6. 将因子值在公告日到下一个公告日之间延续（forward fill）
    
    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    start_date : str
        开始日期，格式 'YYYY-MM-DD'
    end_date : str
        结束日期，格式 'YYYY-MM-DD'
    stock_codes : Optional[List[str]]
        股票代码列表，如为 None 则使用所有可用股票

    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'.
        因子值为标准化的盈利惊喜值，值越高表示盈利改善越明显。
        
    Notes
    -----
    - 只使用季报数据（可以扩展到包括年报、中报）
    - 去年同期数据缺失的记录会被跳过
    - 股价为0或缺失的记录会被过滤
    - 异常值（绝对值过大）会被截断处理
    """
    print(f"\n{'='*60}")
    print("季度盈利惊喜 (Quarterly Earnings Surprise) 因子计算")
    print(f"{'='*60}")
    
    # 步骤1: 加载利润表数据
    print("\n步骤1: 加载利润表数据...")
    income_data = data_manager.load_data('income', stock_codes=stock_codes)
    
    if income_data is None or income_data.empty:
        raise ValueError("无法加载利润表数据")
    
    print(f"✅ 加载利润表数据: {len(income_data):,} 条记录")
    
    # 检查必要字段
    required_fields = ['ts_code', 'ann_date', 'end_date', 'basic_eps']
    missing_fields = [f for f in required_fields if f not in income_data.columns]
    if missing_fields:
        raise ValueError(f"利润表数据缺少必要字段: {missing_fields}")
    
    # 转换日期格式
    income_data = income_data.copy()
    income_data['ann_date'] = pd.to_datetime(income_data['ann_date'])
    income_data['end_date'] = pd.to_datetime(income_data['end_date'])
    
    # 过滤：只保留有公告日和报告期的数据
    income_data = income_data.dropna(subset=['ann_date', 'end_date', 'basic_eps'])
    print(f"✅ 过滤后数据: {len(income_data):,} 条记录")
    
    # 步骤2: 识别季度报告
    print("\n步骤2: 识别季度报告...")
    # 提取季度信息（Q1, Q2, Q3, Q4）
    income_data['year'] = income_data['end_date'].dt.year
    income_data['quarter'] = income_data['end_date'].dt.quarter
    
    # 只保留季报数据（可以通过end_date的月份判断：3,6,9,12月）
    income_data = income_data[income_data['end_date'].dt.month.isin([3, 6, 9, 12])]
    print(f"✅ 季报数据: {len(income_data):,} 条记录")
    print(f"   覆盖股票: {income_data['ts_code'].nunique()} 只")
    print(f"   时间范围: {income_data['end_date'].min()} 至 {income_data['end_date'].max()}")
    
    # 步骤3: 对齐去年同期数据
    print("\n步骤3: 对齐去年同期数据...")
    # 为每条记录添加去年同期的end_date
    income_data['last_year_end_date'] = income_data['end_date'] - pd.DateOffset(years=1)
    
    # 自连接：匹配去年同期数据
    # 连接条件：相同股票 + 相同季度 + 相隔一年
    merged = income_data.merge(
        income_data[['ts_code', 'end_date', 'basic_eps', 'ann_date']],
        left_on=['ts_code', 'last_year_end_date'],
        right_on=['ts_code', 'end_date'],
        suffixes=('_current', '_last_year'),
        how='left'
    )
    
    print(f"✅ 对齐前数据: {len(income_data):,} 条")
    print(f"✅ 对齐后数据: {len(merged):,} 条")
    
    # 过滤：只保留成功匹配到去年同期的记录
    merged = merged.dropna(subset=['basic_eps_last_year'])
    print(f"✅ 成功匹配去年同期: {len(merged):,} 条记录")
    
    if merged.empty:
        raise ValueError("没有找到任何可以匹配去年同期的数据，无法计算因子")
    
    # 步骤4: 计算EPS差值
    print("\n步骤4: 计算EPS同比差值...")
    merged['eps_diff'] = merged['basic_eps_current'] - merged['basic_eps_last_year']
    
    # 统计
    print(f"   EPS_diff 统计:")
    print(f"     均值: {merged['eps_diff'].mean():.4f}")
    print(f"     中位数: {merged['eps_diff'].median():.4f}")
    print(f"     标准差: {merged['eps_diff'].std():.4f}")
    print(f"     正值占比: {(merged['eps_diff'] > 0).sum() / len(merged) * 100:.1f}%")
    
    # 步骤5: 加载行情数据，获取公告日前一天的股价
    print("\n步骤5: 获取公告日前一天的股价...")
    
    # 获取所有需要的股票和日期范围
    all_stock_codes = merged['ts_code'].unique().tolist()
    # 扩展日期范围以确保能获取到公告日前一天的数据
    min_ann_date = merged['ann_date_current'].min() - timedelta(days=5)
    max_ann_date = merged['ann_date_current'].max()
    
    print(f"   加载行情数据: {len(all_stock_codes)} 只股票")
    print(f"   日期范围: {min_ann_date.date()} 至 {max_ann_date.date()}")
    
    daily_data = data_manager.load_data(
        'daily',
        start_date=min_ann_date.strftime('%Y-%m-%d'),
        end_date=max_ann_date.strftime('%Y-%m-%d'),
        stock_codes=all_stock_codes
    )
    
    if daily_data is None or daily_data.empty:
        raise ValueError("无法加载日行情数据")
    
    daily_data = daily_data.copy()
    daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'])
    print(f"✅ 加载行情数据: {len(daily_data):,} 条记录")
    
    # 为每个公告日找到前一交易日的股价
    factor_records = []
    
    print("\n步骤6: 计算因子值...")
    for idx, row in merged.iterrows():
        ts_code = row['ts_code']
        ann_date = row['ann_date_current']
        eps_diff = row['eps_diff']
        
        # 获取该股票在公告日前的行情数据
        stock_daily = daily_data[
            (daily_data['ts_code'] == ts_code) & 
            (daily_data['trade_date'] < ann_date)
        ].sort_values('trade_date')
        
        if stock_daily.empty:
            continue
        
        # 获取最近一个交易日的收盘价
        price_prev = stock_daily.iloc[-1]['close']
        
        if pd.isna(price_prev) or price_prev <= 0:
            continue
        
        # 计算因子值：EPS_diff / Price
        factor_value = eps_diff / price_prev
        
        factor_records.append({
            'ts_code': ts_code,
            'ann_date': ann_date,
            'end_date': row['end_date_current'],
            'eps_current': row['basic_eps_current'],
            'eps_last_year': row['basic_eps_last_year'],
            'eps_diff': eps_diff,
            'price_prev': price_prev,
            'factor': factor_value
        })
    
    factor_df = pd.DataFrame(factor_records)
    print(f"✅ 计算因子值: {len(factor_df):,} 条记录")
    
    if factor_df.empty:
        raise ValueError("计算因子后没有有效数据")
    
    # 步骤7: 异常值处理
    print("\n步骤7: 异常值处理...")
    # 使用3倍标准差截断
    factor_mean = factor_df['factor'].mean()
    factor_std = factor_df['factor'].std()
    lower_bound = factor_mean - 3 * factor_std
    upper_bound = factor_mean + 3 * factor_std
    
    print(f"   原始因子统计:")
    print(f"     均值: {factor_mean:.6f}")
    print(f"     标准差: {factor_std:.6f}")
    print(f"     范围: [{factor_df['factor'].min():.6f}, {factor_df['factor'].max():.6f}]")
    print(f"   截断范围: [{lower_bound:.6f}, {upper_bound:.6f}]")
    
    # 截断处理
    outlier_count = ((factor_df['factor'] < lower_bound) | (factor_df['factor'] > upper_bound)).sum()
    factor_df['factor'] = factor_df['factor'].clip(lower=lower_bound, upper=upper_bound)
    print(f"   处理异常值: {outlier_count} 条 ({outlier_count/len(factor_df)*100:.2f}%)")
    
    # 步骤8: 将因子值扩展到每日数据（Forward Fill）
    print("\n步骤8: 扩展因子值到每日...")
    
    # 加载回测期间的所有交易日数据
    daily_all = data_manager.load_data(
        'daily',
        start_date=start_date,
        end_date=end_date,
        stock_codes=all_stock_codes
    )
    
    if daily_all is None or daily_all.empty:
        raise ValueError("无法加载回测期间的日行情数据")
    
    daily_all = daily_all.copy()
    daily_all['trade_date'] = pd.to_datetime(daily_all['trade_date'])
    
    # 创建所有交易日和股票的组合
    all_dates = daily_all['trade_date'].unique()
    all_combinations = daily_all[['trade_date', 'ts_code']].drop_duplicates()
    
    print(f"   回测期间: {len(all_dates)} 个交易日")
    print(f"   股票数量: {len(all_stock_codes)} 只")
    print(f"   总组合数: {len(all_combinations):,}")
    
    # 将因子数据合并到每日数据上
    # 策略：每个公告日的因子值保持到下一个公告日
    factor_daily_list = []
    
    for ts_code in all_stock_codes:
        # 获取该股票的所有因子记录（按公告日排序）
        stock_factors = factor_df[factor_df['ts_code'] == ts_code].sort_values('ann_date')
        
        if stock_factors.empty:
            continue
        
        # 获取该股票的所有交易日
        stock_dates = all_combinations[all_combinations['ts_code'] == ts_code]['trade_date'].values
        
        # 对每个交易日，找到最近的（且不晚于该日期的）公告日的因子值
        for trade_date in stock_dates:
            # 找到在该交易日之前或当天的最近公告
            valid_factors = stock_factors[stock_factors['ann_date'] <= trade_date]
            
            if not valid_factors.empty:
                # 使用最近的因子值
                latest_factor = valid_factors.iloc[-1]['factor']
                factor_daily_list.append({
                    'trade_date': trade_date,
                    'ts_code': ts_code,
                    'factor': latest_factor
                })
    
    result_df = pd.DataFrame(factor_daily_list)
    
    if result_df.empty:
        raise ValueError("扩展到每日后没有有效数据")
    
    print(f"✅ 每日因子数据: {len(result_df):,} 条记录")
    
    # 转换为MultiIndex格式
    result_df['trade_date'] = pd.to_datetime(result_df['trade_date'])
    result_df = result_df.set_index(['trade_date', 'ts_code'])
    result_df = result_df.sort_index()
    
    # 最终统计
    print(f"\n{'='*60}")
    print("因子计算完成！")
    print(f"{'='*60}")
    print(f"📊 最终因子统计:")
    print(f"   总记录数: {len(result_df):,}")
    print(f"   覆盖股票: {result_df.index.get_level_values('ts_code').nunique()}")
    print(f"   覆盖日期: {result_df.index.get_level_values('trade_date').nunique()}")
    print(f"   因子值范围: [{result_df['factor'].min():.6f}, {result_df['factor'].max():.6f}]")
    print(f"   因子均值: {result_df['factor'].mean():.6f}")
    print(f"   因子标准差: {result_df['factor'].std():.6f}")
    print(f"{'='*60}\n")
    
    return result_df


def run_earnings_surprise_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2025-09-30',
    rebalance_freq: str = 'monthly',
    transaction_cost: float = 0.0003,
) -> dict:
    """
    执行季度盈利惊喜因子的回测
    
    Parameters
    ----------
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    rebalance_freq : str
        调仓频率 ('daily', 'weekly', 'monthly')
    transaction_cost : float
        单边交易成本
        
    Returns
    -------
    dict
        包含回测结果的字典
    """
    from backtest_engine.engine import BacktestEngine
    from backtest_engine.performance import PerformanceAnalyzer
    from scipy.stats import spearmanr
    
    print("\n" + "="*60)
    print("季度盈利惊喜因子回测")
    print("="*60)
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 计算因子
    factor_data = calculate_earnings_surprise_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
    )
    
    if factor_data is None or factor_data.empty:
        return {
            'portfolio_returns': None,
            'performance_metrics': None,
            'factor_data': None,
            'analysis_results': {}
        }
    
    # 初始化回测引擎
    engine = BacktestEngine(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date
    )
    
    # 执行Long-Only回测
    print("\n执行Long-Only回测...")
    portfolio_returns = engine.run_backtest(
        factor_data=factor_data,
        strategy_type='long_only',
        rebalance_freq=rebalance_freq,
        transaction_cost=transaction_cost
    )
    
    if portfolio_returns is None or portfolio_returns.empty:
        print("⚠️ 回测失败：未生成有效收益")
        return {
            'portfolio_returns': None,
            'performance_metrics': None,
            'factor_data': factor_data,
            'analysis_results': {}
        }
    
    # 计算业绩指标
    analyzer = PerformanceAnalyzer(portfolio_returns)
    metrics = analyzer.calculate_metrics()
    
    # IC分析
    print("\n计算IC指标...")
    ic_results = calculate_ic(data_manager, factor_data, start_date, end_date)
    
    # 输出结果
    print("\n" + "="*60)
    print("回测结果")
    print("="*60)
    print(f"\n📊 业绩指标:")
    print(f"  总收益率:     {metrics['total_return']:.2%}")
    print(f"  年化收益率:   {metrics['annualized_return']:.2%}")
    print(f"  年化波动率:   {metrics['volatility']:.2%}")
    print(f"  夏普比率:     {metrics['sharpe_ratio']:.3f}")
    print(f"  最大回撤:     {metrics['max_drawdown']:.2%}")
    
    if ic_results['ic_series'] is not None:
        print(f"\n📊 IC分析:")
        print(f"  IC均值:       {ic_results['ic_mean']:.4f}")
        print(f"  IC标准差:     {ic_results['ic_std']:.4f}")
        print(f"  ICIR:         {ic_results['icir']:.4f}")
        print(f"  IC>0占比:     {ic_results['ic_positive_ratio']:.2%}")
    
    print("="*60 + "\n")
    
    return {
        'portfolio_returns': portfolio_returns,
        'performance_metrics': metrics,
        'factor_data': factor_data,
        'analysis_results': ic_results
    }


def calculate_ic(data_manager, factor_data, start_date, end_date):
    """计算IC指标"""
    from scipy.stats import spearmanr
    
    # 加载收益率数据
    daily_data = data_manager.load_data('daily', start_date=start_date, end_date=end_date)
    
    if daily_data is None or daily_data.empty:
        return {
            'ic_series': None,
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'icir': np.nan,
            'ic_positive_ratio': np.nan
        }
    
    daily_data = daily_data.copy()
    daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'])
    
    # 计算收益率
    daily_data = daily_data.sort_values(['ts_code', 'trade_date'])
    daily_data['returns'] = daily_data.groupby('ts_code')['close'].pct_change()
    
    # 合并因子和收益率
    factor_reset = factor_data.reset_index()
    merged = factor_reset.merge(
        daily_data[['ts_code', 'trade_date', 'returns']],
        on=['ts_code', 'trade_date'],
        how='inner'
    )
    
    # 计算每日IC
    ic_series = merged.groupby('trade_date').apply(
        lambda x: spearmanr(x['factor'], x['returns'])[0] if len(x) > 10 else np.nan
    )
    ic_series = ic_series.dropna()
    
    if len(ic_series) == 0:
        return {
            'ic_series': None,
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'icir': np.nan,
            'ic_positive_ratio': np.nan
        }
    
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_positive_ratio = (ic_series > 0).sum() / len(ic_series)
    
    return {
        'ic_series': ic_series,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_ratio': ic_positive_ratio
    }


if __name__ == '__main__':
    # 测试因子计算
    results = run_earnings_surprise_backtest(
        start_date='2024-01-01',
        end_date='2025-09-30',
        rebalance_freq='monthly',
        transaction_cost=0.0003
    )
