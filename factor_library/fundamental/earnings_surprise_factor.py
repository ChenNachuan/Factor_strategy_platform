"""
盈利意外因子 (Earnings Surprise Factor) - 基于季度财报

因子逻辑：
-----------
1. 计算本期季度EPS与去年同期季度EPS的差值
2. 用财报公告日前一天的股价进行标准化
3. 因子值 = (EPS_current - EPS_last_year_same_quarter) / Price_announcement_date-1
4. 在财报公告日更新因子值，并保持到下一个财报公告日

理论基础：
-----------
- 盈利增长反映公司基本面改善
- 相对去年同期的增长避免季节性影响
- 股价标准化使不同股票间可比
- 基于Post-Earnings-Announcement-Drift (PEAD) 效应

数据来源：
-----------
- income: 利润表（获取季度EPS，字段：eps, ann_date, end_date）
- daily: 日行情（获取股价，字段：close, trade_date）

注意事项：
-----------
- 使用ann_date（公告日）而非end_date（报告期），避免前视偏差
- 因子值在财报发布当日可用，向后填充直到下一财报
- 去年同期定义：报告期end_date相差约365天的季报
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager
from backtest_engine.performance import PerformanceAnalyzer


def calculate_earnings_surprise_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    lookback_days: int = 370,  # 查找去年同期的天数范围（±5天）
    price_lag_days: int = 1,   # 使用公告日前N天的股价
) -> pd.DataFrame:
    """
    计算季度盈利意外因子
    
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
    lookback_days : int
        查找去年同期的天数中心值（默认370天，约一年+5天容差）
    price_lag_days : int
        使用公告日前N天的股价进行标准化（默认1天）
    
    Returns
    -------
    pd.DataFrame
        MultiIndex (trade_date, ts_code) 格式，包含 'factor' 列
        因子值 = (EPS_current - EPS_last_year) / Price
    """
    
    print("=" * 80)
    print("盈利意外因子 (Earnings Surprise Factor) - 计算开始")
    print("=" * 80)
    
    # 1. 获取股票池
    if stock_codes is None:
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
        if all_daily is None or all_daily.empty:
            print("⚠️  无法获取日行情数据，使用默认股票池")
            stock_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        else:
            stock_codes = all_daily['ts_code'].unique().tolist()
    
    if stock_codes:
        print(f"股票池大小: {len(stock_codes)} 只股票")
    
    # 2. 加载利润表数据（获取季度EPS）
    print("\n📊 加载利润表数据...")
    income = data_manager.load_data('income', cleaned=True)
    if income is None or income.empty:
        raise ValueError("❌ 无法获取利润表数据")
    
    # 筛选股票池
    if stock_codes:
        income = income[income['ts_code'].isin(stock_codes)].copy()
    
    # 检查必需字段
    required_fields = ['ts_code', 'ann_date', 'end_date']
    eps_field = 'basic_eps' if 'basic_eps' in income.columns else 'diluted_eps'
    
    if eps_field not in income.columns:
        raise ValueError(f"❌ 利润表缺少EPS字段 (basic_eps 或 diluted_eps)")
    
    required_fields.append(eps_field)
    missing_fields = [f for f in required_fields if f not in income.columns]
    if missing_fields:
        raise ValueError(f"❌ 利润表缺少必需字段: {missing_fields}")
    
    print(f"✅ 使用EPS字段: {eps_field}")
    
    # 提取季报数据（只保留季度财报）
    income_q = income[required_fields].copy()
    income_q = income_q.rename(columns={eps_field: 'eps'})
    income_q = income_q.dropna(subset=['eps', 'ann_date', 'end_date'])
    
    # 转换日期格式
    income_q['ann_date'] = pd.to_datetime(income_q['ann_date'], format='%Y%m%d', errors='coerce')
    income_q['end_date'] = pd.to_datetime(income_q['end_date'], format='%Y%m%d', errors='coerce')
    income_q = income_q.dropna(subset=['ann_date', 'end_date'])
    
    # 只保留在回测期间或之前公告的财报
    income_q = income_q[income_q['ann_date'] <= pd.to_datetime(end_date)]
    
    print(f"✅ 利润表数据加载完成")
    print(f"   原始记录数: {len(income)}")
    print(f"   有效季报数: {len(income_q)}")
    print(f"   覆盖股票数: {income_q['ts_code'].nunique()}")
    print(f"   公告日范围: {income_q['ann_date'].min()} 至 {income_q['ann_date'].max()}")
    
    # 3. 加载日行情数据（获取股价）
    print("\n📈 加载日行情数据...")
    # 需要扩展日期范围以获取公告日前的股价
    extended_start = (pd.to_datetime(start_date) - timedelta(days=30)).strftime('%Y-%m-%d')
    daily = data_manager.load_data('daily', start_date=extended_start, end_date=end_date, 
                                   stock_codes=stock_codes, cleaned=True)
    if daily is None or daily.empty:
        raise ValueError("❌ 无法获取日行情数据")
    
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date', 'close'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    print(f"✅ 日行情数据加载完成")
    print(f"   记录数: {len(daily)}")
    print(f"   覆盖股票数: {daily['ts_code'].nunique()}")
    print(f"   日期范围: {daily['trade_date'].min()} 至 {daily['trade_date'].max()}")
    
    # 4. 计算盈利意外因子
    print("\n🔧 计算盈利意外因子...")
    print(f"   参数: 去年同期范围 ±{lookback_days}天, 股价滞后 {price_lag_days}天")
    
    # 按股票代码分组处理
    factor_list = []
    stock_count = 0
    total_stocks = income_q['ts_code'].nunique()
    
    for ts_code in income_q['ts_code'].unique():
        stock_count += 1
        if stock_count % 100 == 0:
            print(f"   进度: {stock_count}/{total_stocks} ({stock_count/total_stocks*100:.1f}%)")
        
        # 获取该股票的所有季报
        stock_income = income_q[income_q['ts_code'] == ts_code].copy()
        stock_income = stock_income.sort_values('ann_date').reset_index(drop=True)
        
        # 获取该股票的日行情
        stock_daily = daily[daily['ts_code'] == ts_code].copy()
        if stock_daily.empty:
            continue
        
        # 对每一份季报，寻找去年同期季报并计算因子
        for idx, row in stock_income.iterrows():
            current_ann_date = row['ann_date']
            current_end_date = row['end_date']
            current_eps = row['eps']
            
            # 跳过公告日在回测期之前的财报（但我们需要它们作为"去年同期"）
            if current_ann_date < pd.to_datetime(start_date):
                continue
            
            # 寻找去年同期的季报（end_date相差约365天）
            target_end_date = current_end_date - timedelta(days=365)
            
            # 在±5天范围内查找去年同期
            last_year_data = stock_income[
                (stock_income['end_date'] >= target_end_date - timedelta(days=5)) &
                (stock_income['end_date'] <= target_end_date + timedelta(days=5)) &
                (stock_income['ann_date'] < current_ann_date)  # 必须在当前财报之前公告
            ]
            
            if last_year_data.empty:
                continue
            
            # 选择最接近的去年同期季报
            last_year_data = last_year_data.iloc[0]
            last_year_eps = last_year_data['eps']
            
            # 计算EPS差值
            eps_diff = current_eps - last_year_eps
            
            # 获取公告日前N天的股价
            price_date = current_ann_date - timedelta(days=price_lag_days)
            price_data = stock_daily[
                (stock_daily['trade_date'] <= price_date)
            ]
            
            if price_data.empty:
                continue
            
            # 取最近的交易日股价
            price_data = price_data.iloc[-1]
            price = price_data['close']
            
            if price <= 0:
                continue
            
            # 计算因子值
            factor_value = eps_diff / price
            
            # 记录因子值（在公告日当天生效）
            factor_list.append({
                'ts_code': ts_code,
                'ann_date': current_ann_date,
                'end_date': current_end_date,
                'eps_current': current_eps,
                'eps_last_year': last_year_eps,
                'eps_diff': eps_diff,
                'price': price,
                'factor': factor_value
            })
    
    if not factor_list:
        raise ValueError("❌ 未能计算出任何因子值，请检查数据")
    
    factor_df = pd.DataFrame(factor_list)
    
    print(f"\n✅ 因子计算完成")
    print(f"   有效因子记录数: {len(factor_df)}")
    print(f"   覆盖股票数: {factor_df['ts_code'].nunique()}")
    
    # 显示因子统计
    print(f"\n📊 因子值统计:")
    print(f"   均值: {factor_df['factor'].mean():.6f}")
    print(f"   中位数: {factor_df['factor'].median():.6f}")
    print(f"   标准差: {factor_df['factor'].std():.6f}")
    print(f"   最小值: {factor_df['factor'].min():.6f}")
    print(f"   最大值: {factor_df['factor'].max():.6f}")
    
    # 显示EPS差值统计
    print(f"\n📊 EPS差值统计:")
    print(f"   均值: {factor_df['eps_diff'].mean():.4f}")
    print(f"   中位数: {factor_df['eps_diff'].median():.4f}")
    print(f"   EPS增长(>0)占比: {(factor_df['eps_diff'] > 0).sum() / len(factor_df) * 100:.2f}%")
    
    # 5. 将因子值扩展到每个交易日（向后填充直到下一个财报）
    print("\n🔄 将因子值扩展到每个交易日...")
    
    # 获取所有交易日
    all_dates = daily[(daily['trade_date'] >= pd.to_datetime(start_date)) & 
                     (daily['trade_date'] <= pd.to_datetime(end_date))]['trade_date'].unique()
    all_dates = pd.Series(all_dates).sort_values().reset_index(drop=True)
    
    # 创建日期-股票网格
    factor_daily_list = []
    
    for ts_code in factor_df['ts_code'].unique():
        stock_factors = factor_df[factor_df['ts_code'] == ts_code].copy()
        stock_factors = stock_factors.sort_values('ann_date').reset_index(drop=True)
        
        for trade_date in all_dates:
            # 找到该日期之前最近的一个财报公告
            valid_factors = stock_factors[stock_factors['ann_date'] <= trade_date]
            
            if not valid_factors.empty:
                # 使用最近的因子值
                latest_factor = valid_factors.iloc[-1]
                factor_daily_list.append({
                    'trade_date': trade_date,
                    'ts_code': ts_code,
                    'factor': latest_factor['factor']
                })
    
    if not factor_daily_list:
        raise ValueError("❌ 未能生成日频因子数据")
    
    result = pd.DataFrame(factor_daily_list)
    
    # 转换为 MultiIndex 格式
    result = result.set_index(['trade_date', 'ts_code'])
    result = result.sort_index()
    
    print(f"\n✅ 日频因子数据生成完成")
    print(f"   记录数: {len(result)}")
    print(f"   覆盖股票数: {result.index.get_level_values('ts_code').nunique()}")
    print(f"   覆盖交易日数: {result.index.get_level_values('trade_date').nunique()}")
    
    print("\n" + "=" * 80)
    return result


def run_earnings_surprise_backtest(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    stock_codes: Optional[List[str]] = None,
    rebalance_freq: str = 'monthly',
    transaction_cost: float = 0.0003,
) -> dict:
    """
    执行盈利意外因子回测
    
    Parameters
    ----------
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    stock_codes : Optional[List[str]]
        股票池，None则使用全市场
    rebalance_freq : str
        调仓频率 ('daily', 'weekly', 'monthly')
    transaction_cost : float
        单边交易成本
        
    Returns
    -------
    dict
        包含回测结果的字典
    """
    print("\n" + "=" * 80)
    print("盈利意外因子回测 - 执行开始")
    print("=" * 80)
    print(f"回测参数:")
    print(f"  时间范围: {start_date} 至 {end_date}")
    print(f"  调仓频率: {rebalance_freq}")
    print(f"  交易成本: {transaction_cost:.4f}")
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 计算因子
    print("\n步骤 1: 计算盈利意外因子...")
    factor_data = calculate_earnings_surprise_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes
    )
    
    if factor_data is None or factor_data.empty:
        print("❌ 因子计算失败")
        return {
            'factor_data': None,
            'portfolio_returns': None,
            'performance_metrics': None,
            'analysis_results': {}
        }
    
    # 执行回测
    print("\n步骤 2: 执行回测...")
    
    # 准备收益率数据
    stock_list = factor_data.index.get_level_values('ts_code').unique().tolist()
    
    stock_data = data_manager.load_data(
        'daily',
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_list
    )
    
    if stock_data is None or stock_data.empty:
        raise ValueError("❌ 无法加载用于回测的股票数据")
    
    stock_data = stock_data.copy()
    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'], format='%Y%m%d', errors='coerce')
    stock_data = stock_data.dropna(subset=['trade_date'])
    stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
    
    # 计算下一日收益率
    stock_data['next_close'] = stock_data.groupby('ts_code')['close'].shift(-1)
    stock_data['next_return'] = (stock_data['next_close'] / stock_data['close']) - 1
    
    # 合并因子和收益数据
    combined = pd.merge(
        factor_data.reset_index(),
        stock_data[['trade_date', 'ts_code', 'next_return']],
        on=['trade_date', 'ts_code'],
        how='inner'
    )
    
    combined = combined.dropna(subset=['next_return'])
    
    if combined.empty:
        print("❌ 合并后无有效数据")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'performance_metrics': {},
            'analysis_results': {}
        }
    
    # Long-Only策略：等权持有所有有因子值的股票
    portfolio_returns = combined.groupby('trade_date')['next_return'].mean()
    
    # 模拟交易成本
    if rebalance_freq == 'daily':
        rebalance_dates = portfolio_returns.index
    else:
        freq_map = {'weekly': 'W-MON', 'monthly': 'MS'}
        rebalance_dates = pd.date_range(
            start=portfolio_returns.index.min(),
            end=portfolio_returns.index.max(),
            freq=freq_map.get(rebalance_freq, 'MS')
        )
        rebalance_dates = [d for d in rebalance_dates if d in portfolio_returns.index]
    
    # 在调仓日扣除交易成本
    portfolio_returns_with_cost = portfolio_returns.copy()
    for rebal_date in rebalance_dates:
        if rebal_date in portfolio_returns_with_cost.index:
            portfolio_returns_with_cost.at[rebal_date] -= transaction_cost * 2  # 双边成本
    
    # 计算业绩指标
    print("\n步骤 3: 计算业绩指标...")
    cum_returns = (1 + portfolio_returns_with_cost).cumprod()
    total_return = float(cum_returns.iloc[-1] - 1) if len(cum_returns) > 0 else 0.0
    ann_return = float(portfolio_returns_with_cost.mean() * 252)
    volatility = float(portfolio_returns_with_cost.std() * np.sqrt(252))
    sharpe = float(ann_return / volatility) if volatility > 0 else 0.0
    drawdowns = cum_returns / cum_returns.cummax() - 1
    max_dd = float(drawdowns.min())
    
    metrics = {
        'total_return': total_return,
        'annualized_return': ann_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }
    
    # 计算IC分析
    print("\n步骤 4: IC分析...")
    ic_analysis = calculate_ic_analysis(factor_data, data_manager, start_date, end_date)
    
    # 显示结果
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)
    print(f"\n📊 业绩指标:")
    print(f"  总收益率:     {metrics['total_return']:.2%}")
    print(f"  年化收益率:   {metrics['annualized_return']:.2%}")
    print(f"  年化波动率:   {metrics['volatility']:.2%}")
    print(f"  夏普比率:     {metrics['sharpe_ratio']:.3f}")
    print(f"  最大回撤:     {metrics['max_drawdown']:.2%}")
    
    if ic_analysis['ic_series'] is not None:
        print(f"\n📊 IC分析:")
        print(f"  IC均值:       {ic_analysis['ic_mean']:.4f}")
        print(f"  IC标准差:     {ic_analysis['ic_std']:.4f}")
        print(f"  ICIR:         {ic_analysis['icir']:.4f}")
        print(f"  IC>0占比:     {ic_analysis['ic_positive_ratio']:.2%}")
    
    print("\n" + "=" * 80)
    
    return {
        'factor_data': factor_data,
        'portfolio_returns': portfolio_returns_with_cost,
        'performance_metrics': metrics,
        'analysis_results': ic_analysis
    }


def calculate_ic_analysis(
    factor_data: pd.DataFrame,
    data_manager: DataManager,
    start_date: str,
    end_date: str
) -> dict:
    """
    计算因子IC分析
    
    Parameters
    ----------
    factor_data : pd.DataFrame
        因子数据，MultiIndex (trade_date, ts_code)
    data_manager : DataManager
        数据管理器
    start_date : str
        开始日期
    end_date : str
        结束日期
        
    Returns
    -------
    dict
        IC分析结果
    """
    try:
        # 加载收益率数据
        daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
        if daily is None or daily.empty:
            return {
                'ic_series': None,
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'icir': np.nan,
                'ic_positive_ratio': np.nan
            }
        
        daily = daily.copy()
        daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='%Y%m%d', errors='coerce')
        daily = daily.dropna(subset=['trade_date'])
        daily = daily.sort_values(['ts_code', 'trade_date'])
        
        # 计算未来收益率（T+1）
        daily['return'] = daily.groupby('ts_code')['close'].pct_change(1).shift(-1)
        
        # 合并因子和收益率
        factor_reset = factor_data.reset_index()
        merged = pd.merge(
            factor_reset,
            daily[['trade_date', 'ts_code', 'return']],
            on=['trade_date', 'ts_code'],
            how='inner'
        )
        
        if merged.empty:
            return {
                'ic_series': None,
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'icir': np.nan,
                'ic_positive_ratio': np.nan
            }
        
        # 计算每日IC（Spearman相关系数）
        ic_list = []
        for date in merged['trade_date'].unique():
            date_data = merged[merged['trade_date'] == date]
            if len(date_data) >= 10:  # 至少10只股票
                ic = date_data[['factor', 'return']].corr(method='spearman').iloc[0, 1]
                if not np.isnan(ic):
                    ic_list.append({'trade_date': date, 'ic': ic})
        
        if not ic_list:
            return {
                'ic_series': None,
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'icir': np.nan,
                'ic_positive_ratio': np.nan
            }
        
        ic_df = pd.DataFrame(ic_list)
        ic_series = ic_df.set_index('trade_date')['ic']
        
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 else np.nan
        ic_positive_ratio = (ic_series > 0).sum() / len(ic_series)
        
        return {
            'ic_series': ic_series,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'ic_positive_ratio': ic_positive_ratio
        }
    
    except Exception as e:
        print(f"⚠️  IC分析计算失败: {e}")
        return {
            'ic_series': None,
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'icir': np.nan,
            'ic_positive_ratio': np.nan
        }


if __name__ == '__main__':
    # 测试因子计算和回测
    results = run_earnings_surprise_backtest(
        start_date='2020-01-01',
        end_date='2024-12-31',
        rebalance_freq='monthly',
        transaction_cost=0.0003
    )
