import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List

# 路径：把项目根目录加入 sys.path，便于使用绝对包导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager


def calculate_pe_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    use_ttm: bool = True,
) -> pd.DataFrame:
    """
    计算市盈率 (P/E Ratio) 因子
    
    **因子逻辑**：
    PE = 股价 / 每股收益 (EPS)
    PE越低，说明估值越便宜，投资价值越高（价值投资逻辑）
    
    **因子方向**：
    - 低PE股票 → 高因子值（因子值 = -PE，便于统一做多高因子值）
    - 高PE股票 → 低因子值
    
    **数据来源**：
    可以使用两种方式计算：
    1. 使用 daily_basic 中的 pe_ttm（市盈率TTM，推荐）
    2. 使用 股价/每股收益 手动计算
    
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
    use_ttm : bool
        是否使用TTM（Trailing Twelve Months）市盈率
        True: 使用 daily_basic 中的 pe_ttm
        False: 手动计算 PE = 收盘价 / 每股收益

    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'.
        因子值为 -PE（负值），PE越低因子值越高。
        
    Notes
    -----
    - PE为负值（亏损股）的股票会被过滤掉
    - PE > 1000 的极端值会被过滤掉（可能是ST股或异常值）
    - 因子值取负是为了符合"做多高因子值"的统一逻辑
    """
    print(f"\n{'='*60}")
    print("市盈率 (P/E Ratio) 因子计算")
    print(f"{'='*60}")
    
    # 步骤1: 确定股票池
    if stock_codes is None:
        print("未指定股票池，使用全市场股票...")
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date)
        if all_daily is None or all_daily.empty:
            raise ValueError("无法获取日行情数据以确定股票池")
        stock_codes = all_daily['ts_code'].unique().tolist()
        print(f"✅ 股票池: {len(stock_codes)} 只股票")
    else:
        print(f"✅ 使用指定股票池: {len(stock_codes)} 只股票")
    
    # 步骤2: 加载数据
    if use_ttm:
        print("\n使用 daily_basic 中的 PE-TTM 数据...")
        daily_basic = data_manager.load_data(
            'daily_basic', 
            start_date=start_date, 
            end_date=end_date, 
            stock_codes=stock_codes
        )
        
        if daily_basic is None or daily_basic.empty:
            raise ValueError("无法加载 daily_basic 数据")
        
        # 检查是否有 pe_ttm 字段
        if 'pe_ttm' not in daily_basic.columns:
            raise ValueError("daily_basic 数据中缺少 pe_ttm 字段")
        
        daily_basic = daily_basic.copy()
        daily_basic['trade_date'] = pd.to_datetime(daily_basic['trade_date'])
        daily_basic = daily_basic.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        print(f"✅ 成功加载数据")
        print(f"   数据时间范围: {daily_basic['trade_date'].min()} ~ {daily_basic['trade_date'].max()}")
        print(f"   数据量: {len(daily_basic):,} 条记录")
        
        # 步骤3: 筛选有效PE值
        print("\n步骤: 筛选有效PE值...")
        
        # 过滤条件
        valid_mask = (
            (daily_basic['pe_ttm'].notna()) &           # PE不为空
            (daily_basic['pe_ttm'] > 0) &                # PE为正（排除亏损股）
            (daily_basic['pe_ttm'] < 1000)               # PE < 1000（排除极端值）
        )
        
        total_count = len(daily_basic)
        filtered_data = daily_basic[valid_mask].copy()
        filtered_count = len(filtered_data)
        
        print(f"  原始数据: {total_count:,} 条")
        print(f"  过滤后: {filtered_count:,} 条 (保留 {filtered_count/total_count*100:.1f}%)")
        print(f"  过滤掉: {total_count - filtered_count:,} 条")
        print(f"    - PE为空: {daily_basic['pe_ttm'].isna().sum():,} 条")
        print(f"    - PE<=0 (亏损股): {(daily_basic['pe_ttm'] <= 0).sum():,} 条")
        print(f"    - PE>=1000 (极端值): {(daily_basic['pe_ttm'] >= 1000).sum():,} 条")
        
        # 步骤4: 计算因子值
        print("\n步骤: 计算因子值...")
        # 因子值 = -PE，使得低PE对应高因子值
        filtered_data['factor'] = -filtered_data['pe_ttm']
        
        # PE统计信息
        pe_stats = filtered_data['pe_ttm'].describe()
        print(f"\nPE-TTM 统计信息:")
        print(f"  均值: {pe_stats['mean']:.2f}")
        print(f"  中位数: {pe_stats['50%']:.2f}")
        print(f"  标准差: {pe_stats['std']:.2f}")
        print(f"  最小值: {pe_stats['min']:.2f}")
        print(f"  25%分位: {pe_stats['25%']:.2f}")
        print(f"  75%分位: {pe_stats['75%']:.2f}")
        print(f"  最大值: {pe_stats['max']:.2f}")
        
        # 构建结果
        result = filtered_data[['trade_date', 'ts_code', 'factor']].copy()
        
    else:
        # 手动计算PE（备用方案）
        print("\n使用手动计算 PE = 股价 / EPS ...")
        
        # 加载日线数据（获取收盘价）
        daily = data_manager.load_data(
            'daily', 
            start_date=start_date, 
            end_date=end_date, 
            stock_codes=stock_codes
        )
        if daily is None or daily.empty:
            raise ValueError("无法获取日行情数据")
        
        daily = daily.copy()
        daily['trade_date'] = pd.to_datetime(daily['trade_date'])
        
        # 加载财务数据（获取每股收益）
        income = data_manager.load_data('income')
        if income is None or income.empty:
            raise ValueError("无法获取利润表数据")
        
        if stock_codes:
            income = income[income['ts_code'].isin(stock_codes)]
        
        # 计算每股收益（需要total_share数据）
        # 这里需要根据实际数据结构调整
        # 简化处理：使用basic_eps（基本每股收益）
        if 'basic_eps' not in income.columns:
            raise ValueError("利润表数据中缺少 basic_eps 字段")
        
        # 这部分实现较复杂，建议使用 use_ttm=True 的方式
        raise NotImplementedError("手动计算PE功能尚未完全实现，请使用 use_ttm=True")
    
    # 步骤5: 设置MultiIndex
    result = result.set_index(['trade_date', 'ts_code'])
    
    # 只保留在指定日期范围内的数据
    result = result.loc[result.index.get_level_values('trade_date') >= pd.to_datetime(start_date)]
    result = result.loc[result.index.get_level_values('trade_date') <= pd.to_datetime(end_date)]
    
    print(f"\n✅ 市盈率因子计算完成！")
    print(f"   有效记录数: {len(result):,}")
    print(f"   覆盖股票数: {result.index.get_level_values('ts_code').nunique()}")
    print(f"   覆盖交易日数: {result.index.get_level_values('trade_date').nunique()}")
    print(f"{'='*60}\n")
    
    return result


def run_pe_factor_backtest(
    start_date: str = '2020-01-01',
    end_date: str = '2023-12-31',
    stock_codes: Optional[List[str]] = None,
    use_ttm: bool = True,
    rebalance_freq: str = 'monthly',
    transaction_cost: float = 0.0003,
) -> dict:
    """
    运行市盈率因子回测
    
    **策略说明**：
    - 采用Long-Only策略
    - 做多低PE股票（高因子值）
    - 定期调仓
    
    Parameters
    ----------
    start_date, end_date : str
        回测周期
    stock_codes : Optional[List[str]]
        股票池，None则使用全市场
    use_ttm : bool
        是否使用PE-TTM
    rebalance_freq : str
        调仓频率: 'daily', 'weekly', 'monthly'
    transaction_cost : float
        单边交易成本
        
    Returns
    -------
    dict
        包含回测结果的字典:
        - factor_data: 因子数据
        - portfolio_returns: 组合收益
        - performance_metrics: 业绩指标
        - analysis_results: 分析结果（含IC）
    """
    print("=" * 60)
    print("市盈率 (P/E Ratio) 因子回测")
    print("=" * 60)
    print(f"\n回测配置:")
    print(f"  时间范围: {start_date} ~ {end_date}")
    print(f"  使用TTM: {use_ttm}")
    print(f"  调仓频率: {rebalance_freq}")
    print(f"  交易成本: {transaction_cost:.4f}")
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 计算因子
    factor_data = calculate_pe_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        use_ttm=use_ttm,
    )
    
    if factor_data.empty:
        print("⚠️ 因子数据为空，无法回测")
        return {
            'factor_data': None,
            'portfolio_returns': None,
            'performance_metrics': {},
            'analysis_results': {}
        }
    
    # 准备收益率数据
    stock_list = factor_data.index.get_level_values('ts_code').unique().tolist()
    
    stock_data = data_manager.load_data(
        'daily',
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_list
    )
    
    if stock_data is None or stock_data.empty:
        raise ValueError("无法加载用于回测的股票数据")
    
    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
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
    
    # Long-Only策略：等权持有所有有因子值的股票
    portfolio_returns = combined.groupby('trade_date')['next_return'].mean()
    
    # 模拟交易成本
    if rebalance_freq == 'daily':
        rebalance_dates = portfolio_returns.index
    else:
        freq_map = {'weekly': 'W-MON', 'monthly': 'MS'}
        rebalance_dates = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq=freq_map.get(rebalance_freq)
        )
    
    if rebalance_dates is not None and len(portfolio_returns) > 0:
        cost_impact = len(rebalance_dates) * transaction_cost / len(portfolio_returns)
        portfolio_returns -= cost_impact
    
    # 计算业绩指标
    cum_returns = (1 + portfolio_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1 if not cum_returns.empty else 0
    
    days = len(portfolio_returns)
    annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    max_drawdown = drawdown.min() if not drawdown.empty else 0
    
    # IC分析
    ic_series = None
    ic_mean = None
    ic_std = None
    icir = None
    ic_positive_ratio = None
    
    try:
        ic_list = []
        for date in combined['trade_date'].unique():
            date_data = combined[combined['trade_date'] == date]
            if len(date_data) >= 10:  # 至少需要10个样本
                # 计算因子与收益的相关性（Spearman相关系数）
                correlation = date_data[['factor', 'next_return']].corr(method='spearman').iloc[0, 1]
                if not np.isnan(correlation):
                    ic_list.append({'trade_date': date, 'ic': correlation})
        
        if ic_list:
            ic_series = pd.DataFrame(ic_list).set_index('trade_date')['ic']
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_ratio = (ic_series > 0).mean()
    except Exception as e:
        print(f"⚠️ IC计算失败: {e}")
    
    # 打印结果
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    
    print(f"\n📊 业绩指标:")
    print(f"  总收益率: {total_return:.2%}")
    print(f"  年化收益率: {annualized_return:.2%}")
    print(f"  年化波动率: {volatility:.2%}")
    print(f"  夏普比率: {sharpe_ratio:.2f}")
    print(f"  最大回撤: {max_drawdown:.2%}")
    
    if ic_series is not None:
        print(f"\n📊 IC分析:")
        print(f"  IC均值: {ic_mean:.4f}")
        print(f"  IC标准差: {ic_std:.4f}")
        print(f"  ICIR: {icir:.4f}")
        print(f"  IC>0占比: {ic_positive_ratio:.2%}")
    
    print(f"\n📈 因子覆盖:")
    print(f"  有效因子记录数: {len(factor_data)}")
    print(f"  覆盖股票数: {factor_data.index.get_level_values('ts_code').nunique()}")
    print(f"  覆盖交易日数: {factor_data.index.get_level_values('trade_date').nunique()}")
    
    return {
        'factor_data': factor_data,
        'portfolio_returns': portfolio_returns,
        'performance_metrics': {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
        },
        'analysis_results': {
            'ic_series': ic_series,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'ic_positive_ratio': ic_positive_ratio,
        }
    }


def main():
    """主函数：演示市盈率因子计算和回测"""
    print("=" * 60)
    print("市盈率 (P/E Ratio) 因子演示")
    print("=" * 60)
    
    try:
        # 配置参数
        config = {
            'start_date': '2015-01-01',
            'end_date': '2025-09-30',
            'use_ttm': True,
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
        }
        
        print("\n回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # 运行回测
        results = run_pe_factor_backtest(**config)
        
        print("\n✅ 回测完成！")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
