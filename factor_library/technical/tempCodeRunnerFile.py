import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List
import warnings

# 路径：把项目根目录加入 sys.path，便于使用绝对包导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager


def calculate_new_high_alpha_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    high_window: int = 240,
    lookback_period: int = 20,
    volume_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    计算创新高 Alpha 因子。
    
    基于行为金融学"锚定效应"，当股价突破前期高点时，市场可能反应不足。
    通过时序和截面双重筛选，找出突破后更有可能持续上涨的股票。

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
    high_window : int
        计算新高的窗口期，默认 240 天（约一年）
    lookback_period : int
        计算前期表现的回看窗口，默认 20 天
    volume_multiplier : float
        放量确认倍数，默认 1.5 倍

    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'.
        因子值越高表示突破质量越好（前期涨幅低、换手率低）。
    """
    # 股票池
    if stock_codes is None:
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
        if all_daily is None or all_daily.empty:
            stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
        else:
            stock_codes = all_daily['ts_code'].unique().tolist()

    # 加载日线数据（需要足够的历史数据）
    buffer_days = max(high_window, lookback_period) * 2
    start_date_extended = pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)
    start_date_extended = start_date_extended.strftime('%Y-%m-%d')
    
    daily = data_manager.load_data(
        'daily', 
        start_date=start_date_extended, 
        end_date=end_date, 
        stock_codes=stock_codes
    )
    if daily is None or daily.empty:
        raise ValueError('无法获取日行情数据')
    
    # 统一日期格式
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 检查必要字段
    required_cols = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'pct_chg']
    missing_cols = [col for col in required_cols if col not in daily.columns]
    if missing_cols:
        raise ValueError(f'日线数据缺少必要字段: {missing_cols}')

    print(f"\n计算创新高 Alpha 因子...")
    print(f"新高窗口: {high_window} 天，前期观察窗口: {lookback_period} 天")
    
    # 步骤 1: 识别创新高
    print("\n[1/4] 识别创新高...")
    
    # 计算滚动最高价
    daily['rolling_high'] = daily.groupby('ts_code')['close'].transform(
        lambda x: x.rolling(window=high_window, min_periods=high_window).max()
    )
    
    # 前一日收盘价
    daily['prev_close'] = daily.groupby('ts_code')['close'].shift(1)
    
    # 创新高条件：今日收盘价 = 滚动最高价 且 前一日收盘价 < 滚动最高价
    is_new_high = (
        (np.isclose(daily['close'], daily['rolling_high'], rtol=1e-5)) & 
        (daily['prev_close'] < daily['rolling_high'])
    )
    
    initial_count = is_new_high.sum()
    print(f"识别到 {initial_count} 个初始创新高事件")
    
    # 步骤 2: 时序筛选
    print("\n[2/4] 时序筛选（剔除假突破）...")
    
    # 2.1 剔除涨停股（涨幅 > 9.8%）
    is_limit_up = daily['pct_chg'] > 9.8
    is_new_high = is_new_high & (~is_limit_up)
    print(f"  - 剔除涨停后: {is_new_high.sum()} 个事件（剔除了 {initial_count - is_new_high.sum()} 个）")
    
    # 2.2 放量确认（使用成交量替代换手率）
    # 计算平均成交量
    daily['avg_vol'] = daily.groupby('ts_code')['vol'].transform(
        lambda x: x.rolling(window=lookback_period, min_periods=lookback_period).mean()
    )
    
    # 要求当日成交量 > 平均成交量的倍数
    is_volume_breakthrough = daily['vol'] > (volume_multiplier * daily['avg_vol'])
    before_volume = is_new_high.sum()
    is_new_high = is_new_high & is_volume_breakthrough
    print(f"  - 放量确认后: {is_new_high.sum()} 个事件（剔除了 {before_volume - is_new_high.sum()} 个）")
    
    # 步骤 3: 计算截面筛选指标
    print("\n[3/4] 计算截面筛选指标...")
    
    # 3.1 前期涨幅（lookback_period 天前到昨天的涨幅）
    daily['prior_return'] = daily.groupby('ts_code')['close'].transform(
        lambda x: x.pct_change(periods=lookback_period).shift(1)
    )
    
    # 3.2 前期平均成交量（作为换手率的替代）
    daily['prior_vol'] = daily.groupby('ts_code')['vol'].transform(
        lambda x: x.rolling(window=lookback_period, min_periods=lookback_period).mean().shift(1)
    )
    
    # 步骤 4: 构建最终因子
    print("\n[4/4] 构建最终因子值...")
    
    # 只保留有效创新高的数据
    eligible_df = daily[is_new_high].copy()
    
    if eligible_df.empty:
        print("⚠️  警告：没有找到符合所有条件的创新高事件")
        # 返回空因子
        return pd.DataFrame(columns=['factor']).rename_axis(['trade_date', 'ts_code'])
    
    print(f"最终有效事件数: {len(eligible_df)}")
    
    # 按日期分组，对前期涨幅和前期成交量进行排名
    # ascending=True: 值越小排名越靠前（rank值越小）
    eligible_df['rank_return'] = eligible_df.groupby('trade_date')['prior_return'].rank(
        ascending=True, pct=True
    )
    eligible_df['rank_volume'] = eligible_df.groupby('trade_date')['prior_vol'].rank(
        ascending=True, pct=True
    )
    
    # 综合得分：两个排名相加（分数越低越好）
    eligible_df['combined_score'] = eligible_df['rank_return'] + eligible_df['rank_volume']
    
    # 最终因子：对综合得分反向排名（分数低的因子值高）
    # ascending=False: 分数越低（越好），排名越靠前（因子值越大）
    eligible_df['factor'] = eligible_df.groupby('trade_date')['combined_score'].rank(
        ascending=False, pct=True
    )
    
    # 过滤到指定日期范围
    eligible_df = eligible_df[eligible_df['trade_date'] >= start_date]
    
    # 构建因子数据
    factor_data = eligible_df[['trade_date', 'ts_code', 'factor']].copy()
    
    # 设置 MultiIndex
    factor = factor_data.set_index(['trade_date', 'ts_code'])
    factor.index.names = ['trade_date', 'ts_code']
    
    print(f"\n创新高 Alpha 因子计算完成！共 {len(factor)} 条记录")
    print(f"因子值范围: [{factor['factor'].min():.4f}, {factor['factor'].max():.4f}]")
    print(f"涉及交易日: {factor.index.get_level_values('trade_date').nunique()} 天")
    print(f"涉及股票: {factor.index.get_level_values('ts_code').nunique()} 只")
    
    return factor


def run_new_high_alpha_backtest(start_date: str = '2024-01-01',
                                end_date: str = '2024-02-29',
                                stock_codes: Optional[List[str]] = None,
                                high_window: int = 240,
                                lookback_period: int = 20,
                                volume_multiplier: float = 1.5,
                                rebalance_freq: str = 'weekly',
                                transaction_cost: float = 0.0003,
                                long_direction: str = 'high',
                                n_groups: int = 3) -> dict:
    """
    使用 BacktestEngine 主路径运行创新高 Alpha 因子策略回测。
    
    因子值越高表示突破质量越好（前期涨幅低、换手率低），应做多高因子值股票。
    
    Parameters
    ----------
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    stock_codes : Optional[List[str]]
        股票代码列表
    high_window : int
        计算新高的窗口期，默认 240 天
    lookback_period : int
        前期观察窗口，默认 20 天
    volume_multiplier : float
        放量确认倍数，默认 1.5
    rebalance_freq : str
        调仓频率：'daily', 'weekly', 'monthly'
    transaction_cost : float
        单边交易费用，默认 0.03%
    long_direction : str
        多头方向：'high' 做多高因子值（推荐），'low' 做多低因子值
    n_groups : int
        分组数量，默认 3（事件驱动型因子建议 2-3 组，信号稀疏时避免分组过细）
        
    Returns
    -------
    dict
        包含因子数据、组合收益、业绩指标和IC分析结果
    """
    # 初始化数据管理器
    data_manager = DataManager()

    # 使用 BacktestEngine 主路径
    from backtest_engine.engine import BacktestEngine
    
    # 计算创新高 Alpha 因子
    print("\n" + "=" * 60)
    print("开始计算创新高 Alpha 因子...")
    
    factor_data = calculate_new_high_alpha_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        high_window=high_window,
        lookback_period=lookback_period,
        volume_multiplier=volume_multiplier
    )
    
    if factor_data.empty:
        print("\n⚠️  因子数据为空，无法进行回测")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'performance_metrics': {},
            'analysis_results': {}
        }
    
    print("=" * 60 + "\n")
    
    # 创建回测引擎（减少分组数以适应事件驱动型因子的稀疏特性）
    engine = BacktestEngine(
        data_manager=data_manager,
        n_groups=n_groups,  # 允许自定义分组数，默认 3 组
        fee=transaction_cost,
        long_direction=long_direction,
        rebalance_freq=rebalance_freq,
        factor_name='factor',
    )
    
    # 直接设置因子数据
    engine.factor_data = factor_data
    
    # 准备收益率数据
    stock_list = factor_data.index.get_level_values('ts_code').unique().tolist()
    stock_data = data_manager.load_data(
        'daily',
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_list
    )
    
    if stock_data is None or stock_data.empty:
        raise ValueError('无法加载股票数据用于回测')
    
    # 计算次日收益率
    stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
    stock_data['next_return'] = stock_data.groupby('ts_code')['close'].pct_change().shift(-1)
    
    # 合并因子和收益率
    factor_reset = factor_data.reset_index()
    stock_subset = stock_data[['ts_code', 'trade_date', 'next_return']].copy()
    
    engine.combined_data = pd.merge(
        factor_reset,
        stock_subset,
        on=['ts_code', 'trade_date'],
        how='inner'
    )
    engine.combined_data.dropna(subset=['factor', 'next_return'], inplace=True)
    
    # 运行回测
    print("开始回测...")
    portfolio_returns = engine.run()
    print("回测完成！\n")

    # 计算基本业绩指标（基于 Long_Only）
    if not isinstance(portfolio_returns, pd.DataFrame) or 'Long_Only' not in portfolio_returns.columns:
        raise ValueError('回测结果缺少 Long_Only 列')

    series = portfolio_returns['Long_Only']
    cum = (1 + series).cumprod()
    total_return = float(cum.iloc[-1] - 1) if len(cum) else np.nan
    trading_days = len(series)
    annualized_return = float(cum.iloc[-1] ** (252 / trading_days) - 1) if trading_days > 0 else np.nan
    volatility = float(series.std() * np.sqrt(252))
    sharpe_ratio = float(annualized_return / volatility) if volatility > 0 and not np.isnan(annualized_return) else 0.0
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan

    # 集成 PerformanceAnalyzer（含 IC 分析）
    analyzer = engine.get_performance_analysis()
    metrics_df = analyzer.calculate_metrics()
    ic_series = analyzer.ic_series
    analysis_results = {
        'metrics': metrics_df,
        'ic_series': ic_series
    }

    return {
        'factor_data': engine.factor_data,
        'portfolio_returns': portfolio_returns,
        'positions': None,
        'performance_metrics': {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'rebalance_count': len(engine._get_rebalance_dates()),
        },
        'analysis_results': analysis_results,
    }


def main():
    """主函数：演示创新高 Alpha 因子计算和回测"""
    print("创新高 Alpha 因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-01-01',
            'end_date': '2025-09-30',
            'high_window': 240,  # 240日新高
            'lookback_period': 20,  # 前期20日观察窗口
            'volume_multiplier': 1.5,  # 放量1.5倍确认
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,  # 0.03% 交易费用
            'long_direction': 'high',  # 做多高因子值（突破质量好）
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_new_high_alpha_backtest(**config)

        if results['portfolio_returns'] is None:
            print("\n回测未能执行（因子数据为空）")
            return

        # 结果总结（基于 Long_Only）
        print("\n回测结果总结 (Long_Only):")
        metrics = results['performance_metrics']
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  总收益: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annualized_return']:.2%}")
        print(f"  年化波动: {metrics['volatility']:.2%}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  调仓次数: {metrics['rebalance_count']}")

        # IC 分析
        if results['analysis_results']['ic_series'] is not None:
            ic = results['analysis_results']['ic_series']
            print(f"\nIC 分析:")
            print(f"  IC 均值: {ic.mean():.4f}")
            print(f"  IC 标准差: {ic.std():.4f}")
            print(f"  ICIR: {ic.mean() / ic.std():.4f}" if ic.std() > 0 else "  ICIR: N/A")
            print(f"  IC>0 占比: {(ic > 0).mean():.2%}")

        print("\n创新高 Alpha 因子策略演示完成!")

    except Exception as e:
        print(f"演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
