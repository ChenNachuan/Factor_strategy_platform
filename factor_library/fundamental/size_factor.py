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


def calculate_size_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算规模因子（对数市值），用公告日(ann_date)对齐总股本并向后匹配到交易日，避免前视偏差。

    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'.
    """
    # 股票池
    if stock_codes is None:
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
        if all_daily is None or all_daily.empty:
            stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
        else:
            stock_codes = all_daily['ts_code'].unique().tolist()

    # 日线数据
    daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, stock_codes=stock_codes)
    if daily is None or daily.empty:
        raise ValueError('无法获取日行情数据')
    # 统一日期为 datetime 并排序
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        # 尝试按 YYYYMMDD 格式解析
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 资产负债表（总股本）
    bs = data_manager.load_data('balancesheet', cleaned=True)
    if bs is None or bs.empty:
        raise ValueError('无法获取资产负债表数据')
    if stock_codes:
        bs = bs[bs['ts_code'].isin(stock_codes)]

    # 使用公告日作为对齐键；若无 ann_date 则退化为 end_date
    if 'ann_date' not in bs.columns:
        bs['ann_date'] = bs.get('end_date')
    shares_ts = bs[['ts_code', 'ann_date', 'total_share']].dropna(subset=['total_share']).copy()
    shares_ts['ann_date'] = pd.to_datetime(shares_ts['ann_date'], errors='coerce')
    if shares_ts['ann_date'].isna().any():
        shares_ts['ann_date'] = pd.to_datetime(shares_ts['ann_date'].astype(str), format='%Y%m%d', errors='coerce')
    shares_ts = shares_ts.dropna(subset=['ann_date'])
    shares_ts = shares_ts.sort_values(['ts_code', 'ann_date']).reset_index(drop=True)

    # 使用分组逐股进行 merge_asof，避免全局排序检查造成的异常
    merged_parts = []
    daily_groups = daily.groupby('ts_code', sort=False)
    shares_groups = shares_ts.groupby('ts_code', sort=False)
    common_codes = sorted(set(daily['ts_code'].unique()).intersection(shares_ts['ts_code'].unique()))
    for code in common_codes:
        d = daily_groups.get_group(code).sort_values('trade_date').copy() if code in daily_groups.groups else None
        s = shares_groups.get_group(code).sort_values('ann_date').copy() if code in shares_groups.groups else None
        if d is None or s is None or d.empty or s.empty:
            continue
        # 确保 ts_code 列保留在结果中
        part = pd.merge_asof(
            left=d,
            right=s[['ann_date', 'total_share']],
            left_on='trade_date',
            right_on='ann_date',
            direction='backward',
        )
        # ts_code 已在 d 中，合并后自动保留
        merged_parts.append(part)
    merged = pd.concat(merged_parts, axis=0, ignore_index=True) if merged_parts else pd.DataFrame()
    if merged is None or merged.empty:
        raise ValueError('数据合并失败')

    merged = merged.dropna(subset=['total_share'])
    if merged.empty:
        raise ValueError('所有记录都缺少股本信息')

    merged['market_cap'] = merged['close'] * merged['total_share'] / 10000
    merged = merged[merged['market_cap'] > 0]
    merged['log_market_cap'] = np.log(merged['market_cap'])

    factor = merged.set_index(['trade_date', 'ts_code'])[['log_market_cap']]
    factor.columns = ['factor']
    factor.index.names = ['trade_date', 'ts_code']
    return factor


def run_size_factor_backtest(start_date: str = '2024-01-01',
                             end_date: str = '2024-02-29',
                             stock_codes: Optional[List[str]] = None,
                             rebalance_freq: str = 'weekly',
                             transaction_cost: float = 0.0003,
                             long_direction: str = 'high') -> dict:
    """
    使用 BacktestEngine 主路径运行规模因子策略回测，并集成 PerformanceAnalyzer 计算 IC。
    """
    # 初始化数据管理器
    data_manager = DataManager()

    # 使用 BacktestEngine 主路径
    from backtest_engine.engine import BacktestEngine
    engine = BacktestEngine(
        data_manager=data_manager,
        fee=transaction_cost,
        long_direction=long_direction,
        rebalance_freq=rebalance_freq,
        factor_name='factor',  # 显式指定因子列名
    )
    engine.prepare_data(start_date=start_date, end_date=end_date, stock_codes=stock_codes)
    portfolio_returns = engine.run()

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
    """主函数：演示规模因子计算和回测"""
    print("规模因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-09-30',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,  # 0.03% 交易费用
            'long_direction': 'low',
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_size_factor_backtest(**config)

        # 结果总结（基于 Long_Only）
        print("\n回测结果总结 (Long_Only):")
        metrics = results['performance_metrics']
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  总收益: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annualized_return']:.2%}")
        print(f"  年化波动: {metrics['volatility']:.2%}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  调仓次数: {metrics['rebalance_count']}")

        print("\n规模因子策略演示完成!")

    except Exception as e:
        print(f"演示运行失败: {e}")
        raise


if __name__ == "__main__":
    main()