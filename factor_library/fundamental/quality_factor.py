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


def calculate_quality_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算质量因子（基于净资产收益率 ROE），用公告日(ann_date)对齐财务数据并向后匹配到交易日，避免前视偏差。
    
    ROE = 净利润 / 净资产
    ROE 代表的盈利能力与股票的未来收益呈显著正相关。

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
        因子值为 ROE，盈利能力越强，因子值越高。
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

    # 利润表数据（获取净利润）
    income = data_manager.load_data('income', cleaned=True)
    if income is None or income.empty:
        raise ValueError('无法获取利润表数据')
    if stock_codes:
        income = income[income['ts_code'].isin(stock_codes)]
    
    # 资产负债表数据（获取净资产）
    bs = data_manager.load_data('balancesheet', cleaned=True)
    if bs is None or bs.empty:
        raise ValueError('无法获取资产负债表数据')
    if stock_codes:
        bs = bs[bs['ts_code'].isin(stock_codes)]

    # 使用公告日作为对齐键
    # 利润表：提取净利润和公告日
    if 'ann_date' not in income.columns:
        income['ann_date'] = income.get('end_date')
    if 'n_income' not in income.columns:
        raise ValueError('利润表数据缺少 n_income（净利润）字段')
    
    income_ts = income[['ts_code', 'ann_date', 'n_income']].dropna(subset=['n_income']).copy()
    income_ts['ann_date'] = pd.to_datetime(income_ts['ann_date'], errors='coerce')
    if income_ts['ann_date'].isna().any():
        income_ts['ann_date'] = pd.to_datetime(income_ts['ann_date'].astype(str), format='%Y%m%d', errors='coerce')
    income_ts = income_ts.dropna(subset=['ann_date'])
    income_ts = income_ts.sort_values(['ts_code', 'ann_date']).reset_index(drop=True)

    # 资产负债表：提取净资产（total_hldr_eqy_exc_min_int）和公告日
    if 'ann_date' not in bs.columns:
        bs['ann_date'] = bs.get('end_date')
    
    # Tushare 中净资产字段通常为 total_hldr_eqy_exc_min_int（归属母公司股东权益）
    equity_field = None
    for field in ['total_hldr_eqy_exc_min_int', 'total_equity', 'total_assets']:
        if field in bs.columns:
            equity_field = field
            break
    
    if equity_field is None:
        raise ValueError('资产负债表数据缺少净资产相关字段')
    
    bs_ts = bs[['ts_code', 'ann_date', equity_field]].dropna(subset=[equity_field]).copy()
    bs_ts.rename(columns={equity_field: 'equity'}, inplace=True)
    bs_ts['ann_date'] = pd.to_datetime(bs_ts['ann_date'], errors='coerce')
    if bs_ts['ann_date'].isna().any():
        bs_ts['ann_date'] = pd.to_datetime(bs_ts['ann_date'].astype(str), format='%Y%m%d', errors='coerce')
    bs_ts = bs_ts.dropna(subset=['ann_date'])
    bs_ts = bs_ts.sort_values(['ts_code', 'ann_date']).reset_index(drop=True)

    # 先合并利润表和资产负债表的财务数据（按公告日对齐）
    financial = pd.merge(
        income_ts,
        bs_ts,
        on=['ts_code', 'ann_date'],
        how='inner'
    )
    
    # 计算 ROE = 净利润 / 净资产
    financial['roe'] = financial['n_income'] / financial['equity']
    # 过滤掉无效值
    financial = financial[np.isfinite(financial['roe'])]
    financial = financial[['ts_code', 'ann_date', 'roe']].copy()

    # 使用分组逐股进行 merge_asof，将财务数据（按公告日）匹配到交易日
    merged_parts = []
    daily_groups = daily.groupby('ts_code', sort=False)
    financial_groups = financial.groupby('ts_code', sort=False)
    common_codes = sorted(set(daily['ts_code'].unique()).intersection(financial['ts_code'].unique()))
    
    for code in common_codes:
        d = daily_groups.get_group(code).sort_values('trade_date').copy() if code in daily_groups.groups else None
        f = financial_groups.get_group(code).sort_values('ann_date').copy() if code in financial_groups.groups else None
        if d is None or f is None or d.empty or f.empty:
            continue
        
        # 使用 merge_asof 向后匹配：在交易日上找最近公告的财务数据
        part = pd.merge_asof(
            left=d,
            right=f[['ann_date', 'roe']],
            left_on='trade_date',
            right_on='ann_date',
            direction='backward',
        )
        merged_parts.append(part)
    
    merged = pd.concat(merged_parts, axis=0, ignore_index=True) if merged_parts else pd.DataFrame()
    if merged is None or merged.empty:
        raise ValueError('数据合并失败')

    merged = merged.dropna(subset=['roe'])
    if merged.empty:
        raise ValueError('所有记录都缺少 ROE 信息')

    # 构建因子数据
    factor = merged.set_index(['trade_date', 'ts_code'])[['roe']]
    factor.columns = ['factor']
    factor.index.names = ['trade_date', 'ts_code']
    return factor


def run_quality_factor_backtest(start_date: str = '2024-01-01',
                                end_date: str = '2024-02-29',
                                stock_codes: Optional[List[str]] = None,
                                rebalance_freq: str = 'weekly',
                                transaction_cost: float = 0.0003,
                                long_direction: str = 'high') -> dict:
    """
    使用 BacktestEngine 主路径运行质量因子策略回测，并集成 PerformanceAnalyzer 计算 IC。
    
    根据《因子投资：方法与实践》，盈利能力越强的公司，未来预期收益越高。
    因此默认 long_direction='high'，做多高 ROE 股票。
    
    Parameters
    ----------
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    stock_codes : Optional[List[str]]
        股票代码列表
    rebalance_freq : str
        调仓频率：'daily', 'weekly', 'monthly'
    transaction_cost : float
        单边交易费用，默认 0.03%
    long_direction : str
        多头方向：'high' 做多高ROE，'low' 做多低ROE
        
    Returns
    -------
    dict
        包含因子数据、组合收益、业绩指标和IC分析结果
    """
    # 初始化数据管理器
    data_manager = DataManager()

    # 使用 BacktestEngine 主路径
    from backtest_engine.engine import BacktestEngine
    
    # 计算质量因子
    print("\n" + "=" * 60)
    print("开始计算质量因子（ROE）...")
    factor_data = calculate_quality_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes
    )
    print(f"质量因子计算完成！共 {len(factor_data)} 条记录")
    print(f"因子值范围: [{factor_data['factor'].min():.4f}, {factor_data['factor'].max():.4f}]")
    print("=" * 60 + "\n")
    
    # 创建回测引擎
    engine = BacktestEngine(
        data_manager=data_manager,
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
    """主函数：演示质量因子计算和回测"""
    print("质量因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-01-01',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,  # 0.03% 交易费用
            'long_direction': 'high',  # 做多高 ROE 股票
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_quality_factor_backtest(**config)

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

        print("\n质量因子策略演示完成!")

    except Exception as e:
        print(f"演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
