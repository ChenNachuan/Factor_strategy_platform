import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List
import statsmodels.api as sm

# 路径：把项目根目录加入 sys.path，便于使用绝对包导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager


def calculate_overnight_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    lookback_period: int = 20,
) -> pd.DataFrame:
    """
    计算新隔夜因子 (MIF, Market Inefficiency Factor)。
    
    根据国盛证券《如何将隔夜涨跌变为有效的选股因子?》研究报告：
    - MIF 因子刻画知情交易者的信息优势
    - IC 值通常为负（反转因子）
    - 因子值越低，未来预期收益越高

    因子构建步骤：
    1. 计算"隔夜涨跌幅绝对值"与"昨日换手率"的滚动相关系数
    2. 对相关系数因子进行市值中性化
    3. 对"隔夜跳空因子"(abs_overnight_ret_mean) 进行市值中性化
    4. 将步骤2的结果对步骤3的结果进行正交化（回归取残差）

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
    lookback_period : int
        滚动计算周期，默认 20 天

    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'.
        因子值为 MIF，值越低表示未来预期收益越高（反转因子）。
    """
    # 股票池
    if stock_codes is None:
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
        if all_daily is None or all_daily.empty:
            stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
        else:
            stock_codes = all_daily['ts_code'].unique().tolist()

    # 加载日线数据（需要足够的历史数据）
    buffer_days = lookback_period * 3
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
    required_cols = ['open', 'close', 'pre_close', 'vol', 'amount']
    missing_cols = [col for col in required_cols if col not in daily.columns]
    if missing_cols:
        raise ValueError(f'日线数据缺少必要字段: {missing_cols}')

    print(f"\n计算隔夜因子 (MIF)...")
    print(f"回看周期: {lookback_period} 天")
    
    # 步骤 1: 计算中间变量
    print("\n[1/5] 计算中间变量...")
    
    # 隔夜收益率 = 今日开盘价 / 昨日收盘价 - 1
    daily['overnight_ret'] = daily.groupby('ts_code')['open'].transform(
        lambda x: x / x.shift(1)
    ) - 1
    
    # 隔夜涨跌幅绝对值
    daily['abs_overnight_ret'] = daily['overnight_ret'].abs()
    
    # 计算换手率替代指标：成交量 / 成交量均值
    # 注意：这是一个简化版本，真实换手率 = 成交量 / 流通股本
    daily['turnover_proxy'] = daily.groupby('ts_code')['vol'].transform(
        lambda x: x / x.rolling(window=20, min_periods=1).mean()
    )
    
    # 昨日换手率（使用替代指标）
    daily['yesterday_turn'] = daily.groupby('ts_code')['turnover_proxy'].shift(1)
    
    # 对数市值替代：使用成交额作为市值的代理变量
    # 注意：这是简化方案，真实市值需要从基本面数据获取
    daily['log_market_cap'] = np.log(daily['amount'] + 1)  # 加1避免log(0)
    
    # 隔夜跳空因子均值（20日均值）
    daily['abs_overnight_ret_mean'] = daily.groupby('ts_code')['abs_overnight_ret'].transform(
        lambda x: x.rolling(window=lookback_period, min_periods=lookback_period).mean()
    )
    
    # 步骤 2: 计算滚动相关系数
    print(f"[2/5] 计算 {lookback_period} 日滚动相关系数...")
    
    def rolling_corr(group):
        """计算单个股票的滚动相关系数"""
        df = group[['trade_date', 'abs_overnight_ret', 'yesterday_turn']].copy()
        
        # 计算滚动相关系数
        corr_series = df['abs_overnight_ret'].rolling(
            window=lookback_period, 
            min_periods=lookback_period
        ).corr(df['yesterday_turn'])
        
        df['corr_factor'] = corr_series
        return df[['trade_date', 'corr_factor']]
    
    corr_parts = []
    for code, group in daily.groupby('ts_code'):
        corr_result = rolling_corr(group)
        corr_result['ts_code'] = code
        corr_parts.append(corr_result)
    
    corr_data = pd.concat(corr_parts, axis=0, ignore_index=True)
    
    # 合并回原始数据
    daily = pd.merge(
        daily,
        corr_data[['ts_code', 'trade_date', 'corr_factor']],
        on=['ts_code', 'trade_date'],
        how='left'
    )
    
    # 步骤 3 & 4: 市值中性化
    print("[3/5] 对相关系数因子进行市值中性化...")
    print("[4/5] 对隔夜跳空因子进行市值中性化...")
    
    def neutralize_by_market_cap(group, factor_col):
        """市值中性化：对因子进行市值回归，取残差"""
        clean_group = group[[factor_col, 'log_market_cap']].dropna()
        
        if clean_group.shape[0] < 2:
            return pd.Series(np.nan, index=group.index)
        
        Y = clean_group[factor_col]
        X = sm.add_constant(clean_group['log_market_cap'])
        
        try:
            model = sm.OLS(Y, X).fit()
            residuals = pd.Series(model.resid, index=clean_group.index)
            return residuals.reindex(group.index)
        except:
            return pd.Series(np.nan, index=group.index)
    
    # 按日期分组进行中性化（忽略 pandas 弃用警告）
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        daily['corr_factor_neutralized'] = daily.groupby('trade_date', group_keys=False).apply(
            lambda x: neutralize_by_market_cap(x, 'corr_factor')
        )
        
        daily['abs_overnight_ret_desize'] = daily.groupby('trade_date', group_keys=False).apply(
            lambda x: neutralize_by_market_cap(x, 'abs_overnight_ret_mean')
        )
    
    # 步骤 5: 正交化（回归取残差）
    print("[5/5] 对隔夜跳空因子进行正交化，生成 MIF...")
    
    def orthogonalize(group):
        """正交化：corr_factor_neutralized 对 abs_overnight_ret_desize 回归取残差"""
        clean_group = group[['corr_factor_neutralized', 'abs_overnight_ret_desize']].dropna()
        
        if clean_group.shape[0] < 2:
            return pd.Series(np.nan, index=group.index)
        
        Y = clean_group['corr_factor_neutralized']
        X = sm.add_constant(clean_group['abs_overnight_ret_desize'])
        
        try:
            model = sm.OLS(Y, X).fit()
            residuals = pd.Series(model.resid, index=clean_group.index)
            return residuals.reindex(group.index)
        except:
            return pd.Series(np.nan, index=group.index)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        daily['MIF'] = daily.groupby('trade_date', group_keys=False).apply(orthogonalize)
    
    # 过滤到指定日期范围
    daily = daily[daily['trade_date'] >= start_date]
    
    # 构建因子数据
    factor_data = daily[['trade_date', 'ts_code', 'MIF']].copy()
    factor_data = factor_data.dropna()
    
    # 设置 MultiIndex
    factor = factor_data.set_index(['trade_date', 'ts_code'])[['MIF']]
    factor.columns = ['factor']
    factor.index.names = ['trade_date', 'ts_code']
    
    print(f"\nMIF 因子计算完成！共 {len(factor)} 条记录")
    print(f"因子值范围: [{factor['factor'].min():.6f}, {factor['factor'].max():.6f}]")
    
    return factor


def run_overnight_factor_backtest(start_date: str = '2024-01-01',
                                  end_date: str = '2024-02-29',
                                  stock_codes: Optional[List[str]] = None,
                                  lookback_period: int = 20,
                                  rebalance_freq: str = 'weekly',
                                  transaction_cost: float = 0.0003,
                                  long_direction: str = 'low') -> dict:
    """
    使用 BacktestEngine 主路径运行隔夜因子策略回测，并集成 PerformanceAnalyzer 计算 IC。
    
    根据研究报告，MIF 是反转因子，应做多低 MIF 股票（信息优势较弱，被低估）。
    
    Parameters
    ----------
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    stock_codes : Optional[List[str]]
        股票代码列表
    lookback_period : int
        滚动计算周期，默认 20 天
    rebalance_freq : str
        调仓频率：'daily', 'weekly', 'monthly'
    transaction_cost : float
        单边交易费用，默认 0.03%
    long_direction : str
        多头方向：'low' 做多低MIF（推荐），'high' 做多高MIF
        
    Returns
    -------
    dict
        包含因子数据、组合收益、业绩指标和IC分析结果
    """
    # 初始化数据管理器
    data_manager = DataManager()

    # 使用 BacktestEngine 主路径
    from backtest_engine.engine import BacktestEngine
    
    # 计算 MIF 因子
    print("\n" + "=" * 60)
    print("开始计算隔夜因子 (MIF)...")
    
    factor_data = calculate_overnight_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        lookback_period=lookback_period
    )
    
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
    """主函数：演示隔夜因子计算和回测"""
    print("隔夜因子 (MIF) 策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'lookback_period': 20,
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,  # 0.03% 交易费用
            'long_direction': 'low',  # 做多低 MIF（反转因子）
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_overnight_factor_backtest(**config)

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
            print(f"\n💡 注意: MIF 是反转因子，IC 均值通常为负值（做多低MIF）")

        print("\n隔夜因子策略演示完成!")

    except Exception as e:
        print(f"演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
