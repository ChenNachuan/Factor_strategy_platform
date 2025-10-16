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


def calculate_rsi_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    lookback_period: int = 20,
    use_volume_weighted: bool = True,
) -> pd.DataFrame:
    """
    计算 RSI 因子（相对强弱指数），支持成交量加权版本。
    
    根据《如何基于RSI技术指标构建有效的选股因子》研究报告：
    - RSI 是一个反转因子，IC 值通常为负
    - RSI 值越低的股票，未来预期收益越高
    - 成交量配合的 RSI 因子效果更稳健

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
        RSI 计算的回看周期，默认 20 天
    use_volume_weighted : bool
        是否使用换手率加权，默认 True（推荐）

    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'.
        因子值为 RSI 或成交量加权 RSI，值越低表示超卖（反转机会）。
    """
    # 股票池
    if stock_codes is None:
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
        if all_daily is None or all_daily.empty:
            stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
        else:
            stock_codes = all_daily['ts_code'].unique().tolist()

    # 加载日线数据（需要足够的历史数据来计算 RSI）
    # 向前扩展日期以确保有足够的数据计算初始 RSI
    buffer_days = lookback_period * 3  # 预留足够的缓冲期
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
    
    # 统一日期为 datetime 并排序
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 确保有必要的字段
    if 'close' not in daily.columns:
        raise ValueError('日线数据缺少 close（收盘价）字段')
    
    if use_volume_weighted and 'turnover_rate' not in daily.columns:
        print("⚠️  警告：缺少 turnover_rate 字段，将使用非加权 RSI")
        use_volume_weighted = False

    # 计算每日 RSI
    def calculate_daily_rsi(group):
        """计算单个股票的 RSI"""
        df = group[['trade_date', 'close']].copy()
        
        # 计算收益率
        df['returns'] = df['close'].pct_change()
        
        # 分离涨跌
        df['gain'] = df['returns'].where(df['returns'] > 0, 0)
        df['loss'] = -df['returns'].where(df['returns'] < 0, 0)
        
        # 使用指数加权移动平均（EWM）计算平均涨跌幅
        avg_gain = df['gain'].ewm(span=lookback_period, adjust=False).mean()
        avg_loss = df['loss'].ewm(span=lookback_period, adjust=False).mean()
        
        # 计算 RS 和 RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df['daily_rsi'] = rsi
        
        return df[['trade_date', 'daily_rsi']]
    
    # 按股票分组计算 RSI
    print(f"\n计算 RSI 因子（回看周期: {lookback_period} 天）...")
    rsi_parts = []
    for code, group in daily.groupby('ts_code'):
        rsi_result = calculate_daily_rsi(group)
        rsi_result['ts_code'] = code
        rsi_parts.append(rsi_result)
    
    rsi_data = pd.concat(rsi_parts, axis=0, ignore_index=True)
    
    # 合并回原始数据
    daily = pd.merge(
        daily,
        rsi_data[['ts_code', 'trade_date', 'daily_rsi']],
        on=['ts_code', 'trade_date'],
        how='left'
    )
    
    if use_volume_weighted:
        print("使用换手率加权计算最终因子值...")
        
        # 成交量加权 RSI
        def calculate_volume_weighted_rsi(group):
            """使用换手率对 RSI 进行加权平均"""
            df = group[['trade_date', 'daily_rsi', 'turnover_rate']].copy()
            df = df.dropna()
            
            if len(df) < lookback_period:
                return df[['trade_date']].assign(vw_rsi=np.nan)
            
            # 滚动窗口加权平均
            def weighted_avg(window):
                rsi_values = df.loc[window.index, 'daily_rsi']
                weights = df.loc[window.index, 'turnover_rate']
                
                if weights.sum() == 0:
                    return np.nan
                return np.average(rsi_values, weights=weights)
            
            df['vw_rsi'] = df['daily_rsi'].rolling(
                window=lookback_period,
                min_periods=lookback_period
            ).apply(weighted_avg, raw=False)
            
            return df[['trade_date', 'vw_rsi']]
        
        # 按股票分组计算加权 RSI
        vw_rsi_parts = []
        for code, group in daily.groupby('ts_code'):
            vw_result = calculate_volume_weighted_rsi(group)
            vw_result['ts_code'] = code
            vw_rsi_parts.append(vw_result)
        
        vw_rsi_data = pd.concat(vw_rsi_parts, axis=0, ignore_index=True)
        
        # 合并加权 RSI
        daily = pd.merge(
            daily,
            vw_rsi_data[['ts_code', 'trade_date', 'vw_rsi']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        
        factor_col = 'vw_rsi'
    else:
        factor_col = 'daily_rsi'
    
    # 过滤到指定日期范围
    daily = daily[daily['trade_date'] >= start_date]
    
    # 构建因子数据
    factor_data = daily[['trade_date', 'ts_code', factor_col]].copy()
    factor_data = factor_data.dropna()
    
    # 设置 MultiIndex
    factor = factor_data.set_index(['trade_date', 'ts_code'])[[factor_col]]
    factor.columns = ['factor']
    factor.index.names = ['trade_date', 'ts_code']
    
    print(f"RSI 因子计算完成！共 {len(factor)} 条记录")
    
    return factor


def run_rsi_factor_backtest(start_date: str = '2024-01-01',
                            end_date: str = '2024-02-29',
                            stock_codes: Optional[List[str]] = None,
                            lookback_period: int = 20,
                            use_volume_weighted: bool = True,
                            rebalance_freq: str = 'weekly',
                            transaction_cost: float = 0.0003,
                            long_direction: str = 'low') -> dict:
    """
    使用 BacktestEngine 主路径运行 RSI 因子策略回测，并集成 PerformanceAnalyzer 计算 IC。
    
    根据研究报告，RSI 是反转因子，应做多低 RSI 股票（超卖反弹）。
    
    Parameters
    ----------
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    stock_codes : Optional[List[str]]
        股票代码列表
    lookback_period : int
        RSI 计算周期，默认 20 天
    use_volume_weighted : bool
        是否使用成交量加权，默认 True
    rebalance_freq : str
        调仓频率：'daily', 'weekly', 'monthly'
    transaction_cost : float
        单边交易费用，默认 0.03%
    long_direction : str
        多头方向：'low' 做多低RSI（推荐），'high' 做多高RSI
        
    Returns
    -------
    dict
        包含因子数据、组合收益、业绩指标和IC分析结果
    """
    # 初始化数据管理器
    data_manager = DataManager()

    # 使用 BacktestEngine 主路径
    from backtest_engine.engine import BacktestEngine
    
    # 计算 RSI 因子
    print("\n" + "=" * 60)
    factor_type = "成交量加权 RSI" if use_volume_weighted else "标准 RSI"
    print(f"开始计算 {factor_type} 因子...")
    
    factor_data = calculate_rsi_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        lookback_period=lookback_period,
        use_volume_weighted=use_volume_weighted
    )
    
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
    """主函数：演示 RSI 因子计算和回测"""
    print("RSI 因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-01-01',
            'end_date': '2029-09-30',
            'lookback_period': 20,
            'use_volume_weighted': True,  # 使用成交量加权版本
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,  # 0.03% 交易费用
            'long_direction': 'low',  # 做多低 RSI（超卖反弹）
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_rsi_factor_backtest(**config)

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
            print(f"\n💡 注意: RSI 是反转因子，IC 均值通常为负值（做多低RSI）")

        print("\nRSI 因子策略演示完成!")

    except Exception as e:
        print(f"演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
