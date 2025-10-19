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

def calculate_atr_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    period: int = 14
) -> pd.DataFrame:
    """
    计算ATR因子,使用过去period天的价格波动计算平均真实波幅。

    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    start_date : str
        起始日期
    end_date : str
        结束日期
    stock_codes : Optional[List[str]]
        股票代码列表,如果为None则使用所有可用股票
    period : int
        ATR计算周期,默认14天

    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'
    """
    # 股票池处理
    if stock_codes is None:
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
        if all_daily is None or all_daily.empty:
            stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
        else:
            stock_codes = all_daily['ts_code'].unique().tolist()

    # 加载日线数据
    daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, stock_codes=stock_codes)
    if daily is None or daily.empty:
        raise ValueError('无法获取日线行情数据')

    # 数据质量检查
    required_columns = ['trade_date', 'ts_code', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in daily.columns]
    if missing_columns:
        raise ValueError(f'日线数据缺少必要列: {missing_columns}')

    # 统一日期格式为datetime并排序
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    
    # 价格数据质量检查
    daily = daily[(daily['high'] > 0) & (daily['low'] > 0) & (daily['close'] > 0)]
    if daily.empty:
        raise ValueError('所有记录的价格数据无效')

    # 按股票分组计算ATR
    factor_parts = []
    for code in daily['ts_code'].unique():
        stock_data = daily[daily['ts_code'] == code].sort_values('trade_date')
        if len(stock_data) < period:
            continue
            
        # 计算True Range
        stock_data = stock_data.copy()
        stock_data['HL'] = stock_data['high'] - stock_data['low']
        stock_data['HC'] = abs(stock_data['high'] - stock_data['close'].shift(1))
        stock_data['LC'] = abs(stock_data['low'] - stock_data['close'].shift(1))
        stock_data['TR'] = stock_data[['HL', 'HC', 'LC']].max(axis=1)
        
        # 计算ATR
        stock_data['ATR'] = stock_data['TR'].rolling(window=period, min_periods=period).mean()
        
        # 标准化处理：使用过去period天的收盘价均值进行归一化
        stock_data['price_mean'] = stock_data['close'].rolling(window=period, min_periods=period).mean()
        stock_data['ATR_norm'] = stock_data['ATR'] / stock_data['price_mean']
        
        # 提取有效数据
        valid_data = stock_data[['trade_date', 'ts_code', 'ATR_norm']].dropna()
        factor_parts.append(valid_data)

    # 合并所有股票的ATR因子
    if not factor_parts:
        raise ValueError('没有足够的数据计算ATR因子')
    
    factor = pd.concat(factor_parts, axis=0)
    factor = factor.set_index(['trade_date', 'ts_code'])['ATR_norm']
    factor = pd.DataFrame(factor, columns=['factor'])
    
    return factor

def run_atr_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    long_direction: str = 'high',
    atr_period: int = 14
) -> dict:
    """ATR因子策略回测"""
    # 初始化数据管理器
    data_manager = DataManager()

    # 使用BacktestEngine主路径
    from backtest_engine.engine import BacktestEngine
    engine = BacktestEngine(
        data_manager=data_manager,
        fee=transaction_cost,
        long_direction=long_direction,
        rebalance_freq=rebalance_freq,
        factor_name='factor',
    )
    engine.prepare_data(start_date=start_date, end_date=end_date, stock_codes=stock_codes)
    portfolio_returns = engine.run()

    # 计算基本业绩指标
    if not isinstance(portfolio_returns, pd.DataFrame) or 'Long_Only' not in portfolio_returns.columns:
        raise ValueError('回测结果缺少Long_Only列')

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

    # 集成PerformanceAnalyzer
    analyzer = engine.get_performance_analysis()
    metrics_df = analyzer.calculate_metrics()
    ic_series = analyzer.ic_series

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
        'analysis_results': {
            'metrics': metrics_df,
            'ic_series': ic_series
        }
    }

def main():
    """主函数：演示ATR因子计算和回测"""
    print("ATR因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-09-30',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',  # ATR较大意味着波动性较大，可能预示趋势变化
            'atr_period': 14
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_atr_factor_backtest(**config)

        # 结果总结
        print("\n回测结果总结 (Long_Only):")
        metrics = results['performance_metrics']
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  总收益: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annualized_return']:.2%}")
        print(f"  年化波动: {metrics['volatility']:.2%}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  调仓次数: {metrics['rebalance_count']}")

        # 补充ATR特有的分析
        ic_mean = results['analysis_results']['ic_series'].mean()
        ic_std = results['analysis_results']['ic_series'].std()
        print(f"\nATR因子特征:")
        print(f"  IC均值: {ic_mean:.3f}")
        print(f"  IC标准差: {ic_std:.3f}")
        print(f"  IR比率: {(ic_mean/ic_std if ic_std > 0 else 0):.3f}")

        print("\nATR因子策略演示完成!")

    except Exception as e:
        print(f"演示运行失败: {e}")
        raise

if __name__ == "__main__":
    main()
