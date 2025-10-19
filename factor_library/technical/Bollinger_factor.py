import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List

# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager

def calculate_bollinger_bands_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    window: int = 20,
    num_std: float = 2,
    stock_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算布林带因子，包括带宽(BB_Width)和%B指标
    
    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with columns ['bb_width', 'percent_b'].
    """
    # 股票池处理
    if stock_codes is None:
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
        if all_daily is None or all_daily.empty:
            stock_codes = ['000001.SZ', '000002.SZ']  # 默认股票池
        else:
            stock_codes = all_daily['ts_code'].unique().tolist()

    # 加载日线数据
    daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, stock_codes=stock_codes)
    if daily is None or daily.empty:
        raise ValueError('无法获取日行情数据')

    # 数据预处理
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date', 'close'])

    # 数据质量检查
    if daily.empty:
        raise ValueError('数据预处理后无有效记录')

    # 按股票分组计算布林带指标
    factor_dfs = []
    for code in daily['ts_code'].unique():
        stock_data = daily[daily['ts_code'] == code].sort_values('trade_date')
        if len(stock_data) < window:
            continue
            
        # 计算布林带指标
        middle = stock_data['close'].rolling(window=window).mean()
        std = stock_data['close'].rolling(window=window).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        # 计算因子值
        bb_width = (upper - lower) / middle  # 归一化带宽
        percent_b = (stock_data['close'] - lower) / (upper - lower)
        
        # 构建因子DataFrame
        factor_df = pd.DataFrame({
            'bb_width': bb_width,
            'percent_b': percent_b,
            'above_upper': stock_data['close'] > upper,
            'below_lower': stock_data['close'] < lower
        })
        factor_df['ts_code'] = code
        factor_df['trade_date'] = stock_data['trade_date']
        factor_dfs.append(factor_df)

    if not factor_dfs:
        raise ValueError('无法计算任何股票的布林带因子')

    # 合并所有股票的因子值
    factor = pd.concat(factor_dfs, axis=0)
    factor = factor.set_index(['trade_date', 'ts_code'])
    
    return factor

def run_bb_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    window: int = 20,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    factor_type: str = 'bb_width'  # 'bb_width' or 'percent_b'
) -> dict:
    """
    使用布林带因子进行回测
    """
    data_manager = DataManager()

    try:
        from backtest_engine.engine import BacktestEngine
        engine = BacktestEngine(
            data_manager=data_manager,
            fee=transaction_cost,
            long_direction='high' if factor_type == 'bb_width' else 'low',
            rebalance_freq=rebalance_freq,
            factor_name=factor_type,
        )
        
        # 准备数据
        engine.prepare_data(start_date=start_date, end_date=end_date, stock_codes=stock_codes)
        
        # 运行回测
        portfolio_returns = engine.run()
        
        # 计算性能指标
        series = portfolio_returns['Long_Only']
        cum = (1 + series).cumprod()
        perf_metrics = {
            'total_return': float(cum.iloc[-1] - 1),
            'annualized_return': float(cum.iloc[-1] ** (252 / len(series)) - 1),
            'volatility': float(series.std() * np.sqrt(252)),
            'max_drawdown': float((cum / cum.cummax() - 1).min()),
            'rebalance_count': len(engine._get_rebalance_dates()),
        }
        perf_metrics['sharpe_ratio'] = (
            perf_metrics['annualized_return'] / perf_metrics['volatility']
            if perf_metrics['volatility'] > 0 else 0.0
        )
        
        # 获取因子分析结果
        analyzer = engine.get_performance_analysis()
        analysis_results = {
            'metrics': analyzer.calculate_metrics(),
            'ic_series': analyzer.ic_series
        }
        
        return {
            'factor_data': engine.factor_data,
            'portfolio_returns': portfolio_returns,
            'performance_metrics': perf_metrics,
            'analysis_results': analysis_results,
        }
        
    except Exception as e:
        print(f"回测过程发生错误: {str(e)}")
        raise

def main():
    """主函数：演示布林带因子策略"""
    print("布林带因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-09-30',
            'end_date': '2025-09-30',
            'window': 20,
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'factor_type': 'bb_width'  # 使用带宽因子
        }

        print("\n回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_bb_factor_backtest(**config)

        # 展示结果
        print("\n回测结果摘要:")
        metrics = results['performance_metrics']
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  总收益率: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annualized_return']:.2%}")
        print(f"  年化波动: {metrics['volatility']:.2%}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  调仓次数: {metrics['rebalance_count']}")

        # 因子分析结果
        print("\nIC分析:")
        ic_mean = results['analysis_results']['ic_series'].mean()
        ic_std = results['analysis_results']['ic_series'].std()
        print(f"  IC均值: {ic_mean:.3f}")
        print(f"  IC标准差: {ic_std:.3f}")
        print(f"  IR比率: {(ic_mean/ic_std if ic_std > 0 else 0):.3f}")

        print("\n布林带因子策略演示完成!")

    except Exception as e:
        print(f"演示运行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
