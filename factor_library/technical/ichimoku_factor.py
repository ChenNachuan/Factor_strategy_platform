import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager

def calculate_ichimoku_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算基于Ichimoku云图的综合因子。
    
    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'.
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
        raise ValueError('无法获取日行情数据')

    # 数据清洗和预处理
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date', 'high', 'low', 'close'])
    
    # 数据质量检查
    required_cols = ['high', 'low', 'close']
    if not all(col in daily.columns for col in required_cols):
        raise ValueError(f'数据缺少必要列: {required_cols}')
    
    # 异常值处理
    daily = daily[daily['close'] > 0]
    daily = daily[daily['high'] >= daily['low']]

    # 按股票分组计算Ichimoku因子
    factor_results = []
    for code in daily['ts_code'].unique():
        stock_data = daily[daily['ts_code'] == code].sort_values('trade_date')
        if len(stock_data) < 52:  # 确保有足够的数据计算指标
            continue
            
        # 计算Ichimoku指标
        df = pd.DataFrame({
            'High': stock_data['high'],
            'Low': stock_data['low'],
            'Close': stock_data['close']
        })
        
        # 基础指标计算
        tenkan_sen = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        kijun_sen = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
        
        # 因子计算
        price_position = np.where(
            df['Close'] > senkou_span_a,
            np.where(df['Close'] > senkou_span_b, 2, 1),
            0
        )
        cloud_trend = (senkou_span_a > senkou_span_b).astype(int)
        tk_cross = np.where(tenkan_sen > kijun_sen, 1, -1)
        cloud_width = abs(senkou_span_a - senkou_span_b)
        cloud_width_momentum = cloud_width / cloud_width.rolling(window=20).mean()
        
        # 综合因子
        combined_factor = (
            price_position * 0.4 +
            cloud_trend * 0.3 +
            (tk_cross == 1).astype(int) * 0.2 +
            (cloud_width_momentum > 1).astype(int) * 0.1
        )
        
        # 保存结果
        factor_data = pd.DataFrame({
            'trade_date': stock_data['trade_date'],
            'ts_code': code,
            'factor': combined_factor
        })
        factor_results.append(factor_data)
    
    if not factor_results:
        raise ValueError('没有计算出有效的因子值')
    
    # 合并结果
    factor_df = pd.concat(factor_results, ignore_index=True)
    factor_df = factor_df.dropna(subset=['factor'])
    
    # 标准化处理
    factor_df['factor'] = factor_df.groupby('trade_date')['factor'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # 设置多重索引
    factor_df = factor_df.set_index(['trade_date', 'ts_code'])['factor']
    return factor_df.to_frame()

def run_ichimoku_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    long_direction: str = 'high'
) -> dict:
    """运行Ichimoku因子策略回测"""
    data_manager = DataManager()
    
    try:
        from backtest_engine.engine import BacktestEngine
        engine = BacktestEngine(
            data_manager=data_manager,
            fee=transaction_cost,
            long_direction=long_direction,
            rebalance_freq=rebalance_freq,
            factor_name='factor'
        )
        
        # 准备数据和运行回测
        engine.prepare_data(start_date=start_date, end_date=end_date, stock_codes=stock_codes)
        portfolio_returns = engine.run()
        
        # 计算性能指标
        if not isinstance(portfolio_returns, pd.DataFrame) or 'Long_Only' not in portfolio_returns.columns:
            raise ValueError('回测结果异常')
            
        series = portfolio_returns['Long_Only']
        cum_returns = (1 + series).cumprod()
        
        # 基础指标计算
        metrics = {
            'total_return': float(cum_returns.iloc[-1] - 1),
            'annualized_return': float(cum_returns.iloc[-1] ** (252 / len(series)) - 1),
            'volatility': float(series.std() * np.sqrt(252)),
            'max_drawdown': float((cum_returns / cum_returns.cummax() - 1).min()),
            'win_rate': float((series > 0).mean()),
            'rebalance_count': len(engine._get_rebalance_dates())
        }
        
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        
        # 性能分析
        analyzer = engine.get_performance_analysis()
        analysis_results = {
            'metrics': analyzer.calculate_metrics(),
            'ic_series': analyzer.ic_series,
            'turnover': analyzer.calculate_turnover() if hasattr(analyzer, 'calculate_turnover') else None
        }
        
        return {
            'factor_data': engine.factor_data,
            'portfolio_returns': portfolio_returns,
            'performance_metrics': metrics,
            'analysis_results': analysis_results
        }
        
    except Exception as e:
        print(f"回测执行失败: {str(e)}")
        raise

def main():
    """主函数：演示Ichimoku因子计算和回测"""
    print("Ichimoku云图因子策略演示")
    print("=" * 50)
    
    try:
        config = {
            'start_date': '2023-01-01',
            'end_date': '2024-02-29',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high'
        }
        
        print("\n回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
            
        results = run_ichimoku_factor_backtest(**config)
        
        print("\n回测结果总结:")
        metrics = results['performance_metrics']
        print(f"  总收益率: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annualized_return']:.2%}")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  胜率: {metrics['win_rate']:.2%}")
        print(f"  调仓次数: {metrics['rebalance_count']}")
        
        # IC分析结果
        if results['analysis_results']['ic_series'] is not None:
            ic = results['analysis_results']['ic_series']
            print(f"\nIC分析:")
            print(f"  IC均值: {ic.mean():.3f}")
            print(f"  IC标准差: {ic.std():.3f}")
            print(f"  IC_IR: {(ic.mean() / ic.std()):.3f}")
        
    except Exception as e:
        print(f"演示运行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
