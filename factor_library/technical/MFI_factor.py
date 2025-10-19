import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List

# 项目路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager

def calculate_mfi_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    period: int = 14,
) -> pd.DataFrame:
    """
    计算 MFI (Money Flow Index) 因子

    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    start_date : str
        起始日期
    end_date : str
        结束日期
    stock_codes : Optional[List[str]]
        股票代码列表
    period : int
        MFI计算周期，默认14天

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

    # 日线数据获取与检查
    daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, stock_codes=stock_codes)
    if daily is None or daily.empty:
        raise ValueError('无法获取日行情数据')

    # 检查必要字段
    required_fields = ['high', 'low', 'close', 'vol', 'trade_date', 'ts_code']
    missing_fields = [field for field in required_fields if field not in daily.columns]
    if missing_fields:
        raise ValueError(f'日线数据缺少必要字段: {missing_fields}')

    # 日期处理与排序
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    if daily.empty:
        raise ValueError('日期处理后数据集为空')

    # 按股票分组计算MFI
    factor_parts = []
    for code in daily['ts_code'].unique():
        try:
            stock_data = daily[daily['ts_code'] == code].sort_values('trade_date').copy()
            if len(stock_data) < period:
                continue

            # MFI计算
            stock_data['TP'] = (stock_data['high'] + stock_data['low'] + stock_data['close']) / 3
            stock_data['MF'] = stock_data['TP'] * stock_data['vol']
            
            stock_data['TP_Diff'] = stock_data['TP'].diff()
            stock_data['Positive_MF'] = np.where(stock_data['TP_Diff'] > 0, stock_data['MF'], 0)
            stock_data['Negative_MF'] = np.where(stock_data['TP_Diff'] < 0, stock_data['MF'], 0)
            
            pos_sum = stock_data['Positive_MF'].rolling(window=period).sum()
            neg_sum = stock_data['Negative_MF'].rolling(window=period).sum()
            
            # 处理除零情况
            mr = pos_sum / neg_sum
            stock_data['MFI'] = 100 - (100 / (1 + mr))
            stock_data['MFI'] = np.where(neg_sum == 0, 100, stock_data['MFI'])
            stock_data['MFI'] = np.where(pos_sum == 0, 0, stock_data['MFI'])
            
            # 数据质量检查
            stock_data = stock_data.dropna(subset=['MFI'])
            if not stock_data.empty:
                factor_parts.append(stock_data[['trade_date', 'ts_code', 'MFI']])
                
        except Exception as e:
            print(f"处理股票 {code} 时发生错误: {str(e)}")
            continue

    if not factor_parts:
        raise ValueError('没有产生有效的因子数据')

    # 合并结果
    merged = pd.concat(factor_parts, axis=0)
    
    # 最终数据质量检查
    if merged.empty:
        raise ValueError('合并后的因子数据为空')
    if merged['MFI'].isna().all():
        raise ValueError('所有MFI值都是无效的')
    
    factor = merged.set_index(['trade_date', 'ts_code'])[['MFI']]
    factor.columns = ['factor']
    return factor

def run_mfi_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    long_direction: str = 'high'
) -> dict:
    """运行 MFI 因子策略回测"""
    try:
        data_manager = DataManager()
        
        # 使用 BacktestEngine
        from backtest_engine.engine import BacktestEngine
        engine = BacktestEngine(
            data_manager=data_manager,
            fee=transaction_cost,
            long_direction=long_direction,
            rebalance_freq=rebalance_freq,
            factor_name='factor',
        )
        
        # 计算因子并准备数据
        factor_data = calculate_mfi_factor(
            data_manager=data_manager,
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes
        )
        
        if factor_data.empty:
            raise ValueError('因子计算结果为空')
            
        engine.prepare_data(start_date=start_date, end_date=end_date, stock_codes=stock_codes)
        engine.factor_data = factor_data
        
        # 运行回测
        portfolio_returns = engine.run()
        
        # 性能指标计算
        if not isinstance(portfolio_returns, pd.DataFrame) or 'Long_Only' not in portfolio_returns.columns:
            raise ValueError('回测结果格式无效')

        series = portfolio_returns['Long_Only']
        cum = (1 + series).cumprod()
        
        metrics = {
            'total_return': float(cum.iloc[-1] - 1) if len(cum) else np.nan,
            'annualized_return': float(cum.iloc[-1] ** (252 / len(series)) - 1) if len(series) > 0 else np.nan,
            'volatility': float(series.std() * np.sqrt(252)),
            'sharpe_ratio': float((series.mean() * 252) / (series.std() * np.sqrt(252))) if series.std() > 0 else 0.0,
            'max_drawdown': float((cum / cum.cummax() - 1).min()) if not cum.empty else np.nan,
            'rebalance_count': len(engine._get_rebalance_dates()),
        }
        
        # 获取性能分析
        analyzer = engine.get_performance_analysis()
        analysis_results = {
            'metrics': analyzer.calculate_metrics(),
            'ic_series': analyzer.ic_series
        }
        
        return {
            'factor_data': factor_data,
            'portfolio_returns': portfolio_returns,
            'positions': None,
            'performance_metrics': metrics,
            'analysis_results': analysis_results,
        }
        
    except Exception as e:
        print(f"回测执行出错: {str(e)}")
        raise

def main():
    """主函数：演示MFI因子计算和回测"""
    print("MFI因子策略演示")
    print("=" * 50)

    try:
        config = {
            'start_date': '2015-09-30',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        results = run_mfi_factor_backtest(**config)

        print("\n回测结果总结 (Long_Only):")
        metrics = results['performance_metrics']
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  总收益: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annualized_return']:.2%}")
        print(f"  年化波动: {metrics['volatility']:.2%}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  调仓次数: {metrics['rebalance_count']}")

        # IC分析结果
        ic_metrics = results['analysis_results']['metrics']
        if not ic_metrics.empty:
            print("\nIC分析结果:")
            print(f"  IC均值: {ic_metrics['IC_Mean'].iloc[0]:.3f}")
            print(f"  IC标准差: {ic_metrics['IC_Std'].iloc[0]:.3f}")
            print(f"  IC_IR: {ic_metrics['IC_IR'].iloc[0]:.3f}")

        print("\nMFI因子策略演示完成!")

    except Exception as e:
        print(f"演示运行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
