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

def calculate_swma_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    period: int = 4,
) -> pd.DataFrame:
    """
    计算SWMA因子，使用对称加权移动平均。增加异常值处理。
    
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
        SWMA周期，默认为4

    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'
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
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 价格异常值过滤
    daily = daily[daily['close'] > 0]  # 过滤负数和0价格
    
    # 处理涨跌停
    daily['pct_chg'] = daily.groupby('ts_code')['close'].pct_change()
    daily = daily[abs(daily['pct_chg']) < 0.098]  # 过滤涨跌停

    # 计算SWMA因子
    result_parts = []
    for code in daily['ts_code'].unique():
        stock_data = daily[daily['ts_code'] == code].copy()
        if len(stock_data) < period:
            continue
            
        # 计算SWMA
        weights = np.array([1, 2, 2, 1])
        close_prices = stock_data['close'].values
        swma_values = np.convolve(close_prices, weights / weights.sum(), mode='valid')
        # 补充NaN使长度对齐
        padded_swma = np.concatenate([np.full(period-1, np.nan), swma_values])
        
        # 创建结果DataFrame
        result = pd.DataFrame({
            'trade_date': stock_data['trade_date'],
            'ts_code': code,
            'factor': padded_swma
        })
        result_parts.append(result)
    
    if not result_parts:
        raise ValueError('无法计算SWMA因子')
    
    # 合并所有结果
    factor = pd.concat(result_parts, axis=0, ignore_index=True)
    
    # 因子值异常值处理
    factor = factor.set_index(['trade_date', 'ts_code'])
    factor_values = factor['factor']
    
    # 去除极端值 (MAD方法)
    def remove_outliers_mad(x, n_sigmas=3.0):
        median = x.median()
        mad = np.median(np.abs(x - median))
        modified_zscore = 0.6745 * (x - median) / mad
        return x.mask(abs(modified_zscore) > n_sigmas)
    
    # 按日期分组进行异常值处理
    factor_values = factor_values.groupby(level=0).transform(remove_outliers_mad)
    
    # 标准化处理
    factor_values = factor_values.groupby(level=0).transform(lambda x: (x - x.mean()) / x.std())
    
    return pd.DataFrame(factor_values, columns=['factor'])

def run_swma_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    long_direction: str = 'high'
) -> dict:
    """
    使用 BacktestEngine 主路径运行SWMA因子策略回测。
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
        factor_name='factor',
    )
    
    # ...existing code...  # BacktestEngine相关的代码保持不变
    
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
    """主函数：演示SWMA因子计算和回测"""
    print("SWMA因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-09-30',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',  # SWMA策略通常做多高SWMA值
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_swma_factor_backtest(**config)

        # ...existing code...  # 结果展示部分保持不变

        print("\nSWMA因子策略演示完成!")

    except Exception as e:
        print(f"演示运行失败: {e}")
        raise

if __name__ == "__main__":
    main()
