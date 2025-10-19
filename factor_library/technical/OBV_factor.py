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


def calculate_obv_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算 OBV (On-Balance Volume) 因子，并进行标准化处理。
    
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
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 计算 OBV
    result_parts = []
    for code in daily['ts_code'].unique():
        stock_data = daily[daily['ts_code'] == code].sort_values('trade_date')
        
        # 计算价格变动方向
        price_change = stock_data['close'].diff()
        sign = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # 计算 OBV
        obv = (sign * stock_data['vol']).cumsum()
        
        # 创建结果 DataFrame
        result = pd.DataFrame({
            'trade_date': stock_data['trade_date'],
            'ts_code': code,
            'factor': obv
        })
        result_parts.append(result)
    
    # 合并所有股票的结果
    combined = pd.concat(result_parts, axis=0)
    
    # 对每个交易日的 OBV 进行截面标准化
    combined = combined.set_index(['trade_date', 'ts_code'])
    combined = combined.sort_index()
    
    # 按日期分组进行标准化
    grouped = combined.groupby('trade_date')
    combined['factor'] = grouped['factor'].transform(lambda x: (x - x.mean()) / x.std())
    
    return combined[['factor']]


def run_obv_factor_backtest(start_date: str = '2024-01-01',
                          end_date: str = '2024-02-29',
                          stock_codes: Optional[List[str]] = None,
                          rebalance_freq: str = 'weekly',
                          transaction_cost: float = 0.0003,
                          long_direction: str = 'high') -> dict:
    """
    使用 BacktestEngine 主路径运行 OBV 因子策略回测。
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
    
    # 使用 OBV 因子进行回测
    factor_data = calculate_obv_factor(data_manager, start_date, end_date, stock_codes)
    engine.prepare_data(start_date=start_date, end_date=end_date, stock_codes=stock_codes)
    engine.factor_data = factor_data  # 直接设置 OBV 因子数据
    portfolio_returns = engine.run()

    # 保持性能指标计算部分不变


def main():
    """主函数：演示 OBV 因子计算和回测"""
    print("OBV 因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-09-30',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',  # OBV 较高表示累积资金流入更多
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_obv_factor_backtest(**config)

        # 保持结果输出部分不变

    except Exception as e:
        print(f"演示运行失败: {e}")
        raise


if __name__ == "__main__":
    main()
