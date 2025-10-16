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

def calculate_rvi_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    period: int = 10
) -> pd.DataFrame:
    """
    计算RVI因子，基于开盘价、收盘价、最高价和最低价计算相对活力指标。

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

    # 日期处理
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    
    # 按股票分组计算RVI
    factor_data = []
    for code in stock_codes:
        stock_data = daily[daily['ts_code'] == code].sort_values('trade_date')
        if len(stock_data) < 4:  # RVI至少需要4个周期
            continue
            
        # 计算Vigor
        stock_data['Vigor'] = np.where(
            stock_data['high'] != stock_data['low'],
            (stock_data['close'] - stock_data['open']) / (stock_data['high'] - stock_data['low']),
            0.0
        )
        
        # 计算加权移动平均
        def weighted_ma_4(series):
            if len(series) < 4:
                return np.nan
            return (series.iloc[-4] + 2*series.iloc[-3] + 2*series.iloc[-2] + series.iloc[-1]) / 6
            
        stock_data['Num'] = stock_data['Vigor'].rolling(window=4).apply(weighted_ma_4, raw=False)
        stock_data['Den'] = stock_data['Num'].rolling(window=3).mean()
        stock_data['RVI'] = stock_data['Num'] / stock_data['Den']
        
        # 提取结果
        result = stock_data[['trade_date', 'ts_code', 'RVI']].dropna()
        factor_data.append(result)

    if not factor_data:
        raise ValueError('无法计算RVI因子')
        
    # 合并所有股票的因子数据
    factor = pd.concat(factor_data, ignore_index=True)
    factor = factor.set_index(['trade_date', 'ts_code'])['RVI']
    factor = factor.to_frame('factor')
    
    return factor

def run_rvi_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    long_direction: str = 'high'
) -> dict:
    """
    使用 BacktestEngine 运行RVI因子策略回测
    """
    # 初始化数据管理器
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
    engine.prepare_data(start_date=start_date, end_date=end_date, stock_codes=stock_codes)
    portfolio_returns = engine.run()

    # ...existing code from run_size_factor_backtest...
    
def main():
    """主函数：演示RVI因子计算和回测"""
    print("RVI因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-09-30',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',  # RVI通常高值表示强势
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_rvi_factor_backtest(**config)

        # ...existing code from main function...

    except Exception as e:
        print(f"演示运行失败: {e}")
        raise

if __name__ == "__main__":
    main()
