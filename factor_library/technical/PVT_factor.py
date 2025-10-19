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

def calculate_pvt_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算PVT因子，基于价格和成交量的变化趋势。

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

    # 数据质量检查
    required_columns = ['trade_date', 'ts_code', 'close', 'vol']
    missing_columns = [col for col in required_columns if col not in daily.columns]
    if missing_columns:
        raise ValueError(f'数据缺少必要列: {missing_columns}')

    # 统一日期为 datetime 并排序
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    
    # 数据有效性检查
    if daily['close'].isna().any() or daily['vol'].isna().any():
        print("警告：存在缺失的价格或成交量数据，将被删除")
        daily = daily.dropna(subset=['close', 'vol'])
    
    if daily.empty:
        raise ValueError('清洗后数据为空')

    # 按股票分组计算PVT
    factor_parts = []
    for code in daily['ts_code'].unique():
        stock_data = daily[daily['ts_code'] == code].sort_values('trade_date')
        
        # 计算收盘价涨跌幅
        close_return = stock_data['close'].pct_change()
        
        # 计算PVT
        pvt = (stock_data['vol'] * close_return).fillna(0).cumsum()
        
        # 标准化处理
        pvt_std = (pvt - pvt.mean()) / pvt.std() if len(pvt) > 1 else pvt
        
        factor_part = pd.DataFrame({
            'trade_date': stock_data['trade_date'],
            'ts_code': code,
            'factor': pvt_std
        })
        factor_parts.append(factor_part)

    # 合并所有股票的因子值
    factor = pd.concat(factor_parts, axis=0)
    factor = factor.set_index(['trade_date', 'ts_code'])['factor']
    factor = pd.DataFrame(factor)
    
    return factor

def run_pvt_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    long_direction: str = 'high'
) -> dict:
    """使用 BacktestEngine 运行PVT因子策略回测"""
    # ...existing code...
    # 与 size_factor.py 中的 run_size_factor_backtest 完全相同
    # 只需将调用 calculate_size_factor 改为 calculate_pvt_factor

def main():
    """主函数：演示PVT因子计算和回测"""
    print("PVT因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-09-30',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',  # PVT通常是趋势指标，高PVT表示上升趋势
        }

        print("回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_pvt_factor_backtest(**config)

        # 结果总结
        print("\n回测结果总结 (Long_Only):")
        metrics = results['performance_metrics']
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  总收益: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annualized_return']:.2%}")
        print(f"  年化波动: {metrics['volatility']:.2%}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  调仓次数: {metrics['rebalance_count']}")

        # 输出IC分析结果
        print("\nIC分析结果:")
        ic_metrics = results['analysis_results']['metrics']
        print(f"  IC均值: {ic_metrics['IC_Mean'].iloc[0]:.3f}")
        print(f"  ICIR: {ic_metrics['ICIR'].iloc[0]:.3f}")
        print(f"  IC>0占比: {ic_metrics['IC_Positive_Ratio'].iloc[0]:.2%}")

        print("\nPVT因子策略演示完成!")

    except Exception as e:
        print(f"演示运行失败: {e}")
        raise

if __name__ == "__main__":
    main()
