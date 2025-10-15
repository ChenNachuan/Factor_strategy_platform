"""
规模因子计算模块

功能：计算股票的对数市值因子
方法：简化的函数式编程实现，避免不必要的类结构

作者：Factor Strategy Platform
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from typing import Optional, List

# 添加数据管理器路径
sys.path.append(str(Path(__file__).parent.parent.parent / 'data_manager'))
sys.path.append(str(Path(__file__).parent.parent.parent / 'backtest_engine'))

try:
    from data import DataManager
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保相关模块文件存在")


def calculate_size_factor(data_manager: DataManager,
                         start_date: str,
                         end_date: str,
                         stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    计算规模因子（对数市值）
    
    参数:
        data_manager: 数据管理器实例
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        stock_codes: 股票代码列表，None表示使用默认股票池
    
    返回:
        DataFrame: MultiIndex (trade_date, stock_code) 格式的因子数据
    """
    print(f"🧮 开始计算规模因子...")
    print(f"  时间范围: {start_date} 至 {end_date}")
    
    # 使用默认股票池
    if stock_codes is None:
        stock_codes = [
            '000001.SZ', '000002.SZ', '000858.SZ',
            '600000.SH', '600036.SH', '600519.SH'
        ]
    
    print(f"  股票数量: {len(stock_codes)}")
    
    try:
        # 获取股票日行情数据
        daily_data = data_manager.load_data(
            data_type='daily',
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes
        )
        
        if daily_data is None or daily_data.empty:
            raise ValueError("无法获取日行情数据")
        
        # 获取资产负债表数据以获取股本信息（加载所有数据）
        balance_data = data_manager.load_data(
            data_type='balancesheet',
            cleaned=True
        )
        
        if balance_data is None or balance_data.empty:
            raise ValueError("无法获取资产负债表数据")
        
        # 过滤目标股票并获取最新的股本数据
        if stock_codes:
            balance_data = balance_data[balance_data['ts_code'].isin(stock_codes)]
        
        # 获取每个股票的最新股本记录
        balance_data = balance_data.sort_values(['ts_code', 'end_date'])
        latest_shares = balance_data.groupby('ts_code').tail(1)[['ts_code', 'total_share']]
        
        # 过滤掉空值
        latest_shares = latest_shares.dropna(subset=['total_share'])
        
        # 合并数据获取总股本信息
        merged_data = daily_data.merge(
            latest_shares,
            left_on='ts_code',
            right_on='ts_code',
            how='left'
        )
        
        # 检查合并结果
        if merged_data.empty:
            raise ValueError("数据合并失败")
        
        # 过滤掉缺少股本信息的记录
        before_filter = len(merged_data)
        merged_data = merged_data.dropna(subset=['total_share'])
        after_filter = len(merged_data)
        
        if after_filter == 0:
            raise ValueError("所有记录都缺少股本信息")
        
        if before_filter != after_filter:
            print(f"  ⚠️ 过滤缺失股本数据: {before_filter} -> {after_filter} 条记录")
        
        # 计算市值 = 收盘价 × 总股本 / 10000 (单位：万元)
        merged_data['market_cap'] = merged_data['close'] * merged_data['total_share'] / 10000
        
        # 过滤掉市值异常的记录
        merged_data = merged_data[merged_data['market_cap'] > 0]
        
        # 计算对数市值因子
        merged_data['log_market_cap'] = np.log(merged_data['market_cap'])
        
        # 转换为MultiIndex格式 (注意：日行情数据中股票代码字段是ts_code)
        factor_data = merged_data.set_index(['trade_date', 'ts_code'])[['log_market_cap']]
        factor_data.columns = ['factor']
        
        # 重命名索引以符合标准格式 (date, stock_code)
        factor_data.index.names = ['date', 'stock_code']
        
        print(f"✅ 规模因子计算完成!")
        print(f"  最终数据量: {len(factor_data)} 条")
        print(f"  因子均值: {factor_data['factor'].mean():.4f}")
        print(f"  因子标准差: {factor_data['factor'].std():.4f}")
        
        return factor_data
        
    except Exception as e:
        print(f"❌ 规模因子计算失败: {e}")
        raise


def run_size_factor_backtest(start_date: str = '2024-01-01',
                           end_date: str = '2024-02-29',
                           stock_codes: Optional[List[str]] = None,
                           rebalance_freq: str = 'weekly',
                           transaction_cost: float = 0.0) -> dict:
    """
    运行规模因子策略回测
    
    参数:
        start_date: 回测开始日期
        end_date: 回测结束日期  
        stock_codes: 股票代码列表
        rebalance_freq: 调仓频率 ('daily', 'weekly', 'monthly')
        transaction_cost: 交易费用
    
    返回:
        dict: 包含回测结果的字典
    """
    print(f"🚀 开始规模因子策略回测...")
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 计算规模因子
    factor_data = calculate_size_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes
    )
    
    # 执行回测
    print(f"🎯 执行策略回测...")
    
    # 在函数内部导入以避免循环导入
    try:
        from engine import run_backtest
    except ImportError as e:
        print(f"无法导入回测引擎: {e}")
        raise
    
    portfolio_returns, positions = run_backtest(
        factor_data=factor_data,
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=rebalance_freq,
        transaction_cost=transaction_cost
    )
    
    # 计算基本业绩指标
    total_return = (portfolio_returns + 1).prod() - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = total_return / volatility if volatility > 0 else 0
    max_drawdown = (portfolio_returns.cumsum().expanding().max() - portfolio_returns.cumsum()).max()
    
    print(f"✅ 回测完成!")
    print(f"  总收益率: {total_return:.4f} ({total_return:.2%})")
    print(f"  年化波动率: {volatility:.4f} ({volatility:.2%})")
    print(f"  夏普比率: {sharpe_ratio:.4f}")
    print(f"  最大回撤: {max_drawdown:.4f} ({max_drawdown:.2%})")
    print(f"  调仓次数: {len(positions)}")
    
    # 尝试性能分析
    try:
        print(f"📊 执行性能分析...")
        
        # 在函数内部导入以避免循环导入
        from performance import PerformanceAnalyzer
        
        # 准备性能分析所需的数据
        # portfolio_returns 需要是DataFrame格式
        if isinstance(portfolio_returns, pd.Series):
            portfolio_df = pd.DataFrame({'strategy': portfolio_returns})
        else:
            portfolio_df = portfolio_returns
            
        # 创建master_data（包含next_day_return的数据）
        # 这里简化处理，创建一个基本的master_data
        master_data = pd.DataFrame({
            'date': portfolio_returns.index,
            'next_day_return': portfolio_returns.values
        })
        
        analyzer = PerformanceAnalyzer(
            portfolio_returns=portfolio_df,
            factor_data=factor_data,
            master_data=master_data
        )
        
        # 计算性能指标
        analyzer.calculate_metrics()
        
        # 显示结果
        if hasattr(analyzer, 'metrics') and analyzer.metrics is not None:
            print(f"📈 详细性能指标:")
            for col in analyzer.metrics.columns:
                metrics = analyzer.metrics[col]
                print(f"  {col}:")
                print(f"    年化收益: {metrics.get('annualized_return', 0):.4f}")
                print(f"    年化波动: {metrics.get('annualized_volatility', 0):.4f}")
                print(f"    夏普比率: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"    最大回撤: {metrics.get('max_drawdown', 0):.4f}")
        
        # 尝试IC分析
        try:
            analyzer.calculate_ic()
            if hasattr(analyzer, 'ic_series') and analyzer.ic_series is not None:
                ic_mean = analyzer.ic_series.mean()
                ic_std = analyzer.ic_series.std()
                icir = ic_mean / ic_std if ic_std > 0 else 0
                ic_positive_ratio = (analyzer.ic_series > 0).mean()
                
                print(f"🎯 IC分析结果:")
                print(f"  IC均值: {ic_mean:.4f}")
                print(f"  IC标准差: {ic_std:.4f}")
                print(f"  ICIR: {icir:.4f}")
                print(f"  IC>0比例: {ic_positive_ratio:.4f}")
        except Exception as ic_error:
            print(f"⚠️ IC分析失败: {ic_error}")
            
        analysis_results = {
            'performance_calculated': True,
            'ic_calculated': hasattr(analyzer, 'ic_series')
        }
            
    except Exception as e:
        print(f"⚠️ 性能分析失败: {e}")
        analysis_results = None
    
    return {
        'factor_data': factor_data,
        'portfolio_returns': portfolio_returns,
        'positions': positions,
        'performance_metrics': {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'rebalance_count': len(positions)
        },
        'analysis_results': analysis_results
    }


def main():
    """
    主函数：演示规模因子计算和回测
    """
    print("🎯 规模因子策略演示")
    print("=" * 50)
    
    try:
        # 配置参数
        config = {
            'start_date': '2024-01-01',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0  # 零手续费
        }
        
        print(f"📊 回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # 运行回测
        results = run_size_factor_backtest(**config)
        
        # 结果总结
        print(f"\n📋 回测结果总结:")
        metrics = results['performance_metrics']
        print(f"  🎯 策略表现: {metrics['sharpe_ratio']:.3f} (夏普比率)")
        print(f"  💰 总收益: {metrics['total_return']:.2%}")
        print(f"  📊 年化波动: {metrics['volatility']:.2%}")
        print(f"  📉 最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  🔄 调仓次数: {metrics['rebalance_count']}")
        
        print(f"\n🎉 规模因子策略演示完成!")
        
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        raise


if __name__ == "__main__":
    main()