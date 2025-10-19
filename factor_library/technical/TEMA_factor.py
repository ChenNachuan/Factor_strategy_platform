import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class TEMAFactorStrategy:
    def __init__(self, 
                 price_data: pd.DataFrame,
                 tema_period: int = 20,
                 slope_period: int = 5,
                 momentum_period: int = 10,
                 short_n: int = 10,
                 long_n: int = 20):
        """
        初始化TEMA因子策略
        
        Parameters
        ----------
        price_data : pd.DataFrame
            股票价格数据，index为时间，columns为股票代码
        tema_period : int
            TEMA计算周期
        slope_period : int
            斜率计算周期
        momentum_period : int
            动量计算周期
        short_n : int
            短期TEMA周期
        long_n : int
            长期TEMA周期
        """
        self.price_data = price_data
        self.tema_period = tema_period
        self.slope_period = slope_period
        self.momentum_period = momentum_period
        self.short_n = short_n
        self.long_n = long_n
        
        # 设置日志
        self._setup_logging()
        
        # 数据质量检查
        self._check_data_quality()
        
        # 初始化因子数据容器
        self.factor_data = {}
        self.signal_data = None
        self.performance_metrics = {}
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TEMAFactorStrategy')
        
    def _check_data_quality(self):
        """数据质量检查"""
        try:
            # 检查缺失值
            if self.price_data.isnull().any().any():
                self.logger.warning("数据中存在缺失值，将进行处理")
                self.price_data = self.price_data.fillna(method='ffill')
                
            # 检查异常值
            z_scores = stats.zscore(self.price_data, nan_policy='omit')
            if (abs(z_scores) > 3).any().any():
                self.logger.warning("数据中存在异常值，建议进行处理")
                
        except Exception as e:
            self.logger.error(f"数据质量检查失败: {str(e)}")
            raise
            
    def calculate_tema(self, series: pd.Series, n: int) -> pd.Series:
        """计算TEMA"""
        try:
            e1 = series.ewm(span=n, adjust=False).mean()
            e2 = e1.ewm(span=n, adjust=False).mean()
            e3 = e2.ewm(span=n, adjust=False).mean()
            tema = 3 * e1 - 3 * e2 + e3
            return tema
        except Exception as e:
            self.logger.error(f"TEMA计算失败: {str(e)}")
            raise
            
    def calculate_factors(self):
        """计算所有TEMA因子"""
        try:
            # 计算TEMA乖离率因子
            self.factor_data['deviation'] = self.price_data.apply(
                lambda x: (x - self.calculate_tema(x, self.tema_period)) / 
                          self.calculate_tema(x, self.tema_period) * 100
            )
            
            # 计算TEMA斜率因子
            tema_values = self.price_data.apply(
                lambda x: self.calculate_tema(x, self.tema_period)
            )
            self.factor_data['slope'] = (tema_values - tema_values.shift(self.slope_period)) / self.slope_period
            
            # 计算TEMA动量因子
            self.factor_data['momentum'] = (tema_values - tema_values.shift(self.momentum_period)) / \
                                          tema_values.shift(self.momentum_period) * 100
                                          
            # 计算TEMA交叉信号因子
            short_tema = self.price_data.apply(lambda x: self.calculate_tema(x, self.short_n))
            long_tema = self.price_data.apply(lambda x: self.calculate_tema(x, self.long_n))
            self.factor_data['cross_signal'] = pd.DataFrame(
                np.where(short_tema > long_tema, 1,
                        np.where(short_tema < long_tema, -1, 0)),
                index=self.price_data.index,
                columns=self.price_data.columns
            )
            
        except Exception as e:
            self.logger.error(f"因子计算失败: {str(e)}")
            raise
            
    def generate_signals(self, 
                        deviation_threshold: float = 10.0,
                        slope_threshold: float = 0.1,
                        momentum_threshold: float = 5.0,
                        percentile: float = 0.2):
        """生成选股信号"""
        try:
            # 综合多个因子生成信号
            conditions = (
                (abs(self.factor_data['deviation']) < deviation_threshold) &
                (self.factor_data['slope'] > slope_threshold) &
                (self.factor_data['momentum'] > momentum_threshold) &
                (self.factor_data['cross_signal'] == 1)
            )
            
            self.signal_data = conditions.astype(int)
            
            # 每期选取排名靠前的股票
            for date in self.signal_data.index:
                mask = self.signal_data.loc[date] == 1
                if mask.sum() > len(self.signal_data.columns) * percentile:
                    # 使用动量因子排序
                    momentum_rank = self.factor_data['momentum'].loc[date][mask].rank(ascending=False)
                    top_n = int(len(self.signal_data.columns) * percentile)
                    self.signal_data.loc[date][mask] = 0
                    self.signal_data.loc[date][momentum_rank.index[:top_n]] = 1
                    
        except Exception as e:
            self.logger.error(f"信号生成失败: {str(e)}")
            raise
            
    def backtest(self, 
                 initial_capital: float = 1000000.0,
                 transaction_cost: float = 0.003):
        """回测策略"""
        try:
            # 计算收益率
            returns = self.price_data.pct_change()
            
            # 计算策略收益
            strategy_returns = (self.signal_data.shift(1) * returns).sum(axis=1) / \
                              self.signal_data.shift(1).sum(axis=1)
            
            # 计算累积收益
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            # 计算性能指标
            self.performance_metrics = {
                'Total Return': cumulative_returns.iloc[-1] - 1,
                'Annual Return': (cumulative_returns.iloc[-1] ** (252/len(cumulative_returns)) - 1),
                'Sharpe Ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
                'Max Drawdown': (cumulative_returns / cumulative_returns.cummax() - 1).min(),
                'Win Rate': (strategy_returns > 0).mean()
            }
            
            return strategy_returns, cumulative_returns
            
        except Exception as e:
            self.logger.error(f"回测失败: {str(e)}")
            raise
            
    def plot_results(self, cumulative_returns: pd.Series):
        """绘制回测结果"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 绘制累积收益曲线
            plt.plot(cumulative_returns.index, cumulative_returns.values)
            plt.title('Strategy Cumulative Returns')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            
            # 添加网格
            plt.grid(True)
            
            # 显示性能指标
            metrics_text = '\n'.join([
                f'{key}: {value:.2%}' for key, value in self.performance_metrics.items()
            ])
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"结果可视化失败: {str(e)}")
            raise
            
    def factor_analysis(self):
        """因子分析"""
        try:
            # 计算IC值
            ic_series = {}
            forward_returns = self.price_data.shift(-1) / self.price_data - 1
            
            for factor_name, factor_data in self.factor_data.items():
                ic_series[factor_name] = [
                    stats.spearmanr(factor_data.loc[date], forward_returns.loc[date])[0]
                    for date in factor_data.index
                ]
                ic_series[factor_name] = pd.Series(ic_series[factor_name], index=factor_data.index)
            
            # 绘制IC分析图
            plt.figure(figsize=(15, 10))
            for i, (factor_name, ic) in enumerate(ic_series.items(), 1):
                plt.subplot(2, 2, i)
                plt.hist(ic.dropna(), bins=50)
                plt.title(f'{factor_name} IC Distribution')
                plt.axvline(ic.mean(), color='r', linestyle='--')
                plt.text(0.02, 0.95, f'Mean IC: {ic.mean():.3f}\nIC IR: {ic.mean()/ic.std():.3f}',
                        transform=plt.gca().transAxes)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"因子分析失败: {str(e)}")
            raise
