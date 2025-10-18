# 盈利意外因子 (Earnings Surprise Factor)

## 因子说明

### 因子定义
基于季度财报的盈利意外因子，捕捉公司盈利相对去年同期的变化。

**公式**:
```
Factor_Value = (EPS_current - EPS_last_year_same_quarter) / Price_announcement_date-1
```

其中：
- `EPS_current`: 本期季度每股收益
- `EPS_last_year_same_quarter`: 去年同期季度每股收益
- `Price_announcement_date-1`: 财报公告日前一天的股价

### 因子逻辑

1. **对齐数据**：对于每一份新发布的季报，找到其去年同期的季报
   - 例如：2025年Q2 (end_date=2025-06-30) 与 2024年Q2 (end_date=2024-06-30)

2. **计算盈利差值**：
   ```
   EPS_diff = EPS_current - EPS_last_year_same_quarter
   ```

3. **标准化**：用公告日前一天的股价进行标准化
   ```
   Factor_Value = EPS_diff / Price
   ```

4. **因子应用**：
   - 在财报公告日 (ann_date) 当天更新因子值
   - 该因子值一直保留到下一份季报公告日
   - 形成每日更新的因子序列

### 理论基础

- **盈利增长信号**: 相对去年同期的EPS增长反映公司基本面改善
- **避免季节性**: 同比去年同期可以消除季节性影响
- **价格标准化**: 使不同股票间的因子值可比
- **PEAD效应**: Post-Earnings-Announcement-Drift，市场对盈利信息反应不足，存在动量效应

## 数据来源

### 输入数据

1. **利润表 (income)**:
   - `ts_code`: 股票代码
   - `ann_date`: 财报公告日
   - `end_date`: 报告期结束日
   - `basic_eps`: 基本每股收益

2. **日行情 (daily)**:
   - `trade_date`: 交易日期
   - `close`: 收盘价

### 输出数据

MultiIndex DataFrame (trade_date, ts_code) with column 'factor'
- 因子值为标准化后的EPS增长率
- 每日更新（在财报公告日变化）

## 使用方法

### 基本用法

```python
from factor_library.fundamental.earnings_surprise_factor import (
    calculate_earnings_surprise_factor,
    run_earnings_surprise_backtest
)
from data_manager.data import DataManager

# 计算因子
data_manager = DataManager()
factor_data = calculate_earnings_surprise_factor(
    data_manager=data_manager,
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# 执行回测
results = run_earnings_surprise_backtest(
    start_date='2020-01-01',
    end_date='2024-12-31',
    rebalance_freq='monthly',
    transaction_cost=0.0003
)

# 查看结果
print(results['performance_metrics'])
print(results['analysis_results'])
```

### 参数说明

#### calculate_earnings_surprise_factor()

- `data_manager`: DataManager实例
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)
- `stock_codes`: 股票代码列表，None则使用全市场
- `lookback_days`: 查找去年同期的天数范围（默认370天）
- `price_lag_days`: 使用公告日前N天的股价（默认1天）

#### run_earnings_surprise_backtest()

- `start_date`: 回测开始日期
- `end_date`: 回测结束日期
- `stock_codes`: 股票池
- `rebalance_freq`: 调仓频率 ('daily', 'weekly', 'monthly')
- `transaction_cost`: 单边交易成本

## 实现细节

### 去年同期匹配逻辑

```python
# 对于当前季报 (end_date = 2025-06-30)
# 查找去年同期 (target_end_date = 2024-06-30)
target_end_date = current_end_date - timedelta(days=365)

# 在±5天范围内查找最接近的季报
last_year_data = stock_income[
    (stock_income['end_date'] >= target_end_date - timedelta(days=5)) &
    (stock_income['end_date'] <= target_end_date + timedelta(days=5)) &
    (stock_income['ann_date'] < current_ann_date)
]
```

### 股价获取逻辑

```python
# 获取公告日前1天的股价
price_date = current_ann_date - timedelta(days=price_lag_days)
price_data = stock_daily[stock_daily['trade_date'] <= price_date]
price = price_data.iloc[-1]['close']  # 取最近的交易日
```

### 因子值向前填充

```python
# 在每个交易日，使用该日之前最近的一个财报因子值
for trade_date in all_dates:
    valid_factors = stock_factors[stock_factors['ann_date'] <= trade_date]
    if not valid_factors.empty:
        latest_factor = valid_factors.iloc[-1]
        # 使用最近的因子值
```

## 性能优化建议

### 计算速度优化

当前实现对每只股票单独处理，全市场约5000只股票需要较长时间。可以优化：

1. **向量化操作**：
```python
# 使用pandas的merge_asof进行时间对齐
factor_df = pd.merge_asof(
    current_data.sort_values('end_date'),
    last_year_data.sort_values('end_date'),
    on='end_date',
    by='ts_code',
    direction='nearest',
    tolerance=pd.Timedelta(days=5)
)
```

2. **并行处理**：
```python
from multiprocessing import Pool

def process_stock(ts_code):
    # 处理单只股票的因子计算
    pass

with Pool() as pool:
    results = pool.map(process_stock, stock_codes)
```

### 内存优化

```python
# 只加载需要的字段
income = data_manager.load_data(
    'income',
    cleaned=True,
    columns=['ts_code', 'ann_date', 'end_date', 'basic_eps']
)
```

## 回测策略

### Long-Only策略

- 等权持有所有有因子值的股票
- 定期调仓（monthly推荐）
- 扣除双边交易成本

### 可能的增强策略

1. **分位数选股**：
   - 只持有因子值最高的20-30%股票
   - 或按因子值分组，做多高组做空低组

2. **行业中性**：
   - 在每个行业内部进行因子排名
   - 避免行业集中度风险

3. **动态权重**：
   - 按因子值大小分配权重（高因子值高权重）
   - 或结合其他因子（如市值、流动性）

## 注意事项

### 数据质量

1. **财报时间对齐**：
   - 使用`ann_date`（公告日）而非`end_date`（报告期）
   - 避免前视偏差

2. **去年同期匹配**：
   - ±5天的容差可以处理报告期日期的微小差异
   - 确保必须在当前财报之前公告

3. **股价数据**：
   - 使用公告日前一天的股价
   - 处理停牌等特殊情况（取最近可用股价）

### 极端值处理

```python
# 可以添加winsorize处理
from scipy.stats.mstats import winsorize
factor_values = winsorize(factor_values, limits=[0.01, 0.01])
```

### 行业差异

不同行业的盈利增长特征不同：
- **周期股**: 盈利波动大，因子信号可能不稳定
- **成长股**: 持续增长，因子值可能长期为正
- **防御股**: 盈利稳定，因子值波动小

建议分行业分析或使用行业中性化。

## 文件位置

- 因子代码: `factor_library/fundamental/earnings_surprise_factor.py`
- 数据管理: `data_manager/data.py`
- 回测引擎: `backtest_engine/engine.py`

## 相关因子

- **PE因子** (`pe_factor.py`): 估值因子
- **ROE因子** (`quality_factor.py`): 质量因子
- **分析师预测因子**: 基于卖方预测的盈利意外

## 参考文献

1. Ball, R., & Brown, P. (1968). "An Empirical Evaluation of Accounting Income Numbers"
2. Bernard, V. L., & Thomas, J. K. (1989). "Post-Earnings-Announcement Drift"
3. Chan, L. K., Jegadeesh, N., & Lakonishok, J. (1996). "Momentum Strategies"

---

**Happy Factor Investing! 📊💰**
