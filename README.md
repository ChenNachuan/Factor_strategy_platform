# 多因子量化策略平台 (Multi-Factor Quant Strategy Platform)

这是一个为A股市场设计、模块化的多因子量化选股策略研究与回测框架。

## 项目架构

本项目采用关注点分离（Separation of Concerns）的设计原则，将策略的不同功能模块化，便于团队协作与功能扩展。

```
factor_strategy_platform/
│
├── configs/                  # 策略配置文件 (回测周期、因子权重等)
│   └── config.yaml
│
├── data_manager/             # 数据管理器 (负责数据的加载、清洗、对齐)
│   └── data_loader/
│   └── raw_data/
│   └── data_cleaner/
│   └── clean_data/
│
├── factor_library/           # 因子库 (存放所有单因子的计算逻辑)
│   ├── fundamental/          #  └─ 基础面因子
│   └── technical/            #  └─ 技术面因子
│
├── alpha_model/              # Alpha模型 (负责将多因子融合成最终选股分数)
│   └── combiner.py
│
├── portfolio_constructor/    # 组合构建器 (根据分数构建目标持仓)
│   └── builder.py
│
├── risk_manager/             # 风险管理器 (负责计算和配置对冲仓位)
│   └── hedger.py
│
├── backtest_engine/          # 回测引擎 (驱动回测流程，计算并展示业绩)
│   ├── engine.py
│   └── performance.py
│
├── notebooks/                # 研究与探索模块 (用于因子的探索性分析)
│
├── main.py                   # 主程序入口
└── requirements.txt          # 项目依赖库
```

## 核心工作流程

1.  **数据层 (`data_manager`)**: 提供干净、准确的Point-in-Time数据。
2.  **因子层 (`factor_library`)**: 独立计算各类原始因子值。
3.  **模型层 (`alpha_model`)**: 将多个原始因子值融合成最终的Alpha Score。
4.  **执行层 (`portfolio_constructor` & `risk_manager`)**: 根据Alpha Score构建股票多头组合，并计算对冲仓位。
5.  **调度层 (`backtest_engine`)**: 驱动整个流程，在每个调仓周期循环执行，并进行业绩分析。

## 快速运行

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **修改配置**:
    编辑 `configs/config.yaml` 文件，设定你的回测参数。

3.  **启动回测**:
    ```bash
    python main.py
    ```

1. 更新 README.md 文件
请将以下内容添加到你的 README.md 文件末尾。这部分内容将作为新成员的入门指南。

如何添加并回测新因子 (How to Add and Backtest a New Factor)
本框架的核心优势在于其模块化设计。遵循以下三个步骤，你可以轻松地将一个新的选股因子集成到平台中，并验证其有效性。

步骤 1: 创建因子文件
进入 factor_library 文件夹，根据你的因子类型（如基础面 fundamental 或技术面 technical），在相应的子文件夹下创建一个新的 .py 文件。例如，我们要创建一个新的估值因子，可以新建 factor_library/fundamental/valuation_factor.py。

步骤 2: 编写因子类
打开你新建的文件，仿照 size_factor.py 的结构，编写你的因子计算类。这个类必须包含两个核心方法：

__init__(self, master_data): 构造函数，接收经过 loader.py 清洗后的主数据框 master_data 作为输入。

calculate_factor(self): 核心计算函数，不接收额外参数。它必须返回一个 pandas.DataFrame，其中索引为 ['date', 'stock_code']，并且包含一个以你的因子命名的列。

模板示例 (valuation_factor.py):

Python

### factor_library/fundamental/valuation_factor.py
import pandas as pd

class ValuationFactor:
    """
    计算估值因子（以市盈率倒数 E/P 为例）。
    《因子投资：方法与实践》将这类因子归为价值类因子。
    """
    def __init__(self, master_data):
        """
        初始化估值因子计算。

        Args:
            master_data (pd.DataFrame): 必须包含 'pe_ttm' (滚动市盈率) 字段。
        """
        if 'pe_ttm' not in master_data.columns:
            raise ValueError("错误: 输入数据中缺少 'pe_ttm' 字段。")
        self.master_data = master_data
        self.factor_name = 'ep_ratio'

    def calculate_factor(self):
        """
        计算 E/P 因子 (市盈率的倒数)。
        我们选择做多 E/P 更高的股票，即市盈率更低的股票。
        """
        print(f"\\n[因子计算] 正在计算 {self.factor_name} (估值因子)...")
        factor_data = self.master_data[['pe_ttm']].copy()

        # 计算市盈率的倒数，并处理无穷大值
        factor_data[self.factor_name] = 1 / factor_data['pe_ttm']
        factor_data.replace([float('inf'), -float('inf')], None, inplace=True)
        factor_data.dropna(inplace=True)

        print(f"{self.factor_name} 计算完成！")
        return factor_data[[self.factor_name]]

步骤 3: 在 notebooks 中进行回测
在 notebooks 文件夹中创建一个新的 Jupyter Notebook 是验证因子的最佳方式。在这里，你可以清晰地完成数据加载、因子计算和回测分析的全过程。具体流程请参考我们提供的 市值因子分析示例.ipynb。

## 免责声明

本项目仅为量化投资学习与研究目的，不构成任何投资建议。
