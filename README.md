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
│   └── loader.py
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

## 免责声明

本项目仅为量化投资学习与研究目的，不构成任何投资建议。
