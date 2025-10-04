# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Filename: size_factor.py
# Description: 计算规模因子 (Size Factor)
#              这是基于现有数据最直接、最经典的因子之一。
# Author: Quant (Your Mentor)
# Date: 2025-10-04
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os


class SizeFactor:
    """
    计算规模因子类。
    核心逻辑基于《因子投资：方法与实践》对规模因子的定义。
    """

    def __init__(self, master_data):
        """
        初始化规模因子计算。

        Args:
            master_data (pd.DataFrame): 经过loader.py清洗和合并后的主数据框。
                                        必须包含 'mkt_cap_ard' (总市值) 字段。
        """
        if 'mkt_cap_ard' not in master_data.columns:
            raise ValueError("错误: 输入的数据中缺少 'mkt_cap_ard' (总市值) 字段。")

        self.master_data = master_data
        self.factor_name = 'log_market_cap'

    def calculate_factor(self):
        """
        计算对数市值因子。
        在学术界和业界，直接使用市值的对数值作为规模因子的代理变量是一种常见做法，
        可以降低极值的影响，正如《因子投资：方法与实践》中所述。

        Returns:
            pd.DataFrame: 包含因子值的DataFrame，索引为['date', 'stock_code']。
        """
        print(f"\n[因子计算] 正在计算 {self.factor_name} (规模因子)...")

        factor_data = self.master_data[['mkt_cap_ard']].copy()

        factor_data[self.factor_name] = np.log(factor_data['mkt_cap_ard'])

        factor_data.dropna(inplace=True)

        print(f"{self.factor_name} 计算完成！")

        return factor_data[[self.factor_name]]


if __name__ == '__main__':
    # 这是一个简单的测试流程，用于演示如何独立运行和测试这个因子模块。

    # 1. 首先，加载数据
    # 为了能直接运行此文件，我们添加路径查找loader.py
    try:
        # 尝试直接导入，如果项目已正确配置为包
        from data_manager.loader import load_and_clean_data
    except ImportError:
        import sys

        # ******** 这是核心修改 1 ********
        # 将项目根目录 (factor_strategy_platform) 添加到Python路径中
        # __file__ 指的是当前文件 (size_factor.py)
        # os.path.dirname(__file__) 是 factor_library/fundamental
        # os.path.join(..., '..', '..') 向上返回两级到项目根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from data_manager.loader import load_and_clean_data
        # *******************************

    # ******** 这是核心修改 2 ********
    # 构建到数据文件夹的正确路径
    # 从项目根目录出发，进入 data_manager/DemoData
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    SAMPLE_DATA_DIR = os.path.join(project_root_path, "data_manager", "DemoData")
    # **************************

    if os.path.exists(SAMPLE_DATA_DIR):
        master_df, _ = load_and_clean_data(SAMPLE_DATA_DIR)

        if master_df is not None and not master_df.empty:
            # 2. 实例化因子类
            size_calculator = SizeFactor(master_df)

            # 3. 计算因子值
            size_factor_df = size_calculator.calculate_factor()

            print("\n规模因子(对数市值)计算结果预览:")
            print(size_factor_df.head())

            print("\n因子值的描述性统计:")
            print(size_factor_df.describe())
        else:
            print("加载的数据为空，无法计算因子。")
    else:
        print(f"错误: 未找到数据文件夹 '{SAMPLE_DATA_DIR}'。")