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
        可以降低极值的影响，正如《因子投资：方法与实践》中所述 [cite: 3251]。

        Returns:
            pd.DataFrame: 包含因子值的DataFrame，索引为['date', 'stock_code']。
        """
        print(f"\n[因子计算] 正在计算 {self.factor_name} (规模因子)...")

        # 复制一份数据以避免修改原始DataFrame
        factor_data = self.master_data[['mkt_cap_ard']].copy()

        # 1. 对市值取自然对数
        # 我们使用市值的对数作为因子值。通常，市值越小，我们预期其未来收益越高。
        # 在后续的排序中，我们会将这个值从小到大排。
        factor_data[self.factor_name] = np.log(factor_data['mkt_cap_ard'])

        # 2. 剔除计算后产生的无穷大或缺失值
        factor_data.dropna(inplace=True)

        print(f"{self.factor_name} 计算完成！")

        # 返回只包含因子值的DataFrame
        return factor_data[[self.factor_name]]


if __name__ == '__main__':
    # 这是一个简单的测试流程，用于演示如何独立运行和测试这个因子模块。

    # 1. 首先，加载数据
    # 为了能直接运行此文件，我们添加路径查找loader.py
    try:
        from DataManager.loader import load_and_clean_data
    except ImportError:
        import sys

        # 将项目根目录添加到Python路径中
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from data_manager.loader import load_and_clean_data

    SAMPLE_DATA_DIR = "DemoDdata"

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