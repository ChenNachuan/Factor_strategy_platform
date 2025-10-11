import tushare as ts # type: ignore
import pandas as pd
import time
from pathlib import Path

# 初始化 Tushare API
ts.set_token('a5dfb5990b7a5e9eda778b060c8c987f9a7fbd2a5caae8ddc7298197')
pro = ts.pro_api()

print("Tushare API 初始化成功。")

# 定义需要下载的指数列表
print("正在定义指数列表...")
index_list = [
    {'ts_code': '000300.SH', 'name': '沪深300'},
    {'ts_code': '000905.SH', 'name': '中证500'},
    {'ts_code': '000016.SH', 'name': '上证50'},
    {'ts_code': '000852.SH', 'name': '中证1000'},
    {'ts_code': '399001.SZ', 'name': '深证成指'},
    {'ts_code': '000001.SH', 'name': '上证综指'},
]
print(f"成功定义 {len(index_list)} 个指数。")

# 循环获取所有指数的日线行情数据
start_date = '20100101'
end_date = '20250930'

# 创建一个空的 DataFrame 用于存储所有指数的数据
all_index_data = pd.DataFrame()

print(f"开始获取从 {start_date} 到 {end_date} 的指数日线数据...")

# 循环遍历指数列表中的每一个指数
for index in index_list:
    ts_code = index['ts_code']
    try:
        # 调用 pro_bar 接口获取单个指数的日线数据
        df_single_index = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        # 将新获取的数据拼接到总的 DataFrame 中
        all_index_data = pd.concat([all_index_data, df_single_index])
        
        # 打印进度提示
        print(f"成功获取 {ts_code} ({index['name']}) 的数据。")

        # 为了避免触发接口频率限制，稍作等待
        time.sleep(0.3) 
        
    except Exception as e:
        print(f"获取 {ts_code} 数据时出错: {e}")

print("所有指数数据获取完毕！")

# 将数据存储为 Parquet 格式

# 首先，做一个安全检查，确保我们的 all_index_data 不是空的。
if not all_index_data.empty:
    print("\n开始将数据存储到 Parquet 文件...")
    try:
        # 获取当前脚本文件(.py)的绝对路径
        current_script_path = Path(__file__).resolve()
        # 从当前脚本路径向上回溯两级，到达 data_manager 目录
        data_manager_dir = current_script_path.parent.parent
        # 构建目标存储目录路径
        save_dir = data_manager_dir / 'raw_data'
        
        # 确保目标目录存在，如果不存在就创建它
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建完整的文件保存路径
        file_path = save_dir / 'a_index_daily_data.parquet'

        all_index_data.to_parquet(file_path, engine='pyarrow', index=False)
        
        # 输出信息路径
        print(f"数据已成功保存到: {file_path}")
        
    except Exception as e:
        print(f"数据保存失败，错误信息: {e}")
else:
    print("\n未能获取到任何数据，请检查网络或 Tushare Token 设置。")