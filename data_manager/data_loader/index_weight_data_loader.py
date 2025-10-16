import tushare as ts  # type: ignore
import pandas as pd
import time
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根路径以便导入 config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import get_tushare_token, RAW_DATA_PATH

# 初始化 Tushare API
ts.set_token(get_tushare_token())
pro = ts.pro_api()

print("Tushare API 初始化成功。")

# 定义需要下载权重数据的指数列表
print("正在定义指数列表...")
index_list = [
    {'ts_code': '000300.SH', 'name': '沪深300'},
    {'ts_code': '000905.SH', 'name': '中证500'},
    {'ts_code': '000016.SH', 'name': '上证50'},
    {'ts_code': '000852.SH', 'name': '中证1000'},
    {'ts_code': '399006.SZ', 'name': '创业板指'},
    {'ts_code': '000001.SH', 'name': '上证指数'},
    {'ts_code': '399001.SZ', 'name': '深证成指'},
]
print(f"成功定义 {len(index_list)} 个指数。")

# 设置时间范围
# 建议：开始日期和结束日期分别输入当月第一天和最后一天的日期
# 例如：'20220101' 到 '20220131' 获取2022年1月的权重数据
start_date = '20250901'  # 格式: YYYYMMDD
end_date = '20250930'    # 格式: YYYYMMDD

print(f"\n开始获取从 {start_date} 到 {end_date} 的指数成分权重数据...")
print("=" * 60)

# 创建一个空的 DataFrame 用于存储所有指数的权重数据
all_weight_data = pd.DataFrame()

# 生成月度日期列表（每月第一天和最后一天）
def generate_monthly_dates(start_date: str, end_date: str):
    """
    生成月度日期范围列表
    每个月返回第一天和最后一天
    """
    start = pd.to_datetime(start_date, format='%Y%m%d')
    end = pd.to_datetime(end_date, format='%Y%m%d')
    
    # 生成月度范围
    date_range = pd.date_range(start=start, end=end, freq='MS')  # MS = Month Start
    
    monthly_dates = []
    for date in date_range:
        # 每月第一天
        first_day = date.strftime('%Y%m%d')
        # 每月最后一天
        last_day = (date + pd.offsets.MonthEnd(0)).strftime('%Y%m%d')
        monthly_dates.append((first_day, last_day))
    
    return monthly_dates

monthly_dates = generate_monthly_dates(start_date, end_date)
print(f"生成了 {len(monthly_dates)} 个月度时间段。")
print("=" * 60)

# 循环遍历指数列表
for index in index_list:
    ts_code = index['ts_code']
    index_name = index['name']
    
    print(f"\n正在获取 {ts_code} ({index_name}) 的权重数据...")
    
    index_weight_data = pd.DataFrame()
    
    # 对每个月进行查询
    for i, (month_start, month_end) in enumerate(monthly_dates, 1):
        try:
            # 调用 index_weight 接口获取指数成分权重
            # 参数说明：
            # index_code: 指数代码
            # start_date: 开始日期（建议为月初）
            # end_date: 结束日期（建议为月末）
            df_month = pro.index_weight(
                index_code=ts_code,
                start_date=month_start,
                end_date=month_end
            )
            
            if df_month is not None and not df_month.empty:
                index_weight_data = pd.concat([index_weight_data, df_month], ignore_index=True)
                print(f"  [{i}/{len(monthly_dates)}] {month_start[:6]} - 获取到 {len(df_month)} 条记录")
            else:
                print(f"  [{i}/{len(monthly_dates)}] {month_start[:6]} - 无数据")
            
            # 为了避免触发接口频率限制，稍作等待
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  [{i}/{len(monthly_dates)}] {month_start[:6]} - 获取失败: {e}")
            time.sleep(1)  # 出错后等待更长时间
    
    # 将当前指数的数据合并到总数据中
    if not index_weight_data.empty:
        all_weight_data = pd.concat([all_weight_data, index_weight_data], ignore_index=True)
        print(f"✅ {index_name} 数据获取完成，共 {len(index_weight_data)} 条记录。")
    else:
        print(f"⚠️ {index_name} 未获取到任何数据。")
    
    # 每个指数之间稍作等待
    time.sleep(0.01)

print("\n" + "=" * 60)
print("所有指数权重数据获取完毕！")
print("=" * 60)

# 数据存储
if not all_weight_data.empty:
    print(f"\n数据概览:")
    print(f"  总记录数: {len(all_weight_data):,}")
    print(f"  包含指数数: {all_weight_data['index_code'].nunique()}")
    print(f"  包含股票数: {all_weight_data['con_code'].nunique()}")
    print(f"  日期范围: {all_weight_data['trade_date'].min()} ~ {all_weight_data['trade_date'].max()}")
    
    print("\n数据字段:")
    print(f"  {list(all_weight_data.columns)}")
    
    print("\n样本数据（前5行）:")
    print(all_weight_data.head())
    
    print("\n开始将数据存储到 Parquet 文件...")
    try:
        # 构建完整的文件保存路径
        file_path = RAW_DATA_PATH / 'index_weight_data.parquet'
        
        # 保存为 Parquet 格式
        all_weight_data.to_parquet(file_path, engine='pyarrow', index=False)
        
        # 输出保存信息
        print(f"✅ 数据已成功保存到: {file_path}")
        print(f"📦 文件大小: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ 数据保存失败，错误信息: {e}")
else:
    print("\n⚠️ 未能获取到任何数据，请检查:")
    print("  1. 网络连接是否正常")
    print("  2. Tushare Token 是否正确")
    print("  3. Tushare 账户权限是否包含指数权重接口")
    print("  4. 日期范围是否合理")

print("\n" + "=" * 60)
print("🎉 程序执行完成！")
print("=" * 60)
