"""
卖方分析师盈利预测数据下载器

功能：
- 下载 Tushare 提供的分析师盈利预测数据（通过 report_rc 接口）
- 包含 EPS、营业收入、净利润等预测指标
- 支持全市场股票的预测数据获取

数据字段说明（report_rc接口）：
- ts_code: 股票代码
- ann_date: 预测发布日期
- report_date: 报告期（被预测的财报期）
- report_type: 报告类型（1=年报、2=中报、3=季报）
- eps_avg: 平均预测每股收益（EPS）
- eps_max: 最高预测每股收益
- eps_min: 最低预测每股收益
- eps_std: 预测EPS标准差
- revenue_avg: 平均预测营业收入（万元）
- revenue_max: 最高预测营业收入
- revenue_min: 最低预测营业收入
- revenue_std: 预测营业收入标准差
- net_profit_avg: 平均预测净利润（万元）
- net_profit_max: 最高预测净利润
- net_profit_min: 最低预测净利润
- net_profit_std: 预测净利润标准差
- pe_avg: 平均预测市盈率
- roe_avg: 平均预测净资产收益率

注意事项：
- report_rc 接口需要积分权限（建议使用 1000积分 或以上的账户）
- 数据更新频率：日更
- API 限流：每分钟调用不超过100次（建议每次调用间隔0.6秒）
"""

import tushare as ts
import pandas as pd
import time
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import get_tushare_token, RAW_DATA_PATH

# 初始化 Tushare API
ts.set_token(get_tushare_token())
pro = ts.pro_api()

print("=" * 80)
print("卖方分析师盈利预测数据下载器")
print("=" * 80)
print("Tushare API 初始化成功。")

# 获取A股上市公司列表
print("\n正在获取最新的A股上市公司列表...")
try:
    stock_list_df = pro.stock_basic(
        exchange='', 
        list_status='L', 
        fields='ts_code,name,industry,list_date'
    )
    print(f"✅ 成功获取 {len(stock_list_df)} 家上市公司的基本信息。")
except Exception as e:
    print(f"❌ 获取股票列表失败，请检查网络或Token设置: {e}")
    exit()

# 定义下载参数
# 获取近期的预测数据（建议根据需求调整时间范围）
start_date = '20200101'  # 从2020年开始
end_date = datetime.now().strftime('%Y%m%d')  # 到今天

print(f"\n下载参数:")
print(f"  时间范围: {start_date} 至 {end_date}")
print(f"  股票数量: {len(stock_list_df)}")
print(f"  数据接口: report_rc (盈利预测汇总)")

# 创建空的DataFrame用于存储所有数据
all_forecast_data = pd.DataFrame()

# 统计变量
success_count = 0
fail_count = 0
total_records = 0

# 循环获取所有A股公司的盈利预测数据
total_stocks = len(stock_list_df)
print(f"\n准备开始下载 {total_stocks} 只股票的盈利预测数据，这将需要较长时间，请耐心等待...")
print("=" * 80)

# 遍历股票列表中的每一只股票
for index, row in stock_list_df.iterrows():
    ts_code = row['ts_code']
    stock_name = row['name']
    industry = row['industry']
    
    # 打印进度
    progress = (index + 1) / total_stocks * 100
    print(f"[{index + 1}/{total_stocks}] ({progress:.1f}%) 正在处理: {ts_code} ({stock_name}) - {industry}")
    
    try:
        # 调用 report_rc 接口获取该股票的盈利预测数据
        # 这个接口返回的是分析师预测的汇总数据（均值、最大、最小等）
        df_single_stock = pro.report_rc(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='ts_code,ann_date,report_date,report_type,eps_avg,eps_max,eps_min,eps_std,revenue_avg,revenue_max,revenue_min,revenue_std,net_profit_avg,net_profit_max,net_profit_min,net_profit_std,pe_avg,roe_avg'
        )
        
        # 如果成功获取到数据
        if df_single_stock is not None and not df_single_stock.empty:
            # 将新获取的数据拼接到总DataFrame中
            all_forecast_data = pd.concat(
                [all_forecast_data, df_single_stock], 
                ignore_index=True
            )
            record_count = len(df_single_stock)
            total_records += record_count
            success_count += 1
            print(f"  ✅ 成功获取 {record_count} 条预测记录 | 累计: {total_records} 条")
        else:
            fail_count += 1
            print(f"  ⚠️  未找到预测数据")
        
        # API频率限制：每次调用后等待0.6秒
        # Tushare 对积分接口有更严格的限制，建议适当延长等待时间
        time.sleep(0.6)
        
    except Exception as e:
        fail_count += 1
        print(f"  ❌ 获取数据时出错: {e}")
        # 即使出错也继续处理下一只股票
        continue
    
    # 每处理100只股票，显示一次汇总统计
    if (index + 1) % 100 == 0:
        print("-" * 80)
        print(f"阶段性统计 [{index + 1}/{total_stocks}]:")
        print(f"  成功: {success_count} | 失败: {fail_count} | 总记录数: {total_records}")
        print("-" * 80)

print("\n" + "=" * 80)
print("所有股票的盈利预测数据获取完成！")
print("=" * 80)

# 显示最终统计
print("\n📊 下载统计:")
print(f"  总股票数: {total_stocks}")
print(f"  成功获取: {success_count} ({success_count/total_stocks*100:.1f}%)")
print(f"  未找到数据: {fail_count} ({fail_count/total_stocks*100:.1f}%)")
print(f"  总记录数: {total_records}")

# 数据质量检查
if not all_forecast_data.empty:
    print("\n📈 数据质量检查:")
    print(f"  数据总行数: {len(all_forecast_data):,}")
    print(f"  数据总列数: {len(all_forecast_data.columns)}")
    print(f"  覆盖股票数: {all_forecast_data['ts_code'].nunique()}")
    print(f"  时间范围: {all_forecast_data['ann_date'].min()} 至 {all_forecast_data['ann_date'].max()}")
    
    # 检查关键字段的缺失率
    print("\n  关键字段缺失率:")
    key_fields = ['eps_avg', 'revenue_avg', 'net_profit_avg', 'pe_avg', 'roe_avg']
    for field in key_fields:
        if field in all_forecast_data.columns:
            missing_rate = all_forecast_data[field].isna().sum() / len(all_forecast_data) * 100
            print(f"    {field}: {missing_rate:.2f}%")
    
    # 显示数据样例
    print("\n  数据样例（前5行）:")
    print(all_forecast_data.head())
    
    # 显示数据类型
    print("\n  数据类型:")
    print(all_forecast_data.dtypes)
    
    # 保存数据到Parquet文件
    output_file = RAW_DATA_PATH / 'analyst_forecast_data.parquet'
    print(f"\n💾 正在保存数据到: {output_file}")
    
    try:
        all_forecast_data.to_parquet(output_file, index=False, engine='pyarrow')
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ 数据保存成功！文件大小: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ 保存数据时出错: {e}")
else:
    print("\n⚠️  警告: 未获取到任何数据，请检查:")
    print("  1. Tushare账户积分是否足够（report_rc接口需要积分权限）")
    print("  2. API Token是否正确配置")
    print("  3. 网络连接是否正常")

print("\n" + "=" * 80)
print("程序执行完毕！")
print("=" * 80)
