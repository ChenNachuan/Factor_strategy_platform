# /config.py
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# 这允许您在项目根目录创建一个 .env 文件来存储您的 TUSHARE_TOKEN
# 例如: TUSHARE_TOKEN=your_token_here
load_dotenv()

def get_tushare_token() -> str:
    """
    从环境变量中获取 Tushare Token。

    Returns:
        str: Tushare Token.
    """
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        # 如果环境变量中没有，可以在这里设置一个默认的或者直接抛出错误
        # 为了保持原有逻辑，这里暂时保留了原来的token作为备用，但强烈建议移至.env文件
        token = 'a5dfb5990b7a5e9eda778b060c8c987f9a7fbd2a5caae8ddc7298197'
        print("警告: 未在环境变量中找到 TUSHARE_TOKEN。建议创建 .env 文件并设置该变量。")
    return token

# 项目根目录
# Path(__file__).parent 可以获取当前文件所在目录
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent

# 数据目录 - 指向 data_manager 下的实际数据路径
DATA_PATH = PROJECT_ROOT / 'data_manager'
RAW_DATA_PATH = DATA_PATH / 'raw_data'
CLEAN_DATA_PATH = DATA_PATH / 'clean_data'

# 确保目录存在
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
CLEAN_DATA_PATH.mkdir(parents=True, exist_ok=True)
