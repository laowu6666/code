"""
配置文件，包含所有系统设置和参数
"""
from pathlib import Path
import os

# 基础路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# 数据文件路径
DIRECTORY_CSV = DATA_DIR / "B737_directory.csv"
RESULT_CSV = DATA_DIR / "B737_result.csv"

# 向量存储配置
VECTOR_STORE_PATHS = {
    "zh": VECTOR_STORE_DIR / "directory_zh",
    "en": VECTOR_STORE_DIR / "directory_en",
    "fr": VECTOR_STORE_DIR / "directory_fr",
    "result": VECTOR_STORE_DIR / "result"
}

# DeepSeek API配置
DEEPSEEK_API_CONFIG = {
    "api_key": "sk-7c9aa010b869401b8ffc78ac5fe75944",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "temperature": 0.7,
    "max_tokens": 2000,
}

# 支持的语言配置
SUPPORTED_LANGUAGES = {
    "zh": "中文",
    "en": "English",
    "fr": "Français"
}

# 检索配置
RETRIEVAL_CONFIG = {
    "top_k": 5,
    "score_threshold": 0.5
}

# 缓存配置
CACHE_DIR = BASE_DIR / "cache"
CACHE_EXPIRY = 24 * 60 * 60  # 24小时

# 创建必要的目录
for directory in [DATA_DIR, VECTOR_STORE_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# 日志配置
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_CONFIG = {
    "path": LOG_DIR / "app.log",
    "rotation": "500 MB",
    "retention": "10 days",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
} 