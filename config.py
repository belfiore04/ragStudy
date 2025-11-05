from pathlib import Path
import os

# 向量模型与索引
EMB_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_INDEX_ROOT = Path("./projects")
K_RETRIEVE_DEFAULT = 6


# 对话模型配置（DeepSeek 兼容 OpenAI SDK）
MODEL_NAME = "deepseek-chat"
MODEL_BASE_URL = "https://api.deepseek.com/v1"
API_ENV_KEY = "DEEPSEEK_API_KEY"


# 渲染配置

PDF_RENDER_DPI = 150
