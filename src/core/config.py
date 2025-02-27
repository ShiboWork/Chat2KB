import os
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-default-key")
EMBEDDING_MODEL_PATH = "/root/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/8d6b950845285729817bf8e1af1861502c2fed0c"

# 初始化全局组件
def init_components():
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    
    encoder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH
    )
    
    return client, encoder
