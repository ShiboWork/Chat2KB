# 初始化组件
from kbTransformer import knowledgeTransformer
from config import init_components
from knowledge_base import *

client, encoder = init_components()
knowledge_base = KnowledgeBase(encoder)
retriever = knowledge_base.build_retriever()

# 创建转换器实例（指定监控路径和输出路径）
transformer = knowledgeTransformer(
    client=client,
    retriever=retriever,
    monitor_dir="../../chatHistory",  # 历史记录存放路径
    export_dir="../../knowledgeRepo"   # 知识库输出路径
)

# 执行转换（每次处理最多5个文件）
transformer.convert_to_knowledge(batch_size=5)

# 查看未处理文件
pending_files = transformer.detect_unprocessed_files()
print(f"待处理文件: {pending_files}")
