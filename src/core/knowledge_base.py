from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader

class KnowledgeBase:
    def __init__(self, encoder):
        self.encoder = encoder
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        
    def load_documents(self):
        txt_loader = DirectoryLoader(
            "../../knowledge_base",
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},  # 添加自动检测编码
            show_progress=True
        )
        documents = txt_loader.load()
        if not documents:
            raise ValueError("知识库目录中未找到任何txt文件")
        return documents
        
    def build_retriever(self):
        """构建检索器"""
        documents = self.load_documents()
        splits = self.text_splitter.split_documents(documents)
        vector_db = Chroma.from_documents(
            splits, 
            self.encoder,
            collection_metadata={"hnsw:space": "cosine"}
        )
        return vector_db.as_retriever(
            search_kwargs={"k": min(3, len(splits))}
        )
