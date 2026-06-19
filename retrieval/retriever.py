"""
检索模块 —— Day 5 实现最简版，第二周升级。

第一周（Day 5）最简版：
    pypdf 抽文本 -> 固定长度切块 -> sentence-transformers 向量化
    -> 内存里算余弦相似度检索 top-k

第二周升级：
    Docling 解析 PDF -> BGE-M3 三向量混合检索 -> BGE-Reranker-v2 重排 -> Qdrant 存储

下面是 Day 5 的骨架，填充 TODO 即可。
"""
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class SimpleRetriever:
    """第一周用的极简内存检索器。够跑通链路，不追求质量。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        # 轻量 embedding 模型，第一周够用；第二周换 BGE-M3
        self.encoder = SentenceTransformer(model_name)
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def load_pdf(self, pdf_path: str, chunk_size: int = 500, overlap: int = 80):
        """读 PDF -> 抽文本 -> 切块 -> 向量化 -> 存内存。"""
        reader = PdfReader(pdf_path)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        # 固定长度滑窗切块（带重叠，避免切断语义）
        self.chunks = []
        i = 0
        while i < len(full_text):
            self.chunks.append(full_text[i:i + chunk_size])
            i += chunk_size - overlap

        # 批量向量化并归一化（方便用点积算余弦相似度）
        self.embeddings = self.encoder.encode(
            self.chunks, normalize_embeddings=True
        )
        print(f"[retriever] 载入 {len(self.chunks)} 个文本块")

    def search(self, query: str, top_k: int = 4) -> str:
        """检索与 query 最相关的 top_k 个块，拼成一段文本返回。"""
        if self.embeddings is None:
            raise RuntimeError("请先调用 load_pdf 载入文档")
        q = self.encoder.encode([query], normalize_embeddings=True)[0]
        # 归一化后点积 == 余弦相似度
        scores = self.embeddings @ q
        top_idx = np.argsort(scores)[::-1][:top_k]
        return "\n---\n".join(self.chunks[i] for i in top_idx)
