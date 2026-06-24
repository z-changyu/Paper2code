"""Day 8: 升级版检索器
- 解析：pypdf（Docling 因环境冲突暂用 pypdf 替代，待独立环境再升级）
- 向量化：BGE-M3（通过 sentence-transformers 加载，取稠密语义向量）
- 存储/检索：Qdrant 向量数据库（本地嵌入模式，无需单独起服务）

相比第一周 SimpleRetriever 的升级点：
- embedding 从 bge-small 升级到 bge-m3，语义表示更强
- 存储从“内存 numpy 裸算”升级到 Qdrant 向量库（持久化、ANN 检索、可扩展）
"""
import uuid
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class AdvancedRetriever:
    def __init__(self, collection: str = "papers", db_path: str = "./qdrant_local"):
        # BGE-M3 通过 sentence-transformers 加载，稳定、依赖轻
        self.model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        # Qdrant 本地嵌入模式：数据存本地文件，无需 Docker/服务
        self.client = QdrantClient(path=db_path)
        self.collection = collection

    def load_pdf(self, pdf_path: str, chunk_size: int = 500, overlap: int = 80):
        # 1) pypdf 解析为纯文本
        reader = PdfReader(pdf_path)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        # 2) 固定长度滑窗切块（带重叠，避免切断语义）
        chunks = []
        i = 0
        while i < len(full_text):
            chunks.append(full_text[i:i + chunk_size])
            i += chunk_size - overlap

        if not chunks:
            raise RuntimeError("PDF 解析结果为空，可能是扫描版或解析失败")

        # 3) BGE-M3 向量化（归一化，便于用 COSINE 距离）
        vecs = self.model.encode(chunks, normalize_embeddings=True)
        dim = vecs.shape[1]

        # 4) 建 collection 并入库
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vecs[idx].tolist(),
                payload={"text": chunks[idx]},
            )
            for idx in range(len(chunks))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        print(f"[advanced_retriever] 入库 {len(chunks)} 个文本块，向量维度 {dim}")

    def search(self, query: str, top_k: int = 4) -> str:
        qv = self.model.encode([query], normalize_embeddings=True)[0]
        hits = self.client.query_points(
            collection_name=self.collection,
            query=qv.tolist(),
            limit=top_k,
        ).points
        return "\n---\n".join(h.payload["text"] for h in hits)