"""检索服务:把 Docling + BGE-M3 + Qdrant 包装成 HTTP 接口。
跑在 paper2code-rag 环境,端口 8002。
供 paper2code 环境的 Agent 通过 HTTP 远程调用。
"""
from fastapi import FastAPI
from pydantic import BaseModel
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

app = FastAPI(title="Paper2Code 检索服务")

# 全局初始化(服务启动时加载一次)
_opts = PdfPipelineOptions()
_opts.do_ocr = False  # 文字版PDF不需要OCR
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_opts)}
)
model = SentenceTransformer("BAAI/bge-m3", device="cpu")  # CPU,不抢vLLM显存
client = QdrantClient(path="./qdrant_local")
COLLECTION = "papers"


# ---- 请求/响应模型 ----
class IndexRequest(BaseModel):
    pdf_path: str

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 4


# ---- 接口 ----
@app.post("/index")
def index_pdf(req: IndexRequest):
    """解析PDF、向量化、入库。"""
    md = converter.convert(req.pdf_path).document.export_to_markdown()
    # 切块
    chunks, i = [], 0
    while i < len(md):
        chunks.append(md[i:i+500])
        i += 500 - 80
    vecs = model.encode(chunks, normalize_embeddings=True)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.COSINE),
    )
    points = [PointStruct(id=str(uuid.uuid4()), vector=vecs[idx].tolist(),
                          payload={"text": chunks[idx]}) for idx in range(len(chunks))]
    client.upsert(collection_name=COLLECTION, points=points)
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """检索相关片段。"""
    qv = model.encode([req.query], normalize_embeddings=True)[0]
    hits = client.query_points(collection_name=COLLECTION,
                               query=qv.tolist(), limit=req.top_k).points
    return {"results": "\n---\n".join(h.payload["text"] for h in hits)}


@app.get("/health")
def health():
    return {"status": "alive"}