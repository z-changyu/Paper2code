"""MCP server:把检索能力按 MCP 协议暴露(SSE transport)。
跑在 paper2code-rag 环境,端口 8003。
复用与 FastAPI 服务相同的 Docling+BGE-M3+Qdrant 检索核心。
"""
from mcp.server.fastmcp import FastMCP
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# FastMCP 是 mcp SDK 提供的高层封装,用法类似 FastAPI
mcp = FastMCP("paper2code-retrieval")

# 检索核心(和 A 一样,全局初始化一次)
_opts = PdfPipelineOptions()
_opts.do_ocr = False
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_opts)}
)
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
client = QdrantClient(path="./qdrant_local")
COLLECTION = "papers"


@mcp.tool()
def index_pdf(pdf_path: str) -> str:
    """解析指定路径的PDF论文,切块、向量化并存入向量库。返回入库的块数。"""
    md = converter.convert(pdf_path).document.export_to_markdown()
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
    return f"已入库 {len(chunks)} 个文本块"


@mcp.tool()
def retrieve(query: str, top_k: int = 4) -> str:
    """检索与查询最相关的论文片段。query是查询文本,top_k是返回数量。"""
    qv = model.encode([query], normalize_embeddings=True)[0]
    hits = client.query_points(collection_name=COLLECTION,
                               query=qv.tolist(), limit=top_k).points
    return "\n---\n".join(h.payload["text"] for h in hits)


if __name__ == "__main__":
    # 用 SSE transport 启动,监听端口
    mcp.run(transport="sse")