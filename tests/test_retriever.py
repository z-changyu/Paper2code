"""Day 5: 单独测试检索器。"""
from retrieval.retriever import SimpleRetriever

r = SimpleRetriever()
r.load_pdf("data/papers/test.pdf")  # 确保这个PDF存在

# 测几个典型查询，看检索到的内容是否相关
for query in ["model architecture", "training hyperparameters learning rate", "experiment results"]:
    print(f"\n=== 查询: {query} ===")
    result = r.search(query, top_k=2)
    print(result[:400])  # 只看前400字
