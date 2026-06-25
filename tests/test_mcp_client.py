"""测试 MCP client 能否连上 server 并调用工具。"""
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

# MCP server 的 SSE endpoint(端口按第三步实际输出调整)
SERVER_URL = "http://localhost:8000/sse"

async def main():
    async with sse_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 列出 server 暴露的工具
            tools = await session.list_tools()
            print("可用工具:", [t.name for t in tools.tools])
            # 调用 retrieve 工具
            result = await session.call_tool("retrieve",
                                             {"query": "learning rate", "top_k": 2})
            print("检索结果:", result.content[0].text[:300])

asyncio.run(main())