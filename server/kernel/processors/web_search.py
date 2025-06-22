# -*- coding: utf-8 -*-
"""
网络搜索处理器
负责搜索在线Triton/CUDA文档和资源
"""
import time
import aiohttp
from typing import List, Dict, Any
import traceback

from .base import BaseProcessor
from ..models import (
    WebSearchRequest, KernelResponse, WebSearchResult, 
    TaskType, TaskStatus
)

class WebSearchProcessor(BaseProcessor):
    """网络搜索处理器"""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        
        # 预定义的搜索源
        self.search_sources = {
            "triton_docs": [
                "https://triton-lang.org/main/",
                "https://github.com/openai/triton",
                "https://pytorch.org/tutorials/intermediate/triton_tutorial.html"
            ],
            "cuda_docs": [
                "https://docs.nvidia.com/cuda/",
                "https://developer.nvidia.com/cuda-toolkit",
                "https://github.com/NVIDIA/cuda-samples"
            ],
            "general": [
                "https://pytorch.org/docs/stable/",
                "https://stackoverflow.com/questions/tagged/cuda",
                "https://stackoverflow.com/questions/tagged/triton"
            ]
        }
    
    async def process(self, 
                     request: WebSearchRequest, 
                     gpu_ids: List[int], 
                     context: Dict[str, Any]) -> KernelResponse:
        """处理网络搜索请求"""
        start_time = time.time()
        self.log_task_start("Web Search", request.request_id)
        
        try:
            # 执行网络搜索
            result = await self._perform_web_search(request)
            
            # 创建响应
            response = KernelResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=TaskType.WEB_SEARCH,
                status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                result=result,
                processing_time=time.time() - start_time
            )
            
            self.log_task_end("Web Search", request.request_id, result.success, response.processing_time)
            return response
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            self.logger.error(f"Web search failed for request {request.request_id}: {error_msg}")
            
            return self.create_error_response(
                request, TaskType.WEB_SEARCH, error_msg, 
                "WebSearchError", error_traceback
            )
    
    async def _perform_web_search(self, request: WebSearchRequest) -> WebSearchResult:
        """执行网络搜索"""
        start_time = time.time()
        
        try:
            # 获取搜索源
            sources = self.search_sources.get(request.search_type, self.search_sources["general"])
            
            # 执行搜索
            search_results = []
            async with aiohttp.ClientSession() as session:
                for source in sources[:request.max_results]:
                    try:
                        result = await self._search_source(session, source, request.query)
                        if result:
                            search_results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Failed to search {source}: {e}")
                        continue
            
            # 生成摘要
            summary = self._generate_summary(search_results, request.query)
            
            return WebSearchResult(
                success=True,
                search_results=search_results,
                summary=summary,
                search_time=time.time() - start_time
            )
            
        except Exception as e:
            return WebSearchResult(
                success=False,
                search_time=time.time() - start_time
            )
    
    async def _search_source(self, session: aiohttp.ClientSession, source: str, query: str) -> Dict[str, Any]:
        """搜索单个源"""
        try:
            async with session.get(source, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # 简单的内容匹配
                    if query.lower() in content.lower():
                        return {
                            'title': f"Results from {source}",
                            'url': source,
                            'snippet': self._extract_snippet(content, query),
                            'relevance_score': 0.7
                        }
                
        except Exception as e:
            self.logger.debug(f"Failed to fetch {source}: {e}")
            
        return None
    
    def _extract_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """提取相关片段"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # 找到查询词的位置
        index = content_lower.find(query_lower)
        if index == -1:
            return content[:max_length] + "..."
        
        # 提取查询词周围的内容
        start = max(0, index - 50)
        end = min(len(content), index + max_length)
        snippet = content[start:end]
        
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def _generate_summary(self, results: List[Dict[str, Any]], query: str) -> str:
        """生成搜索结果摘要"""
        if not results:
            return f"未找到关于 '{query}' 的相关信息"
        
        summary = f"找到 {len(results)} 个关于 '{query}' 的相关资源：\n\n"
        
        for i, result in enumerate(results[:3], 1):
            summary += f"{i}. {result['title']}\n"
            summary += f"   链接: {result['url']}\n"
            summary += f"   摘要: {result['snippet'][:100]}...\n\n"
        
        return summary 