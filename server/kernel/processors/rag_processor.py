# -*- coding: utf-8 -*-
"""
RAG（检索增强生成）处理器
负责从本地知识库检索相关文档 - 集成Triton 3.2完整知识库
"""
import os
import time
from typing import List, Dict, Any, Optional
import traceback

from .base import BaseProcessor
from .triton_rag_db import TritonRAGDatabase
from ..models import (
    RAGRequest, KernelResponse, RAGResult, 
    TaskType, TaskStatus
)

class RAGProcessor(BaseProcessor):
    """RAG处理器 - 集成Triton 3.2完整知识库"""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.knowledge_bases = {
            "triton_docs": "/workspace/monocases/docs/triton",
            "cuda_docs": "/workspace/monocases/docs/cuda",
            "general": "/workspace/monocases/docs/general"
        }
        
        # 初始化Triton 3.2知识库
        try:
            self.triton_db = TritonRAGDatabase()
            self.logger.info("Triton 3.2 RAG database initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Triton RAG database: {e}")
            self.triton_db = None
    
    async def process(self, 
                     request: RAGRequest, 
                     gpu_ids: List[int], 
                     context: Dict[str, Any]) -> KernelResponse:
        """处理RAG查询请求"""
        start_time = time.time()
        self.log_task_start("RAG Query", request.request_id)
        
        try:
            # 执行RAG查询
            result = await self._perform_rag_query(request)
            
            # 创建响应
            response = KernelResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=TaskType.RAG_QUERY,
                status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                result=result,
                processing_time=time.time() - start_time
            )
            
            self.log_task_end("RAG Query", request.request_id, result.success, response.processing_time)
            return response
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            self.logger.error(f"RAG query failed for request {request.request_id}: {error_msg}")
            
            return self.create_error_response(
                request, TaskType.RAG_QUERY, error_msg, 
                "RAGQueryError", error_traceback
            )
    
    async def _perform_rag_query(self, request: RAGRequest) -> RAGResult:
        """执行RAG查询 - 集成Triton 3.2知识库"""
        try:
            docs = []
            confidence_score = 0.0
            
            # 1. 从Triton 3.2知识库搜索
            if self.triton_db and request.knowledge_base in ["triton", "triton_docs", "all"]:
                triton_results = self.triton_db.search(
                    query=request.query,
                    top_k=max(request.top_k // 2, 1)
                )
                
                for result in triton_results:
                    docs.append({
                        'title': result['metadata'].get('title', 'Triton Documentation'),
                        'content': result['content'][:1000],  # 截取内容
                        'path': f"triton_3.2_db/{result['id']}",
                        'score': 1.0 - result['distance'],
                        'source': 'triton_3.2_knowledge_base',
                        'category': result['metadata'].get('category', 'general'),
                        'type': result['metadata'].get('type', 'documentation')
                    })
                
                if triton_results:
                    confidence_score = max(confidence_score, 0.9)
            
            # 2. 从传统文件系统搜索
            file_docs = await self._search_documents(request.query, request.knowledge_base, request.top_k - len(docs))
            docs.extend(file_docs)
            
            # 3. 智能答案生成
            answer = await self._generate_enhanced_answer(docs, request.query)
            
            # 4. 提供专门的API和最佳实践建议
            additional_info = await self._get_additional_triton_info(request.query)
            if additional_info:
                answer += f"\n\n### 相关参考信息：\n{additional_info}"
            
            return RAGResult(
                success=True,
                retrieved_docs=docs,
                answer=answer,
                confidence_score=max(confidence_score, 0.6),
                knowledge_base_used=request.knowledge_base
            )
            
        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            return RAGResult(
                success=False,
                knowledge_base_used=request.knowledge_base,
                answer=f"查询失败: {str(e)}"
            )
    
    async def _search_documents(self, query: str, kb_name: str, top_k: int) -> List[Dict[str, Any]]:
        """搜索文档"""
        docs = []
        kb_path = self.knowledge_bases.get(kb_name, self.knowledge_bases["general"])
        
        # 简单的文件搜索实现
        if os.path.exists(kb_path):
            for root, dirs, files in os.walk(kb_path):
                for file in files[:top_k]:  # 限制返回数量
                    if file.endswith(('.md', '.txt', '.py', '.cu')):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                content = f.read()
                                if query.lower() in content.lower():
                                    docs.append({
                                        'title': file,
                                        'content': content[:1000],  # 截取前1000字符
                                        'path': os.path.join(root, file),
                                        'score': 0.8  # 简单的固定分数
                                    })
                        except:
                            continue
        
        return docs
    
    async def _generate_enhanced_answer(self, docs: List[Dict[str, Any]], query: str) -> str:
        """生成增强的答案"""
        if not docs:
            return "未找到相关文档。请尝试更具体的查询词或检查拼写。"
        
        # 按分数排序文档
        docs_sorted = sorted(docs, key=lambda x: x.get('score', 0), reverse=True)
        
        # 分类组织信息
        api_docs = [doc for doc in docs_sorted if doc.get('type') == 'api']
        pattern_docs = [doc for doc in docs_sorted if doc.get('type') == 'pattern']
        best_practice_docs = [doc for doc in docs_sorted if doc.get('type') == 'optimization']
        error_docs = [doc for doc in docs_sorted if doc.get('type') in ['debugging', 'error']]
        
        answer_parts = []
        
        # API参考信息
        if api_docs:
            answer_parts.append("## API参考")
            for doc in api_docs[:2]:
                answer_parts.append(f"**{doc['title']}**")
                answer_parts.append(doc['content'][:500] + "...")
        
        # 代码模式
        if pattern_docs:
            answer_parts.append("\n## 代码模式")
            for doc in pattern_docs[:2]:
                answer_parts.append(f"**{doc['title']}**")
                answer_parts.append(doc['content'][:500] + "...")
        
        # 最佳实践
        if best_practice_docs:
            answer_parts.append("\n## 最佳实践")
            for doc in best_practice_docs[:2]:
                answer_parts.append(f"**{doc['title']}**")
                answer_parts.append(doc['content'][:500] + "...")
        
        # 调试和错误处理
        if error_docs:
            answer_parts.append("\n## 调试建议")
            for doc in error_docs[:1]:
                answer_parts.append(f"**{doc['title']}**")
                answer_parts.append(doc['content'][:400] + "...")
        
        # 如果没有专门分类的文档，使用通用格式
        if not answer_parts:
            answer_parts.append("## 相关信息")
            for doc in docs_sorted[:3]:
                answer_parts.append(f"**{doc['title']}** (评分: {doc.get('score', 0):.2f})")
                answer_parts.append(doc['content'][:400] + "...")
        
        return "\n\n".join(answer_parts)
    
    async def _get_additional_triton_info(self, query: str) -> Optional[str]:
        """获取额外的Triton信息"""
        if not self.triton_db:
            return None
        
        additional_info = []
        
        # 检查是否是API查询
        if any(keyword in query.lower() for keyword in ['api', 'function', 'method', 'tl.', 'triton.']):
            api_keywords = ['jit', 'load', 'store', 'dot', 'sum', 'max', 'min', 'program_id']
            for keyword in api_keywords:
                if keyword in query.lower():
                    api_ref = self.triton_db.get_api_reference(keyword)
                    if api_ref:
                        additional_info.append(f"**{keyword} API参考**: {api_ref[:200]}...")
                    break
        
        # 检查是否是优化查询
        if any(keyword in query.lower() for keyword in ['optimize', 'performance', 'fast', 'efficient']):
            practices = self.triton_db.get_best_practices(query)
            if practices:
                additional_info.append(f"**优化建议**: {practices[0][:200]}...")
        
        # 检查是否是错误查询
        if any(keyword in query.lower() for keyword in ['error', 'fail', 'bug', 'issue', 'problem']):
            debug_tips = self.triton_db.debug_error(query)
            if debug_tips:
                additional_info.append(f"**调试建议**: {debug_tips[0][:200]}...")
        
        # 检查是否是代码模式查询
        if any(keyword in query.lower() for keyword in ['example', 'pattern', 'template', 'sample']):
            pattern_keywords = ['matrix', 'vector', 'reduction', 'attention']
            for keyword in pattern_keywords:
                if keyword in query.lower():
                    pattern = self.triton_db.get_code_pattern(keyword)
                    if pattern:
                        additional_info.append(f"**{keyword}代码模式**: {pattern[:200]}...")
                    break
        
        return "\n\n".join(additional_info) if additional_info else None
    
    def _generate_answer(self, docs: List[Dict[str, Any]], query: str) -> str:
        """生成答案（向后兼容）"""
        if not docs:
            return "未找到相关文档"
        
        # 简单地合并文档内容
        combined_content = "\n\n".join([doc['content'] for doc in docs[:3]])
        return f"根据检索到的文档，相关信息如下：\n\n{combined_content}" 