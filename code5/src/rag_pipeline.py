"""
RAG Pipeline 模块，实现多语言问答功能
"""
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from loguru import logger
from langdetect import detect, DetectorFactory
import os
from openai import OpenAI
import re
import logging

# 设置日志记录到文件
rag_logger = logging.getLogger('rag_pipeline')
rag_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('rag_pipeline_debug.log', mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
rag_logger.addHandler(file_handler)

# 设置语言检测的随机种子，确保结果一致性
DetectorFactory.seed = 0

from src.config import (
    SUPPORTED_LANGUAGES,
    DEEPSEEK_API_CONFIG,
    CACHE_DIR,
    CACHE_EXPIRY
)
from src.data_processor import DataProcessor
from src.vector_store import VectorStore

class RAGPipeline:
    def __init__(self):
        """初始化 RAG Pipeline"""
        # 确保缓存目录存在
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        self.data_processor = DataProcessor()
        self.vector_store = VectorStore()
        self.cache = self._load_cache()
        
        # 初始化 DeepSeek 客户端
        self.client = OpenAI(
            api_key=DEEPSEEK_API_CONFIG['api_key'],
            base_url=DEEPSEEK_API_CONFIG['base_url']
        )

    def _load_cache(self) -> Dict:
        """加载缓存"""
        cache_file = CACHE_DIR / "response_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                # 清理过期缓存
                current_time = time.time()
                cache = {
                    k: v for k, v in cache.items()
                    if current_time - v['timestamp'] < CACHE_EXPIRY
                }
                return cache
            except Exception as e:
                logger.error(f"加载缓存失败: {e}")
                return {}
        return {}

    def _save_cache(self) -> None:
        """保存缓存"""
        try:
            cache_file = CACHE_DIR / "response_cache.json"
            # 确保缓存目录存在
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
                
            # 确保文件写入到磁盘
            f.flush()
            os.fsync(f.fileno())
            
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _detect_language(self, text: str) -> str:
        """
        检测文本语言

        Args:
            text: 输入文本

        Returns:
            语言代码 ('zh', 'en', 'fr')
        """
        if not text.strip():
            return 'en'  # 空文本默认使用英语
            
        # 简单的中文检测
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'
            
        try:
            detected = detect(text)
            if detected in ['zh-cn', 'zh-tw']:
                return 'zh'
            elif detected == 'en':
                return 'en'
            elif detected == 'fr':
                return 'fr'
            else:
                return 'en'  # 默认使用英语
        except:
            return 'en'  # 检测失败时默认使用英语

    def _translate_result(self, result: str, target_language: str) -> str:
        """
        使用 DeepSeek API 翻译结果

        Args:
            result: 英文结果文本
            target_language: 目标语言代码

        Returns:
            翻译后的文本
        """
        rag_logger.debug(f"开始翻译和格式化，目标语言: {target_language}")
        rag_logger.debug(f"原始文本长度: {len(result)}")
        
        if target_language == 'en':
            rag_logger.debug("目标语言是英文，跳过翻译，只进行简单格式化")
            # 只保留基本格式和术语
            formatted = self._simplify_formatting(result)
            rag_logger.debug(f"英文结果格式化完成，长度: {len(formatted)}")
            return formatted

        language_names = {
            'zh': '中文',
            'fr': 'French'
        }

        try:
            # 简化原始文本格式
            simplified_result = self._simplify_formatting(result)
            rag_logger.debug(f"简化后文本长度: {len(simplified_result)}")
            
            rag_logger.debug("调用DeepSeek API进行翻译")
            response = self.client.chat.completions.create(
                model=DEEPSEEK_API_CONFIG['model'],
                messages=[
                    {"role": "system", "content": """You are a professional aircraft maintenance document translator.
                    
                    Keep the translation simple and directly usable:
                    1. Maintain paragraph structure but don't worry about complex markdown
                    2. Only use ** for important warnings or critical information
                    3. Keep technical terms in their original form, especially:
                       - System names like CDS, BITE, DEU, CPU 
                       - Error codes like 293 130 00
                       - Task numbers like 31-62-801"""},
                    {"role": "user", "content": f"""Translate this aircraft maintenance text from English to {language_names[target_language]}.
                    
                    IMPORTANT: Keep all technical terms, numbers, error codes and task references in their original form.
                    
                    Text to translate:
                    {simplified_result}"""}
                ],
                temperature=0.1,
                max_tokens=DEEPSEEK_API_CONFIG['max_tokens']
            )
            translated_text = response.choices[0].message.content.strip()
            rag_logger.debug(f"翻译完成，文本长度: {len(translated_text)}")
            
            return translated_text
        except Exception as e:
            rag_logger.error(f"翻译失败: {e}")
            return result  # 翻译失败时返回原文
    
    def _simplify_formatting(self, text: str) -> str:
        """
        简化文本格式，只保留基本结构和术语
        
        Args:
            text: 输入文本
            
        Returns:
            简化后的文本
        """
        # 简单处理，保留段落和基本结构
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # 保留行基本结构
            processed_lines.append(line)
        
        result = '\n'.join(processed_lines)
        
        # 确保技术术语保持原样
        tech_terms = ['CDS', 'BITE', 'DEU', 'CDU', 'CPU', 'AMM', 'FIM', 'SSM', 'WDM',
                      'ELEC', 'OVERHEAT', 'TEMP', 'IEMP', 'ZEMP', 'AEMP', 'SCEA', 'ALL']
        
        # 我们不做额外处理，只返回原始文本，让API自己处理
        return result

    def query(self, query: str, target_language: Optional[str] = None) -> Dict:
        """
        处理用户查询

        Args:
            query: 用户输入的查询文本
            target_language: 用户期望的输出语言 (例如 "zh", "en", "fr", 或 "跟随提问语言")

        Returns:
            Dict with 'answer' and 'sources'
        """
        # 新增：特殊文件名直出逻辑
        special_files = {
            "如何拆卸发动机灭火器": "data/如何拆卸发动机灭火器.md",
            "如何去除燃油滤清器元件": "data/如何去除燃油滤清器元件.md"
        }
        if query.strip() in special_files:
            file_path = special_files[query.strip()]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                # 可选：支持多语言翻译
                query_language = self._detect_language(query)
                final_output_language = query_language
                if target_language and target_language != "跟随提问语言":
                    lang_map = {"中文": "zh", "英文": "en", "法语": "fr"}
                    final_output_language = lang_map.get(target_language, query_language)
                if final_output_language != 'zh':
                    file_content = self._translate_result(file_content, final_output_language)
                return {'answer': file_content, 'sources': []}
            except Exception as e:
                return {'answer': f"读取知识库文件失败: {e}", 'sources': []}
        
        # 检测查询语言
        query_language = self._detect_language(query)
        rag_logger.debug(f"检测到查询语言: {query_language}")
        rag_logger.debug(f"用户指定输出语言: {target_language}")

        # 确定最终的输出语言
        final_output_language = query_language # 默认跟随提问语言
        if target_language and target_language != "跟随提问语言":
            # 将中文名称映射到语言代码
            lang_map = {
                "中文": "zh",
                "英文": "en",
                "法语": "fr"
            }
            final_output_language = lang_map.get(target_language, query_language)
        
        rag_logger.debug(f"最终输出语言确定为: {final_output_language}")
        
        # 尝试从查询中提取任务编号（如 31-62 TASK 801）
        task_pattern = re.compile(r'(\d{2}-\d{2}(?:-\d{2,3})?)\s*(?:TASK|任务)?\s*(\d{3})')
        task_match = task_pattern.search(query)
        
        if task_match:
            # 提取到任务编号
            task_section = task_match.group(1)
            task_number = task_match.group(2)
            fim_task = f"{task_section}-{task_number}"
            
            result_data = self.data_processor.get_result_by_fim_task(fim_task)
            if result_data:
                # 翻译结果
                translated_result = self._translate_result(result_data['result'], final_output_language)
                
                return {
                    'answer': translated_result,
                    'sources': [{
                        'fim_task': fim_task,
                        'title': result_data['title'],
                        'score': 1.0,  # 精确匹配，设置最高分
                    }]
                }
        
        # 尝试从查询中提取故障码
        # 移除所有空格并查找8位数字模式
        normalized_query = ''.join(query.split())
        fault_code_match = re.search(r'\d{8}', normalized_query)
        
        if fault_code_match:
            # 提取到故障码，直接查询对应的 FIM Task
            fault_code = fault_code_match.group()
            # 转换为标准格式 (XXX XXX XX)
            formatted_fault_code = f"{fault_code[:3]} {fault_code[3:6]} {fault_code[6:]}"
            
            fim_task = self.data_processor.get_fim_task_by_fault_code(formatted_fault_code)
            if fim_task:
                result_data = self.data_processor.get_result_by_fim_task(fim_task)
                if result_data:
                    # 获取故障描述
                    description = self.data_processor.get_fault_description_by_code(formatted_fault_code, final_output_language)
                    
                    # 翻译结果
                    translated_result = self._translate_result(result_data['result'], final_output_language)
                    
                    return {
                        'answer': translated_result,
                        'sources': [{
                            'fault_code': formatted_fault_code,
                            'description': description,
                            'fim_task': fim_task,
                            'title': result_data['title'],
                            'score': 1.0,  # 精确匹配，设置最高分
                        }]
                    }
        
        # 如果缓存中存在且未过期，直接返回缓存结果
        cache_key = f"{query}_{final_output_language}"
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if time.time() - cached_item['timestamp'] < CACHE_EXPIRY:
                logger.info(f"从缓存返回结果: {query}")
                # 更新时间戳以便LRU机制正常工作
                cached_item['timestamp'] = time.time()
                self._save_cache()
                return cached_item['data']
            else:
                # 缓存过期，删除
                del self.cache[cache_key]

        rag_logger.debug("执行向量存储检索")
        # 默认情况下，执行向量存储检索
        retrieved_docs = self.vector_store.search(query, query_language, top_k=5) # 默认检索5个文档
        
        if retrieved_docs:
            # 构建上下文信息
            context_parts = []
            for fault_code, description, fim_task, score in retrieved_docs:
                # 获取对应的结果数据
                result_data = self.data_processor.get_result_by_fim_task(fim_task)
                if result_data:
                    context_parts.append(f"FIM Task: {fim_task}\nDescription: {description}\nContent: {result_data['result'][:1000]}...")  # 限制长度
            
            context = "\n\n".join(context_parts)
            prompt = f"Context from maintenance manuals:\n{context}\n\nUser question: {query}\n\nPlease provide a specific answer based on the context above."
            
            # 使用DeepSeek生成答案
            rag_logger.debug(f"使用 DeepSeek API 生成答案，查询语言: {query_language}")
            response = self.client.chat.completions.create(
                model=DEEPSEEK_API_CONFIG['model'],
                messages=[
                    {"role": "system", "content": f"""You are a helpful AI assistant for aircraft maintenance.
                    Analyze the provided context from maintenance manuals and answer the user's question.
                    Focus on providing specific, actionable information based *only* on the provided context documents.
                    If the context doesn't contain the answer, state that clearly.
                    Keep your answer concise and to the point.
                    Always cite the source document and section if possible.
                    The user is asking in {query_language}. Respond in English, the translation will be handled later."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # 较低的温度以获得更确定的答案
                max_tokens=DEEPSEEK_API_CONFIG['max_tokens']
            )
            answer = response.choices[0].message.content.strip()
            rag_logger.debug(f"DeepSeek API 返回答案，长度: {len(answer)}")
            
            # 翻译和格式化最终答案
            translated_answer = self._translate_result(answer, final_output_language)
            rag_logger.debug(f"翻译和格式化完成，最终答案长度: {len(translated_answer)}")

            # 提取来源信息
            sources_info = []
            for fault_code, description, fim_task, score in retrieved_docs:
                result_data = self.data_processor.get_result_by_fim_task(fim_task)
                title = result_data['title'] if result_data else f"FIM Task {fim_task}"
                sources_info.append({
                    'fault_code': fault_code,
                    'description': description,
                    'fim_task': fim_task,
                    'title': title,
                    'score': score
                })
            
            # 缓存结果
            logger.info(f"缓存新结果: {query}")
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': {
                    'answer': translated_answer,
                    'sources': sources_info
                }
            }
            self._save_cache()
            
            return {
                'answer': translated_answer,
                'sources': sources_info
            }
        else:
            # 如果没有检索到相关文档，返回通用提示
            rag_logger.info("没有检索到相关文档")
            no_context_answer = "I couldn't find specific information in the available manuals to answer your question. You might want to rephrase your query or check the manual sections directly."
            
            # 翻译提示信息
            translated_no_context_answer = self._translate_result(no_context_answer, final_output_language)
            
            return {
                'answer': translated_no_context_answer,
                'sources': []
            } 