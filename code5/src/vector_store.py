"""
向量存储模块，负责文本向量化和相似度检索
"""
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger
from pathlib import Path
import pickle
from src.config import (
    VECTOR_STORE_PATHS,
    SUPPORTED_LANGUAGES,
    RETRIEVAL_CONFIG
)

class VectorStore:
    def __init__(self):
        """初始化向量存储"""
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.indexes: Dict[str, faiss.IndexFlatIP] = {}
        self.stored_texts: Dict[str, List[Tuple[str, str, str]]] = {}  # language -> [(fault_code, description, fim_task)]
        self._load_or_create_indexes()

    def _load_or_create_indexes(self) -> None:
        """加载或创建向量索引"""
        for lang in SUPPORTED_LANGUAGES:
            index_path = VECTOR_STORE_PATHS[lang]
            if not index_path.exists():
                logger.info(f"为语言 {lang} 创建新的向量索引")
                self.indexes[lang] = faiss.IndexFlatIP(768)  # mpnet-base 模型维度为768
                self.stored_texts[lang] = []
            else:
                try:
                    # 加载索引
                    self.indexes[lang] = faiss.read_index(str(index_path / "index.faiss"))
                    # 加载存储的文本
                    with open(index_path / "texts.pkl", 'rb') as f:
                        self.stored_texts[lang] = pickle.load(f)
                    logger.info(f"成功加载语言 {lang} 的向量索引")
                except Exception as e:
                    logger.error(f"加载语言 {lang} 的向量索引失败: {e}")
                    raise

    def add_texts(self, texts: List[Tuple[str, str, str]], language: str) -> None:
        """添加文本到向量存储

        Args:
            texts: List of (fault_code, description, fim_task)
            language: 语言代码
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的语言: {language}")

        # 编码文本
        descriptions = [text[1] for text in texts]
        embeddings = self.model.encode(descriptions, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy()

        # 添加到索引
        self.indexes[language].add(embeddings_np)
        self.stored_texts[language].extend(texts)

        # 保存索引和文本
        self._save_index(language)

    def _save_index(self, language: str) -> None:
        """保存索引到文件"""
        index_dir = VECTOR_STORE_PATHS[language]
        index_dir.mkdir(parents=True, exist_ok=True)

        # 保存faiss索引
        faiss.write_index(self.indexes[language], str(index_dir / "index.faiss"))

        # 保存文本数据
        with open(index_dir / "texts.pkl", 'wb') as f:
            pickle.dump(self.stored_texts[language], f)

        logger.info(f"已保存语言 {language} 的向量索引")

    def search(self, query: str, language: str, top_k: int = None) -> List[Tuple[str, str, str, float]]:
        """搜索最相似的文本

        Args:
            query: 查询文本
            language: 语言代码
            top_k: 返回结果数量，默认使用配置值

        Returns:
            List of (fault_code, description, fim_task, score)
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的语言: {language}")

        if top_k is None:
            top_k = RETRIEVAL_CONFIG['top_k']

        # 检查是否有数据
        if not self.stored_texts[language] or self.indexes[language].ntotal == 0:
            logger.warning(f"语言 {language} 的向量索引为空")
            return []

        # 编码查询
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy()

        # 搜索
        scores, indices = self.indexes[language].search(query_embedding_np, top_k)

        # 整理结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < RETRIEVAL_CONFIG['score_threshold']:
                continue
            if idx >= len(self.stored_texts[language]):
                logger.warning(f"索引 {idx} 超出范围")
                continue
            fault_code, description, fim_task = self.stored_texts[language][idx]
            results.append((fault_code, description, fim_task, float(score)))

        return sorted(results, key=lambda x: x[3], reverse=True)

    def clear_index(self, language: str) -> None:
        """清除指定语言的索引

        Args:
            language: 语言代码
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的语言: {language}")

        # 重新创建索引
        self.indexes[language] = faiss.IndexFlatIP(768)
        self.stored_texts[language] = []

        # 删除索引文件
        index_dir = VECTOR_STORE_PATHS[language]
        if index_dir.exists():
            for file in index_dir.glob("*"):
                file.unlink()
            index_dir.rmdir()

        logger.info(f"已清除语言 {language} 的向量索引")

    def get_index_stats(self, language: str) -> Dict[str, int]:
        """获取索引统计信息

        Args:
            language: 语言代码

        Returns:
            Dict with statistics
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的语言: {language}")

        return {
            "total_vectors": len(self.stored_texts[language]),
            "dimension": self.indexes[language].d if language in self.indexes else 0
        }