"""
向量存储模块的单元测试
"""
import pytest
import numpy as np
from pathlib import Path
import shutil
from src.vector_store import VectorStore
from src.config import VECTOR_STORE_PATHS, SUPPORTED_LANGUAGES

@pytest.fixture(scope="module")
def test_data():
    """测试数据"""
    return [
        ("282 010 48", "Refuel quantity indicator: blank - all tanks.", "28-21 TASK 804"),
        ("282 011 48", "Refuel quantity indicator: Display is not correct.", "28-21 TASK 802"),
        ("282 012 48", "Refuel quantity indicator: Shows Ind FAlL.", "28-21 TASK 805")
    ]

@pytest.fixture(scope="module")
def vector_store():
    """创建 VectorStore 实例"""
    # 清理现有的向量存储
    for lang in SUPPORTED_LANGUAGES:
        if VECTOR_STORE_PATHS[lang].exists():
            shutil.rmtree(VECTOR_STORE_PATHS[lang])
    
    return VectorStore()

def test_initialization(vector_store):
    """测试初始化"""
    assert vector_store.model is not None
    assert isinstance(vector_store.indexes, dict)
    assert isinstance(vector_store.stored_texts, dict)
    for lang in SUPPORTED_LANGUAGES:
        assert lang in vector_store.indexes
        assert lang in vector_store.stored_texts

def test_add_and_search(vector_store, test_data):
    """测试添加和搜索功能"""
    # 添加英文测试数据
    vector_store.add_texts(test_data, 'en')
    
    # 测试搜索
    query = "blank refuel indicator"
    results = vector_store.search(query, 'en', top_k=2)
    
    assert len(results) > 0
    assert all(isinstance(r, tuple) and len(r) == 4 for r in results)
    assert all(isinstance(r[3], float) for r in results)  # 检查相似度分数
    
    # 验证最相关的结果
    best_match = results[0]
    assert "blank" in best_match[1].lower()
    assert "refuel" in best_match[1].lower()

def test_clear_index(vector_store, test_data):
    """测试清除索引功能"""
    # 先添加一些数据
    vector_store.add_texts(test_data, 'en')
    
    # 获取初始统计信息
    initial_stats = vector_store.get_index_stats('en')
    assert initial_stats['total_vectors'] > 0
    
    # 清除索引
    vector_store.clear_index('en')
    
    # 验证索引已清除
    after_clear_stats = vector_store.get_index_stats('en')
    assert after_clear_stats['total_vectors'] == 0

def test_index_persistence(test_data):
    """测试索引持久化"""
    # 创建新的实例并添加数据
    vs1 = VectorStore()
    vs1.add_texts(test_data, 'en')
    
    # 创建另一个实例，验证数据是否正确加载
    vs2 = VectorStore()
    stats = vs2.get_index_stats('en')
    assert stats['total_vectors'] == len(test_data)
    
    # 验证搜索结果
    results = vs2.search("blank indicator", 'en')
    assert len(results) > 0

def test_multilingual_support(vector_store, test_data):
    """测试多语言支持"""
    # 测试每种支持的语言
    for lang in SUPPORTED_LANGUAGES:
        # 添加数据
        vector_store.add_texts(test_data, lang)
        
        # 验证索引创建
        stats = vector_store.get_index_stats(lang)
        assert stats['total_vectors'] == len(test_data)
        
        # 验证搜索功能
        results = vector_store.search("refuel", lang)
        assert len(results) > 0

def test_error_handling(vector_store, test_data):
    """测试错误处理"""
    # 测试不支持的语言
    with pytest.raises(ValueError):
        vector_store.add_texts(test_data, "unsupported")
    
    with pytest.raises(ValueError):
        vector_store.search("query", "unsupported")
    
    with pytest.raises(ValueError):
        vector_store.clear_index("unsupported")
    
    with pytest.raises(ValueError):
        vector_store.get_index_stats("unsupported") 