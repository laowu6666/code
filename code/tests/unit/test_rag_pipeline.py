"""
RAG Pipeline 模块的单元测试
"""
import pytest
import json
import time
from pathlib import Path
import shutil
from src.rag_pipeline import RAGPipeline
from src.config import CACHE_DIR, CACHE_EXPIRY

def clean_cache_dir():
    """清理缓存目录"""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    """在每个测试前后清理缓存"""
    clean_cache_dir()
    yield
    clean_cache_dir()

@pytest.fixture(scope="function")
def rag_pipeline():
    """创建 RAGPipeline 实例"""
    return RAGPipeline()

@pytest.fixture(scope="module")
def test_queries():
    """测试查询"""
    return {
        'zh': "燃油量指示器显示空白",
        'en': "fuel quantity indicator blank",
        'fr': "indicateur de quantité de carburant vide"
    }

def test_initialization(rag_pipeline):
    """测试初始化"""
    assert rag_pipeline.data_processor is not None
    assert rag_pipeline.vector_store is not None
    assert rag_pipeline.model is not None
    assert isinstance(rag_pipeline.cache, dict)

def test_language_detection(rag_pipeline):
    """测试语言检测"""
    # 测试中文检测
    assert rag_pipeline._detect_language("燃油系统故障") == 'zh'
    
    # 测试英文检测
    assert rag_pipeline._detect_language("fuel system fault") == 'en'
    
    # 测试法语检测
    assert rag_pipeline._detect_language("panne du système de carburant") == 'fr'
    
    # 测试默认语言（不支持的语言）
    assert rag_pipeline._detect_language("未知语言测试") == 'zh'

def test_query_processing(rag_pipeline, test_queries):
    """测试查询处理"""
    # 测试每种语言的查询
    for lang, query in test_queries.items():
        response = rag_pipeline.query(query)
        
        # 验证响应格式
        assert isinstance(response, dict)
        assert 'answer' in response
        assert 'sources' in response
        
        # 验证源信息
        if response['sources']:
            source = response['sources'][0]
            assert 'fault_code' in source
            assert 'description' in source
            assert 'fim_task' in source
            assert 'title' in source
            assert 'score' in source

def test_caching(rag_pipeline, test_queries):
    """测试缓存机制"""
    # 确保缓存目录为空
    assert CACHE_DIR.exists()
    assert len(list(CACHE_DIR.glob("*"))) == 0, "缓存目录不为空"

    # 首次查询
    query = test_queries['zh']
    first_response = rag_pipeline.query(query)
    
    # 等待文件系统同步
    time.sleep(1)
    
    # 验证缓存文件创建
    cache_file = CACHE_DIR / "response_cache.json"
    assert cache_file.exists(), "缓存文件未创建"
    
    # 读取缓存内容
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    
    # 验证缓存内容
    cache_key = f"zh:{query}"
    assert cache_key in cache, f"缓存键 {cache_key} 不存在"
    assert 'answer' in cache[cache_key], "缓存中缺少 answer 字段"
    assert 'sources' in cache[cache_key], "缓存中缺少 sources 字段"
    assert 'timestamp' in cache[cache_key], "缓存中缺少 timestamp 字段"
    
    # 再次查询，验证使用缓存
    second_response = rag_pipeline.query(query)
    assert first_response == second_response, "缓存响应与原始响应不匹配"

def test_cache_expiry(rag_pipeline, test_queries):
    """测试缓存过期"""
    query = test_queries['en']
    
    # 创建过期缓存
    cache_key = f"en:{query}"
    expired_cache = {
        cache_key: {
            'answer': 'Cached response',
            'sources': [],
            'timestamp': time.time() - (CACHE_EXPIRY + 100)  # 设置为过期
        }
    }
    
    # 写入过期缓存
    cache_file = CACHE_DIR / "response_cache.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(expired_cache, f)
    
    # 重新加载 pipeline 以加载新缓存
    new_pipeline = RAGPipeline()
    
    # 验证过期缓存不被使用
    response = new_pipeline.query(query)
    assert response['answer'] != 'Cached response'

def test_error_handling(rag_pipeline):
    """测试错误处理"""
    # 测试空查询
    response = rag_pipeline.query("")
    assert isinstance(response, dict)
    assert 'answer' in response
    assert 'sources' in response
    
    # 测试异常长查询
    long_query = "test " * 1000
    response = rag_pipeline.query(long_query)
    assert isinstance(response, dict)
    assert 'answer' in response
    assert 'sources' in response 