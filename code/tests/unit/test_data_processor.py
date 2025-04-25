"""
数据处理模块的单元测试
"""
import pytest
import pandas as pd
from src.data_processor import DataProcessor
from src.config import SUPPORTED_LANGUAGES

@pytest.fixture
def data_processor():
    """创建 DataProcessor 实例"""
    return DataProcessor()

def test_load_data(data_processor):
    """测试数据加载功能"""
    assert data_processor.directory_df is not None
    assert data_processor.result_df is not None
    assert isinstance(data_processor.directory_df, pd.DataFrame)
    assert isinstance(data_processor.result_df, pd.DataFrame)

def test_get_fault_descriptions(data_processor):
    """测试获取故障描述功能"""
    for lang in SUPPORTED_LANGUAGES:
        descriptions = data_processor.get_fault_descriptions(lang)
        assert isinstance(descriptions, list)
        assert len(descriptions) > 0
        assert all(isinstance(desc, tuple) and len(desc) == 3 for desc in descriptions)

def test_get_result_by_fim_task(data_processor):
    """测试获取处理结果功能"""
    # 使用已知存在的 FIM Task
    fim_task = "28-21 TASK 804"
    result = data_processor.get_result_by_fim_task(fim_task)
    assert result is not None
    assert isinstance(result, dict)
    assert 'title' in result
    assert 'result' in result

    # 测试不存在的 FIM Task
    result = data_processor.get_result_by_fim_task("non-existent-task")
    assert result is None

def test_get_fim_task_by_fault_code(data_processor):
    """测试通过故障码获取 FIM Task 功能"""
    # 使用已知存在的故障码
    fault_code = "282 010 48"
    fim_task = data_processor.get_fim_task_by_fault_code(fault_code)
    assert fim_task is not None
    assert isinstance(fim_task, str)

    # 测试不存在的故障码
    fim_task = data_processor.get_fim_task_by_fault_code("non-existent-code")
    assert fim_task is None

def test_get_fault_description_by_code(data_processor):
    """测试获取故障描述功能"""
    fault_code = "282 010 48"
    for lang in SUPPORTED_LANGUAGES:
        description = data_processor.get_fault_description_by_code(fault_code, lang)
        assert description is not None
        assert isinstance(description, str)

def test_validate_data(data_processor):
    """测试数据验证功能"""
    # 检查基本的数据结构
    assert data_processor.directory_df is not None
    assert data_processor.result_df is not None
    
    # 检查必要的列是否存在
    required_directory_columns = [
        'Fault Code',
        'Fault Description (中文)',
        'Fault Description (English)',
        'Fault Description (Français)',
        'Go To FIM Task'
    ]
    required_result_columns = ['Go To FIM Task', 'title', 'result']
    
    for col in required_directory_columns:
        assert col in data_processor.directory_df.columns
        
    for col in required_result_columns:
        assert col in data_processor.result_df.columns
    
    # 检查是否有基本的数据
    assert len(data_processor.directory_df) > 0
    assert len(data_processor.result_df) > 0
    
    # 检查至少有一些 FIM Task 是匹配的
    directory_fim_tasks = set(data_processor.directory_df['Go To FIM Task'])
    result_fim_tasks = set(data_processor.result_df['Go To FIM Task'])
    assert len(directory_fim_tasks.intersection(result_fim_tasks)) > 0

def test_unsupported_language(data_processor):
    """测试不支持的语言处理"""
    with pytest.raises(ValueError):
        data_processor.get_fault_descriptions("unsupported")

    with pytest.raises(ValueError):
        data_processor.get_fault_description_by_code("282 010 48", "unsupported") 