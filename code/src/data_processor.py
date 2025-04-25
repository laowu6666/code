"""
数据处理模块，负责CSV文件的读取和预处理
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
from src.config import (
    DIRECTORY_CSV,
    RESULT_CSV,
    SUPPORTED_LANGUAGES
)

class DataProcessor:
    def __init__(self):
        """初始化数据处理器"""
        self.directory_df = None
        self.result_df = None
        self.load_data()

    def load_data(self) -> None:
        """加载CSV文件数据"""
        try:
            self.directory_df = pd.read_csv(DIRECTORY_CSV)
            self.result_df = pd.read_csv(RESULT_CSV)
            logger.info("数据文件加载成功")
        except Exception as e:
            logger.error(f"数据文件加载失败: {e}")
            raise

    @staticmethod
    def normalize_fault_code(fault_code: str) -> str:
        """
        标准化故障码格式，移除所有空格
        
        Args:
            fault_code: 原始故障码
            
        Returns:
            标准化后的故障码
        """
        if not fault_code:
            return ""
        return ''.join(fault_code.split())

    def get_fault_descriptions(self, language: str) -> List[Tuple[str, str, str]]:
        """
        获取指定语言的故障描述数据

        Args:
            language: 语言代码 ('zh', 'en', 'fr')

        Returns:
            List of tuples (Fault Code, Fault Description, FIM Task)
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的语言: {language}")

        description_column = {
            'zh': 'Fault Description (中文)',
            'en': 'Fault Description (English)',
            'fr': 'Fault Description (Français)'
        }[language]

        return list(zip(
            self.directory_df['Fault Code'],
            self.directory_df[description_column],
            self.directory_df['Go To FIM Task']
        ))

    def get_result_by_fim_task(self, fim_task: str) -> Optional[Dict[str, str]]:
        """
        根据FIM Task获取处理结果

        Args:
            fim_task: FIM Task编号

        Returns:
            Dict with 'title' and 'result' if found, None otherwise
        """
        result = self.result_df[self.result_df['Go To FIM Task'] == fim_task]
        if len(result) == 0:
            logger.warning(f"未找到FIM Task对应的结果: {fim_task}")
            return None

        return {
            'title': result.iloc[0]['title'],
            'result': result.iloc[0]['result']
        }

    def get_all_fault_codes(self) -> List[str]:
        """获取所有故障码"""
        return self.directory_df['Fault Code'].tolist()

    def get_fim_task_by_fault_code(self, fault_code: str) -> Optional[str]:
        """
        根据故障码获取FIM Task

        Args:
            fault_code: 故障码

        Returns:
            FIM Task if found, None otherwise
        """
        # 标准化输入的故障码
        normalized_input = self.normalize_fault_code(fault_code)
        
        # 获取所有故障码并标准化
        fault_codes = self.directory_df['Fault Code'].apply(self.normalize_fault_code)
        
        # 在标准化后的故障码中查找
        mask = fault_codes == normalized_input
        result = self.directory_df[mask]
        
        if len(result) == 0:
            logger.warning(f"未找到故障码对应的FIM Task: {fault_code}")
            return None
        return result.iloc[0]['Go To FIM Task']

    def get_fault_description_by_code(self, fault_code: str, language: str) -> Optional[str]:
        """
        根据故障码获取指定语言的故障描述

        Args:
            fault_code: 故障码
            language: 语言代码 ('zh', 'en', 'fr')

        Returns:
            故障描述 if found, None otherwise
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的语言: {language}")

        description_column = {
            'zh': 'Fault Description (中文)',
            'en': 'Fault Description (English)',
            'fr': 'Fault Description (Français)'
        }[language]

        # 标准化输入的故障码
        normalized_input = self.normalize_fault_code(fault_code)
        
        # 获取所有故障码并标准化
        fault_codes = self.directory_df['Fault Code'].apply(self.normalize_fault_code)
        
        # 在标准化后的故障码中查找
        mask = fault_codes == normalized_input
        result = self.directory_df[mask]
        
        if len(result) == 0:
            logger.warning(f"未找到故障码: {fault_code}")
            return None
        return result.iloc[0][description_column]

    def normalize_fim_task(self, fim_task: str) -> str:
        """标准化FIM Task ID的格式
        
        处理以下情况：
        1. 移除多余的空格
        2. 统一 'TASK' 前后的空格
        3. 处理可能的特殊字符
        """
        if not fim_task:
            return ""
        
        # 如果是完整的法语描述，直接返回
        if fim_task and fim_task.startswith(('Indication', 'Levier', 'Jauge')):
            return fim_task
        
        # 标准化处理
        normalized = fim_task.strip()
        # 移除TASK前后的空格
        normalized = normalized.replace(' TASK ', 'TASK')
        normalized = normalized.replace('TASK ', 'TASK')
        normalized = normalized.replace(' TASK', 'TASK')
        
        return normalized

    def validate_data(self) -> bool:
        """验证数据完整性"""
        try:
            # 检查必要的列是否存在
            required_directory_columns = [
                'Fault Code',
                'Fault Description (中文)',
                'Fault Description (English)',
                'Fault Description (Français)',
                'Go To FIM Task'
            ]
            required_result_columns = ['Go To FIM Task', 'title', 'result']

            # 检查必要的列
            for col in required_directory_columns:
                if col not in self.directory_df.columns:
                    logger.error(f"directory.csv 缺少必要的列: {col}")
                    return False

            for col in required_result_columns:
                if col not in self.result_df.columns:
                    logger.error(f"result.csv 缺少必要的列: {col}")
                    return False

            # 获取所有需要的FIM Task
            directory_fim_tasks = set(self.directory_df['Go To FIM Task'].apply(self.normalize_fim_task))
            result_fim_tasks = set(self.result_df['Go To FIM Task'].apply(self.normalize_fim_task))
            
            # 检查缺失的任务
            missing_tasks = directory_fim_tasks - result_fim_tasks
            if missing_tasks:
                logger.warning(f"以下FIM Task在result.csv中缺失: {missing_tasks}")
                # 计算缺失率
                missing_rate = len(missing_tasks) / len(directory_fim_tasks)
                if missing_rate > 0.3:  # 如果缺失率超过30%，则认为数据不完整
                    logger.error(f"缺失的FIM Task过多，缺失率: {missing_rate:.1%}")
                    return False
                else:
                    logger.warning(f"发现 {len(missing_tasks)} 个缺失的FIM Task，缺失率: {missing_rate:.1%}，继续处理")

            return True
        
        except Exception as e:
            logger.error(f"数据验证失败: {str(e)}")
            return False 