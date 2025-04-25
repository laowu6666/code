"""
索引构建脚本，用于创建多语言向量索引
"""
import sys
from pathlib import Path
import time
from loguru import logger

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data_processor import DataProcessor
from src.vector_store import VectorStore
from src.config import SUPPORTED_LANGUAGES

def setup_logger():
    """配置日志"""
    logger.add(
        ROOT_DIR / "logs/index_creation.log",
        rotation="100 MB",
        retention="10 days",
        level="INFO"
    )

def create_indexes() -> None:
    """创建所有语言的向量索引"""
    try:
        # 初始化数据处理器和向量存储
        data_processor = DataProcessor()
        vector_store = VectorStore()

        # 验证数据完整性
        if not data_processor.validate_data():
            logger.error("数据验证失败，请检查CSV文件")
            return

        # 为每种语言创建索引
        for lang in SUPPORTED_LANGUAGES:
            logger.info(f"开始处理 {SUPPORTED_LANGUAGES[lang]} 索引")
            
            # 获取该语言的故障描述数据
            fault_descriptions = data_processor.get_fault_descriptions(lang)
            total_count = len(fault_descriptions)

            if total_count == 0:
                logger.warning(f"语言 {lang} 没有找到故障描述数据")
                continue

            # 清除现有索引
            vector_store.clear_index(lang)

            # 批量添加文本到索引
            batch_size = 100
            for i in range(0, total_count, batch_size):
                batch = fault_descriptions[i:i+batch_size]
                vector_store.add_texts(batch, lang)
                
                # 显示进度
                progress = min(100.0, (i + batch_size) / total_count * 100)
                logger.info(f"处理进度: {progress:.1f}% ({i+len(batch)}/{total_count})")

            # 获取并显示索引统计信息
            stats = vector_store.get_index_stats(lang)
            logger.info(f"{SUPPORTED_LANGUAGES[lang]} 索引统计:")
            logger.info(f"- 总向量数: {stats['total_vectors']}")
            logger.info(f"- 向量维度: {stats['dimension']}")

        logger.info("所有语言的索引创建完成")

    except Exception as e:
        error_msg = f"索引创建过程中出错: {str(e)}"
        logger.error(error_msg)
        raise

def main():
    """主函数"""
    print("开始创建多语言向量索引...")
    
    # 设置日志
    setup_logger()

    start_time = time.time()
    
    try:
        create_indexes()
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"索引创建完成！总耗时: {duration:.2f} 秒")
        print(f"索引创建完成！总耗时: {duration:.2f} 秒")
        
    except Exception as e:
        logger.exception("索引创建失败")
        print(f"创建索引时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 