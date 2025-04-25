import asyncio
import torch
import logging

# 设置新的事件循环
asyncio.set_event_loop(asyncio.new_event_loop())

# 修复 torch.classes.__path__ 的问题
torch.classes.__path__ = [torch.__path__[0]]

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 其他导入
import os
import sys
from pathlib import Path
import re
import urllib.parse
import json
import time
import streamlit as st

# Add project root to Python path before any other imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ['PYTHONPATH'] = str(ROOT_DIR)

# 防止PyTorch类监视错误
os.environ['STREAMLIT_WATCH_EXCLUDE_MODULES'] = 'torch'

# 添加日志记录到文件
logging.basicConfig(
    filename='debug_markdown.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # 每次覆盖日志文件
)
logger = logging.getLogger('markdown_debug')

import streamlit as st

# 定义Markdown预处理函数，用于优化渲染效果
def preprocess_markdown(text):
    """
    处理Markdown文本以确保在Streamlit中正确渲染
    
    Args:
        text: 输入的Markdown文本
        
    Returns:
        处理后的Markdown文本
    """
    # 记录原始文本
    logger.debug(f"预处理前的原始Markdown文本长度: {len(text or '')}")
    if text and len(text) < 300:  # 只记录较短的文本全文，避免日志过大
        logger.debug(f"预处理前的原始Markdown文本:\n{text}")
    else:
        # 记录文本的开头和结尾部分
        logger.debug(f"预处理前的原始Markdown文本 (截断):\n{text[:150] if text else ''}...\n...\n{text[-150:] if text else ''}")
    
    # 处理任务链接
    # 将 #task=XXX 格式转换为可点击的按钮
    if text and "#task=" in text:
        # 匹配形如 [查看完整任务 31-62-801](#task=31-62-801) 或其他语言版本的链接
        task_link_pattern = r'\[(.*?)\]\(#task=([^\)]+)\)'
        
        def replace_task_link(match):
            link_text = match.group(1)
            task_id = match.group(2)
            # 处理不同格式的任务ID (例如 31-62 TASK 801 vs 31-62-801)
            cleaned_task_id = task_id
            # 如果包含"TASK"，提取数字部分
            if "TASK" in cleaned_task_id:
                # 提取前缀(如31-62)和任务号(801)
                parts = re.match(r'(\d+(?:-\d+)+)\s*(?:TASK|任务)\s*(\d+)', cleaned_task_id)
                if parts:
                    cleaned_task_id = f"{parts.group(1)}-{parts.group(2)}"
            
            # 确保任务ID格式标准化
            cleaned_task_id = cleaned_task_id.strip()
            # 创建一个普通的Markdown链接
            new_link = f'[{link_text}](/?task={urllib.parse.quote(cleaned_task_id)})'
            logger.debug(f"转换任务链接: {link_text} -> {new_link}")
            return new_link
            
        text = re.sub(task_link_pattern, replace_task_link, text)
        logger.debug(f"处理后的任务链接: {text[:200]}...")

    # 识别文本中直接出现的任务引用，将其转换为可点击链接
    # 例如：31-62 TASK 801 或 31-62-801
    task_reference_pattern = r'(\d{2}-\d{2}(?:-\d{2,3})?)\s+(?:TASK|任务)\s+(\d{3})'
    
    def replace_task_reference(match):
        section = match.group(1)
        number = match.group(2)
        task_id = f"{section}-{number}"
        # 创建一个带自定义样式的HTML链接，但不添加图标，避免重复
        new_link = f'<a href="/?task={urllib.parse.quote(task_id)}" class="task-link">{match.group(0)}</a>'
        logger.debug(f"转换任务引用: {match.group(0)} -> HTML链接")
        return new_link
        
    # 仅处理不在代码块内的任务引用
    if "```" in text:
        # 分割代码块和非代码块
        parts = text.split("```")
        # 奇数索引是代码块(从0开始)
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(task_reference_pattern, replace_task_reference, parts[i])
        text = "```".join(parts)
    else:
        text = re.sub(task_reference_pattern, replace_task_reference, text)
    
    # 处理可能已经存在的HTML任务链接 (例如 <a href="/?task=31-62-801">...)
    html_task_pattern = r'<a\s+href="[^"]*task=([^"]+)"[^>]*>(.*?)<\/a>'
    
    def replace_html_task(match):
        task_id = match.group(1)
        link_text = match.group(2)
        # 转换为带自定义样式的HTML链接，但不添加图标
        new_link = f'<a href="/?task={task_id}" class="task-link">{link_text}</a>'
        logger.debug(f"转换HTML任务链接: <a>...</a> -> 自定义HTML链接")
        return new_link
    
    # 处理HTML任务链接
    if '<a href=' in text and 'task=' in text:
        text = re.sub(html_task_pattern, replace_html_task, text)
    
    # 简化渲染处理，只保留基本结构
    if text:
        # 处理可能的连续空行
        lines = text.split('\n')
        processed_lines = []
        prev_empty = False
        
        for line in lines:
            is_empty = not line.strip()
            
            # 跳过连续空行
            if is_empty and prev_empty:
                continue
                
            processed_lines.append(line)
            prev_empty = is_empty
            
        text = '\n'.join(processed_lines)
    
    logger.debug(f"预处理后的文本长度: {len(text)}")
    return text

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Attempt to import the RAG pipeline
try:
    from src.rag_pipeline import RAGPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    st.set_page_config(page_title="错误", page_icon="⚠️", layout="wide")
    st.title("⚠️ 应用加载错误")
    st.error(f"无法导入 RAG pipeline: {e}。请检查 src/rag_pipeline.py 及相关依赖是否正确安装。", icon="⚠️")
    PIPELINE_AVAILABLE = False
    st.stop()
except Exception as e:
    st.set_page_config(page_title="错误", page_icon="⚠️", layout="wide")
    st.title("⚠️ 应用加载错误")
    st.error(f"加载 RAG pipeline 时发生未知错误: {e}", icon="⚠️")
    PIPELINE_AVAILABLE = False
    st.stop()

# --- Page Configuration ---
# Configure the page only if imports were successful
st.set_page_config(
    page_title="航维智询 | 航空维修知识智能问答系统",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded" # Expand sidebar by default
)

# Initialize session state for UI control
if "show_manual_manager" not in st.session_state:
    st.session_state.show_manual_manager = False
    
# 添加任务查看状态
if "viewing_task" not in st.session_state:
    st.session_state.viewing_task = None
    
# 添加任务历史记录
if "task_history" not in st.session_state:
    st.session_state.task_history = []

# Function to toggle manual manager view
def toggle_manual_manager():
    st.session_state.show_manual_manager = not st.session_state.show_manual_manager
    
# 函数：设置当前查看的任务
def set_viewing_task(task_id):
    st.session_state.viewing_task = task_id
    # 添加到历史记录，避免重复
    if task_id and task_id not in st.session_state.task_history:
        st.session_state.task_history.append(task_id)
        # 保持历史记录最多10项
        if len(st.session_state.task_history) > 10:
            st.session_state.task_history.pop(0)
    
# 函数：清除当前查看的任务
def clear_viewing_task():
    st.session_state.viewing_task = None

# --- Sidebar Content ---
st.sidebar.title("✈️ 航维智询")
st.sidebar.caption("航空维修知识智能问答系统")

# 添加聊天历史功能
st.sidebar.header("💬 聊天记录")

# 添加聊天记录持久化的相关函数
def save_chat_sessions(chat_sessions):
    """将聊天会话保存到文件"""
    try:
        chat_dir = Path("chat_history")
        chat_dir.mkdir(exist_ok=True)
        
        # 保存时添加时间戳以防止覆盖
        timestamp = int(time.time())
        chat_file = chat_dir / f"chat_sessions_{timestamp}.json"
        
        # 如果存在太多历史文件，只保留最新的10个
        chat_files = list(chat_dir.glob("chat_sessions_*.json"))
        if len(chat_files) > 10:
            # 按修改时间排序
            chat_files.sort(key=lambda x: x.stat().st_mtime)
            # 删除最旧的文件
            for old_file in chat_files[:-10]:
                old_file.unlink()
        
        # 保存当前会话数据
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_sessions, f, ensure_ascii=False, indent=2)
        
        # 保存最新会话的引用
        latest_file = chat_dir / "latest_sessions.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(chat_sessions, f, ensure_ascii=False, indent=2)
            
        logger.debug(f"聊天会话已保存到: {chat_file}")
        return True
    except Exception as e:
        logger.error(f"保存聊天会话失败: {str(e)}")
        return False

def load_chat_sessions():
    """从文件加载聊天会话"""
    try:
        latest_file = Path("chat_history") / "latest_sessions.json"
        if latest_file.exists():
            with open(latest_file, 'r', encoding='utf-8') as f:
                chat_sessions = json.load(f)
            logger.debug(f"从 {latest_file} 加载了聊天会话")
            return chat_sessions
        else:
            logger.debug("没有找到保存的聊天会话文件")
            return {"默认会话": []}
    except Exception as e:
        logger.error(f"加载聊天会话失败: {str(e)}")
        return {"默认会话": []}

# 初始化聊天会话管理状态
if "chat_sessions" not in st.session_state:
    # 从文件加载聊天会话
    loaded_sessions = load_chat_sessions()
    st.session_state.chat_sessions = loaded_sessions
    # 如果没有会话或会话为空，创建一个默认会话
    if not loaded_sessions:
        st.session_state.chat_sessions = {"默认会话": []}
    
    # 设置当前会话为第一个会话
    st.session_state.current_session = next(iter(st.session_state.chat_sessions.keys()))
    st.session_state.session_counter = len(st.session_state.chat_sessions) + 1

# 如果存在旧的消息记录，将其迁移到第一个会话中
if "messages" in st.session_state and "chat_sessions" in st.session_state:
    if st.session_state.messages and "默认会话" in st.session_state.chat_sessions:
        # 只有在第一次加载时迁移
        if len(st.session_state.chat_sessions["默认会话"]) == 0:
            st.session_state.chat_sessions["默认会话"] = st.session_state.messages.copy()

# 会话管理功能
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    # 显示当前会话名称
    st.markdown(f"**当前会话:** {st.session_state.current_session}")
with col2:
    # 添加新会话按钮
    if st.button("➕", help="新建会话"):
        # 创建新会话
        session_name = f"会话 {st.session_state.session_counter}"
        st.session_state.chat_sessions[session_name] = []
        st.session_state.current_session = session_name
        st.session_state.session_counter += 1
        # 保存更新后的会话状态
        save_chat_sessions(st.session_state.chat_sessions)
        st.rerun()

# 显示会话列表
for session_name in st.session_state.chat_sessions.keys():
    col1, col2 = st.sidebar.columns([4, 1])
    # 计算会话中的消息数
    msg_count = len(st.session_state.chat_sessions[session_name])
    # 获取最后一条用户消息作为标题(如果有)
    session_title = session_name
    if msg_count > 0:
        for msg in reversed(st.session_state.chat_sessions[session_name]):
            if msg["role"] == "user":
                # 截取前20个字符作为标题
                session_title = msg["content"][:20] + ("..." if len(msg["content"]) > 20 else "")
                break
    
    # 检查是否为当前会话
    is_current = session_name == st.session_state.current_session
    
    with col1:
        # 使用按钮来显示会话，点击切换到该会话
        if st.button(
            f"{'🔹' if is_current else '🔸'} {session_title}", 
            key=f"session_{session_name}",
            help=f"消息数: {msg_count}",
            use_container_width=True
        ):
            st.session_state.current_session = session_name
            # 将选中会话的消息同步到messages中
            st.session_state.messages = st.session_state.chat_sessions[session_name].copy()
            st.rerun()
    
    with col2:
        # 添加删除按钮
        if not is_current and st.button("🗑️", key=f"del_{session_name}", help="删除此会话"):
            del st.session_state.chat_sessions[session_name]
            # 如果没有会话了，创建一个新的默认会话
            if not st.session_state.chat_sessions:
                st.session_state.chat_sessions["新会话"] = []
                st.session_state.current_session = "新会话"
            # 如果删除的是当前会话，切换到第一个可用会话
            elif session_name == st.session_state.current_session:
                st.session_state.current_session = next(iter(st.session_state.chat_sessions.keys()))
            # 同步当前会话到messages
            st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_session].copy()
            # 保存更新后的会话状态
            save_chat_sessions(st.session_state.chat_sessions)
            st.rerun()

# 侧边栏分隔线
st.sidebar.divider()

# --- Language Settings UI ---
st.sidebar.header("🌐 语言设置")
with st.sidebar.expander("多语言问答设置", expanded=True):
    st.markdown("**输出语言**")
    output_language = st.radio(
        "选择回答语言",
        options=["跟随提问语言", "中文", "英文", "法语"],
        horizontal=True,
        index=0
    )

# --- Manual Management UI toggle ---
st.sidebar.header("📚 手册管理")
st.sidebar.button("打开手册管理窗口", on_click=toggle_manual_manager, type="secondary" if not st.session_state.show_manual_manager else "primary")

st.sidebar.header("关于")
st.sidebar.info("本系统利用检索增强生成 (RAG) 技术，旨在解决航空维修人员在查阅手册时遇到的资料过载、检索低效等问题。")

st.sidebar.header("核心痛点")
with st.sidebar.expander("维修人员面临的挑战", expanded=True):
    st.markdown("""
    *   **资料过载**: 海量 PDF 文档难消化。
    *   **检索低效**: 关键词搜索难以理解复杂意图。
    *   **时间压力**: 快速获取准确信息至关重要。
    *   **信息碎片化**: 内容分散，需手动整合。
    *   **经验依赖**: 新手难以快速定位知识。
    """)

st.sidebar.header("系统优势")
with st.sidebar.expander("本系统如何提供帮助", expanded=True):
    st.markdown("""
    *   **智能检索**: 理解问题语境和专业术语。
    *   **多源整合**: 自动关联分散信息。
    *   **自然交互**: 支持用日常语言提问。
    *   **证据溯源**: 回答均链接到原始文档。
    *   **效率提升**: 将查询时间缩短至秒级。
    """)

st.sidebar.divider()

# --- Load RAG Pipeline (Cached) ---
@st.cache_resource  # Caches the pipeline object across reruns
def load_rag_pipeline():
    """Loads the RAG pipeline using caching to avoid reloading models."""
    try:
        # Display loading status
        with st.spinner("正在加载问答引擎 (模型和索引)..."):
            pipeline = RAGPipeline()
        return pipeline
    except FileNotFoundError as e:
        st.error(f"初始化 RAG pipeline 失败: {e}. 向量存储文件未找到。请先运行 scripts/create_indexes.py 生成索引。", icon="⚠️")
        return None
    except Exception as e:
        st.error(f"初始化 RAG pipeline 失败: {e}。请检查配置文件和向量存储。", icon="⚠️")
        return None

# --- Manual Management UI (as a pseudo-window) ---
if st.session_state.show_manual_manager:
    # Create a container for the entire manual management interface
    with st.container():
        # Header with back button
        col1, col2 = st.columns([5, 1])
        with col1:
            st.title("📚 航空维修手册管理")
        with col2:
            st.button("返回问答 ↩", on_click=toggle_manual_manager, type="secondary")
        
        st.markdown('<hr style="height:2px;border-width:0;background-color:#2E86C1">', unsafe_allow_html=True)
        
        # Introduction text
        st.markdown("""
        管理您的维修手册资源，上传新文档并配置搜索设置。系统会自动处理上传的PDF文件并建立索引，支持多种类型的手册。
        """)
        
        # Create a layout with sidebar for navigation
        main_content, settings_sidebar = st.columns([3, 1])
        
        # --- SIDEBAR FOR NAVIGATION AND SETTINGS ---
        with settings_sidebar:
            st.markdown("### 控制面板")
            
            # Quick stats about loaded manuals
            with st.container():
                st.markdown("##### 手册统计")
                col1, col2 = st.columns(2)
                col1.metric("已加载手册", "8")
                col2.metric("总文件数", "14")
                
                st.caption("上次索引时间: 2023-06-15 14:30")
            
            st.divider()
            
            # Manual type selection
            st.markdown("##### 手册类型选择")
            manual_options = [
                "AMM - 维护手册",
                "FIM/TSM - 故障手册",
                "SSM - 系统原理图",
                "IPC - 零件目录",
                "SRM - 结构修理",
                "CMM - 部件维修",
                "WDM - 线路图",
                "EM - 发动机"
            ]
            
            # 添加全选按钮
            select_all = st.checkbox("选择所有手册类型", value=True)
            
            # 根据全选按钮状态设置默认值
            if select_all:
                default_selections = manual_options.copy()
            else:
                default_selections = []
                
            selected_manuals = st.multiselect(
                "选择要管理的手册类型",
                options=manual_options,
                default=default_selections,
                help="可以同时选择多种手册类型进行管理"
            )
            
            # 如果没有选择任何手册，显示提示
            if not selected_manuals and not select_all:
                st.warning("请至少选择一种手册类型，或勾选\"选择所有手册类型\"")
                
            # 处理显示所有手册的逻辑
            show_all_manuals = select_all or len(selected_manuals) == 0
            
            st.divider()
            
            # Actions panel
            st.markdown("##### 操作")
            if st.button("📥 批量导入", use_container_width=True):
                st.info("批量导入功能将在未来版本中支持")
            
            if st.button("🔄 重建索引", use_container_width=True):
                with st.spinner("正在索引文档..."):
                    time.sleep(1)  # 模拟处理时间
                st.success("索引已更新完成！")
            
            if st.button("🗑️ 清理未使用文件", use_container_width=True):
                st.info("清理功能将在未来版本中支持")
            
            st.divider()
            
            # Search settings (simplified)
            with st.expander("🔍 搜索设置", expanded=False):
                st.checkbox("AMM - 维护手册", value=True, key="search_amm")
                st.checkbox("FIM/TSM - 故障手册", value=True, key="search_fim")
                st.checkbox("SSM - 系统图", value=False, key="search_ssm")
                st.checkbox("IPC - 零件目录", value=False, key="search_ipc")
                st.checkbox("SRM - 结构修理", value=False, key="search_srm")
                st.checkbox("CMM - 部件维修", value=False, key="search_cmm")
                st.checkbox("WDM - 线路图", value=False, key="search_wdm")
                st.checkbox("EM - 发动机", value=False, key="search_em")
                
                st.slider("检索文档数", 3, 10, 5)
                st.slider("相关性阈值", 0.1, 1.0, 0.7, 0.1)
                
                if st.button("保存设置", use_container_width=True):
                    st.success("设置已保存")
        
        # --- MAIN CONTENT AREA ---
        with main_content:
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📁 文件管理", "⚙️ 高级设置"])
            
            # === FILE MANAGEMENT TAB ===
            with tab1:
                # Filter controls
                filter_col1, filter_col2 = st.columns([3, 1])
                with filter_col1:
                    search_term = st.text_input("搜索文件", placeholder="输入文件名、机型或章节...")
                with filter_col2:
                    sort_option = st.selectbox("排序方式", ["上传日期 ↓", "上传日期 ↑", "文件名 A-Z", "文件名 Z-A", "文件大小"])
                
                # AMM SECTION
                if "AMM - 维护手册" in selected_manuals or show_all_manuals:
                    with st.expander("AMM - 飞机维护手册", expanded=True):
                        # Upload area
                        upload_col, desc_col = st.columns([1, 2])
                        with upload_col:
                            amm_upload = st.file_uploader("上传AMM文件", 
                                                         type=["pdf"], 
                                                         accept_multiple_files=True,
                                                         key="upload_amm")
                        with desc_col:
                            st.markdown("""
                            **飞机维护手册** 包含飞机系统的详细描述和维修程序。
                            * **SDS** - 系统描述部分：包含系统的原理和工作细节
                            * **PP** - 维修程序部分：包含维修和故障排除步骤
                            """)
                        
                        # File listings - using a cleaner card-based approach
                        st.markdown("##### 已加载AMM文件")
                        
                        # Sample data for demonstration
                        amm_files = [
                            {"name": "B737-AMM-SDS-Ch28.pdf", "type": "SDS", "aircraft": "B737", "chapter": "28-燃油", "size": "12.4 MB", "date": "2023-05-10"},
                            {"name": "A320-AMM-SDS-Ch29.pdf", "type": "SDS", "aircraft": "A320", "chapter": "29-液压", "size": "9.2 MB", "date": "2023-04-22"},
                            {"name": "A320-AMM-PP-Ch29-01.pdf", "type": "PP", "aircraft": "A320", "chapter": "29-液压系统检查", "size": "5.7 MB", "date": "2023-06-01"}
                        ]
                        
                        # Display files in a clean table
                        for i, file in enumerate(amm_files):
                            col1, col2, col3 = st.columns([5, 2, 1])
                            with col1:
                                st.markdown(f"""
                                **{file['name']}**  
                                {file['aircraft']} | {file['chapter']} | {file['type']}
                                """)
                            with col2:
                                st.markdown(f"{file['size']} | {file['date']}")
                            with col3:
                                st.button("🗑️", key=f"del_amm_{i}")
                            st.divider()
                
                # FIM/TSM SECTION
                if "FIM/TSM - 故障手册" in selected_manuals or show_all_manuals:
                    with st.expander("FIM/TSM - 故障隔离手册", expanded=True):
                        # Upload area
                        upload_col, desc_col = st.columns([1, 2])
                        with upload_col:
                            fim_upload = st.file_uploader("上传FIM/TSM文件", 
                                                         type=["pdf"], 
                                                         accept_multiple_files=True,
                                                         key="upload_fim")
                        with desc_col:
                            st.markdown("""
                            **故障隔离手册** 提供针对特定故障现象的排查和修复流程。
                            * 包含故障树分析和测试步骤
                            * 常用于确定故障原因和修复方法
                            """)
                        
                        # File listings
                        st.markdown("##### 已加载FIM/TSM文件")
                        
                        # Sample data for demonstration
                        fim_files = [
                            {"name": "B737-FIM-28-10.pdf", "aircraft": "B737", "chapter": "28-10 燃油漏油故障", "size": "8.1 MB", "date": "2023-03-15"},
                            {"name": "A320-TSM-Ch79.pdf", "aircraft": "A320", "chapter": "79 滑油系统故障", "size": "6.4 MB", "date": "2023-02-28"}
                        ]
                        
                        # Display files in a clean table
                        for i, file in enumerate(fim_files):
                            col1, col2, col3 = st.columns([5, 2, 1])
                            with col1:
                                st.markdown(f"""
                                **{file['name']}**  
                                {file['aircraft']} | {file['chapter']}
                                """)
                            with col2:
                                st.markdown(f"{file['size']} | {file['date']}")
                            with col3:
                                st.button("🗑️", key=f"del_fim_{i}")
                            st.divider()
                
                # OTHER MANUAL SECTIONS
                # Dynamically create sections for other manual types when selected
                other_manuals = {
                    "SSM - 系统原理图": {
                        "key": "ssm",
                        "desc": "**系统原理图手册** 包含飞机系统的原理图、接线图和系统图表。",
                        "files": []
                    },
                    "IPC - 零件目录": {
                        "key": "ipc",
                        "desc": "**图解零件目录** 提供飞机各部件的分解图和零件号。",
                        "files": []
                    },
                    "SRM - 结构修理": {
                        "key": "srm",
                        "desc": "**结构修理手册** 包含飞机结构的修理方法和程序。",
                        "files": []
                    },
                    "CMM - 部件维修": {
                        "key": "cmm",
                        "desc": "**部件维修手册** 提供各可更换部件的维修指南。",
                        "files": []
                    },
                    "WDM - 线路图": {
                        "key": "wdm",
                        "desc": "**线路图手册** 提供电气系统的详细接线图和线路布置。",
                        "files": []
                    },
                    "EM - 发动机": {
                        "key": "em",
                        "desc": "**发动机手册** 包含发动机的维护和修理指南。",
                        "files": []
                    }
                }
                
                for manual_name, manual_info in other_manuals.items():
                    if manual_name in selected_manuals or show_all_manuals:
                        with st.expander(manual_name, expanded=(manual_name in selected_manuals)):
                            # Upload area
                            upload_col, desc_col = st.columns([1, 2])
                            with upload_col:
                                st.file_uploader(f"上传{manual_name}文件", 
                                                type=["pdf"], 
                                                accept_multiple_files=True,
                                                key=f"upload_{manual_info['key']}")
                            with desc_col:
                                st.markdown(manual_info['desc'])
                            
                            # File listings
                            st.markdown(f"##### 已加载{manual_name}文件")
                            
                            if not manual_info['files']:
                                st.info(f"尚未上传{manual_name}文件。点击上方'上传{manual_name}文件'添加。")
            
            # === ADVANCED SETTINGS TAB ===
            with tab2:
                st.subheader("高级配置")
                
                # AI模型配置
                with st.expander("AI模型配置", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.selectbox("语言模型", ["deepseek", "GPT-4"], index=0)
                        st.slider("模型温度", 0.0, 1.0, 0.7, 0.1)
                    with col2:
                        st.selectbox("向量模型", ["默认嵌入模型", "自定义模型"], index=0)
                        st.slider("上下文长度", 1, 10, 4)
                
                # 索引配置
                with st.expander("索引设置", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.selectbox("索引类型", ["基于分块", "基于段落", "基于章节"], index=0)
                        st.number_input("分块大小", 100, 1000, 500, 50)
                    with col2:
                        st.selectbox("分块重叠度", ["无重叠", "低重叠", "中等重叠", "高重叠"], index=1)
                        st.checkbox("启用元数据提取", value=True)
                
                # 多语言设置
                with st.expander("多语言设置", expanded=True):
                    st.checkbox("启用跨语言搜索", value=True)
                    st.multiselect("支持的语言", 
                                  ["中文", "英文", "法语", "德语", "西班牙语", "日语", "俄语"], 
                                  default=["中文", "英文"])
                
                # 预处理设置
                with st.expander("文档预处理", expanded=False):
                    st.checkbox("文本格式清理", value=True)
                    st.checkbox("表格数据提取", value=True)
                    st.checkbox("图像OCR识别", value=False)
                    st.checkbox("目录结构提取", value=True)
                
                # 保存按钮
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("保存高级设置", use_container_width=True, type="primary"):
                        st.success("高级设置已保存")
                with col2:
                    if st.button("恢复默认值", use_container_width=True):
                        st.info("已恢复默认设置")
else:
    # --- Main Page Title --- (Only show when not in manual manager mode)
    st.title("✈️ 航维智询 - 航空维修智能问答系统")
    st.caption("基于本地维修手册，为您提供快速、准确的技术问答支持。")

    # 添加全局CSS样式
    st.markdown("""
    <style>
    /* 任务链接样式 */
    a[href*="task="] {
        font-weight: bold;
        color: #2196F3;
        text-decoration: none;
    }
    a[href*="task="]:hover {
        text-decoration: underline;
    }

    /* 自定义任务链接样式 */
    .task-link {
        display: inline-flex;
        align-items: center;
        background: #e3f2fd;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 0 2px;
        color: #0d47a1 !important;
        font-weight: 500;
        text-decoration: none !important;
        border: 1px solid #bbdefb;
        transition: all 0.2s ease;
    }
    
    .task-link:hover {
        background: #bbdefb;
        border-color: #64b5f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .task-icon {
        margin-right: 5px;
        font-size: 0.9em;
    }

    /* 提高内嵌HTML可见性 */
    .task-section {
        padding: 10px;
        background-color: #f7f7f7;
        border-left: 3px solid #2196F3;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- 处理任务详情请求 ---
    # 检查URL参数中是否有task请求
    query_params = st.query_params
    if "task" in query_params:
        task_id = query_params["task"]
        set_viewing_task(task_id)
        
    # 添加JavaScript处理链接点击
    st.markdown("""
    <script>
    // 监听消息事件
    window.addEventListener('message', function(event) {
        if (event.data.task) {
            // 设置URL参数并刷新页面
            const searchParams = new URLSearchParams(window.location.search);
            searchParams.set('task', event.data.task);
            window.location.search = searchParams.toString();
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # 如果查看任务，先加载RAG Pipeline
    rag_pipeline = None
    if st.session_state.viewing_task and PIPELINE_AVAILABLE:
        rag_pipeline = load_rag_pipeline()
    
    # 显示任务详情
    if st.session_state.viewing_task:
        task_id = st.session_state.viewing_task
        
        with st.container():
            col1, col2 = st.columns([5, 1])
            with col1:
                st.header(f"📋 任务 {task_id} 详情")
            with col2:
                if st.button("返回 ↩", key="back_from_task"):
                    clear_viewing_task()
                    # 清除URL参数
                    st.query_params.clear()
                    st.rerun()
            
            # 从数据中获取任务详情
            if rag_pipeline:
                try:
                    result_data = rag_pipeline.data_processor.get_result_by_fim_task(task_id)
                    if result_data:
                        # 处理结果数据
                        st.success(f"已找到任务 {task_id}")

                        try:
                            # 显示任务标题和详情
                            task_title = result_data.get('title', '无标题')
                            st.subheader(f"📋 {task_title}")
                            
                            # 预处理并显示任务内容
                            task_content = result_data.get('result', '无内容')
                            
                            # 记录原始内容信息
                            logger.debug(f"任务内容长度: {len(task_content)}")
                            logger.debug(f"任务内容前100字符: {task_content[:100]}")
                            logger.debug(f"任务内容是否包含HTML: {'<' in task_content and '>' in task_content}")
                            
                            # 预处理Markdown
                            formatted_content = preprocess_markdown(task_content)
                            
                            # 创建显眼的任务框
                            st.markdown(f"""
                            <div class="task-section">
                                <p><strong>任务ID:</strong> {task_id}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 显示主要内容
                            st.markdown(formatted_content, unsafe_allow_html=True)
                        except Exception as e:
                            logger.error(f"显示任务内容时出错: {str(e)}")
                            st.error(f"显示任务内容时出错: {str(e)}")
                            # 尝试以纯文本方式显示
                            st.text(task_content)
                        
                        # 添加相关记录
                        with st.expander("最近查看的任务", expanded=False):
                            for hist_task in reversed(st.session_state.task_history):
                                if hist_task != task_id:  # 不显示当前任务
                                    st.markdown(f"[任务 {hist_task}](/?task={hist_task})")
                    else:
                        st.error(f"未找到任务 {task_id} 的详细信息")
                except Exception as e:
                    st.error(f"获取任务详情时出错: {str(e)}")
            else:
                st.error("无法获取任务详情：RAG pipeline不可用")
                
            # 分隔线
            st.markdown("---")
            
            # 显示返回按钮
            if st.button("返回主页", key="back_to_main"):
                clear_viewing_task()
                st.query_params.clear()
                st.rerun()
                
            # 提前退出，不显示问答界面
            st.stop()

    # --- Main Application Logic ---
    if PIPELINE_AVAILABLE:
        rag_pipeline = load_rag_pipeline()  # Function call

        if rag_pipeline:
            st.success("问答引擎已加载！", icon="✅")
            st.info("您可以开始提问了，例如：'错误代码29313000，我应该如何解决？'", icon="💡")

            # Display the loaded model name in the sidebar now that pipeline is loaded
            model_display_name = "N/A"
            if hasattr(rag_pipeline, 'llm') and rag_pipeline.llm:
                model_display_name = getattr(rag_pipeline.llm, 'model_name', 'N/A')
            st.sidebar.caption(f"当前模型: {model_display_name}")

            # Initialize chat history in session state
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # 确保当前会话的消息同步到messages
            if "chat_sessions" in st.session_state and "current_session" in st.session_state:
                if st.session_state.current_session in st.session_state.chat_sessions:
                    st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_session].copy()

            # Display chat messages from history on app rerun
            logger.debug(f"开始显示历史消息: 消息数量={len(st.session_state.messages) if 'messages' in st.session_state else 0}")
            for message in st.session_state.messages:
                message_role = message["role"]
                logger.debug(f"显示历史消息: 角色={message_role}")
                
                with st.chat_message(message_role):
                    # 如果新消息是用户消息，直接显示，否则预处理Markdown
                    if message_role == "user":
                        logger.debug("处理用户历史消息: 直接显示")
                        st.markdown(message["content"], unsafe_allow_html=True)
                    else:
                        # 使用预处理函数来确保历史消息也能正确渲染
                        logger.debug("处理助手历史消息: 先预处理再显示")
                        content_before = message["content"]
                        
                        # 记录原始内容的Markdown分析
                        has_md_elements = False
                        if content_before:
                            has_headings = any(line.strip().startswith('#') for line in content_before.split('\n'))
                            has_code = '```' in content_before
                            has_lists = any(line.strip().startswith(('- ', '* ', '+ ', '1. ')) for line in content_before.split('\n'))
                            has_md_elements = has_headings or has_code or has_lists
                            logger.debug(f"历史消息原始Markdown分析: 标题={has_headings}, 代码块={has_code}, 列表={has_lists}")
                        
                        formatted_content = preprocess_markdown(content_before)
                        
                        # 检查原内容和格式化后的内容是否有显著差异
                        content_diff = len(formatted_content) - len(content_before)
                        logger.debug(f"历史消息格式化: 长度变化={content_diff} (原长度={len(content_before)}, 新长度={len(formatted_content)})")
                        
                        # 如果内容没有明显的Markdown元素，可能需要特殊处理
                        if not has_md_elements and len(content_before) > 100:
                            logger.warning(f"历史消息可能缺少Markdown格式: {content_before[:100]}...")
                        
                        # 尝试直接使用unsafe_allow_html参数来渲染
                        try:
                            st.markdown(formatted_content, unsafe_allow_html=True)
                            logger.debug("历史消息渲染成功(使用unsafe_allow_html=True)")
                        except Exception as e:
                            logger.error(f"历史消息渲染失败: {str(e)}")
                            # 回退方案
                            st.write(formatted_content)
                            logger.debug("使用st.write作为回退方案显示内容")
                        
                    # Display sources if they exist for past assistant messages
                    if message_role == "assistant" and "sources" in message and message["sources"]:
                        source_count = len(message["sources"])
                        logger.debug(f"处理历史消息来源: 来源数量={source_count}")
                        with st.expander("查看来源", expanded=False):
                            for i, source in enumerate(message["sources"]):
                                # Try to display relevant info from metadata more cleanly
                                file = source.get('full_path', 'N/A')
                                display_path = 'N/A'
                                if isinstance(file, str):
                                    try:
                                        # Attempt to make path relative to project root for display
                                        display_path = Path(file).relative_to(Path.cwd()).as_posix()
                                    except ValueError:
                                        display_path = file

                                aircraft = source.get('aircraft_type', 'N/A')
                                manual = source.get('manual_type', 'N/A')
                                lang = source.get('language', 'N/A')
                                section = source.get('specific_section', 'N/A') or "N/A"
                                chunk_idx = source.get('chunk_index', 'N/A')
                                # Use st.markdown for better Markdown rendering
                                st.markdown(f"""
                                **{i + 1}.** `{display_path}` (Chunk: {chunk_idx})
                                > *机型:* `{aircraft}` | *手册:* `{manual}` | *语言:* `{lang}`
                                > *章节:* `{section[:60]}{'...' if len(section) > 60 else ''}`
                                """, unsafe_allow_html=True)
                            # Moved divider outside the inner loop, inside the expander
                            st.divider()

            # React to user input using chat_input
            if prompt := st.chat_input("请输入您关于航空维修的问题..."):
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # 同步到当前会话
                if "chat_sessions" in st.session_state and "current_session" in st.session_state:
                    if st.session_state.current_session in st.session_state.chat_sessions:
                        st.session_state.chat_sessions[st.session_state.current_session] = st.session_state.messages.copy()
                        # 保存更新后的会话状态
                        save_chat_sessions(st.session_state.chat_sessions)
                        
                # Get assistant response
                with st.chat_message("assistant"):
                    # Create an empty container for streaming output
                    message_placeholder = st.empty()
                    full_response = ""
                    sources = []
                    
                    # Add a thinking indicator
                    with st.spinner("正在思考并检索相关手册..."):
                        start_query_time = time.time()
                        # Query the pipeline
                        if rag_pipeline:
                            result = rag_pipeline.query(prompt)
                            response_text = result.get('answer', "抱歉，处理您的问题时出现错误。")
                            sources = result.get('sources', [])
                            
                            # 记录从模型获取的原始响应
                            logger.debug(f"从RAG pipeline获取的原始响应长度: {len(response_text)}")
                            logger.debug(f"响应前100字符: {response_text[:100]}...")
                            logger.debug(f"响应包含Markdown格式检查: {'```' in response_text}")
                            logger.debug(f"响应原始内容:\n{response_text}")
                            
                            # 检查响应文本格式
                            has_md_format = False
                            if response_text:
                                has_headings = any(line.strip().startswith('#') for line in response_text.split('\n'))
                                has_code_blocks = response_text.count('```') >= 2
                                has_lists = any(line.strip().startswith(('- ', '* ', '+ ', '1. ')) for line in response_text.split('\n'))
                                has_md_format = has_headings or has_code_blocks or has_lists
                                
                            if not has_md_format:
                                logger.warning("响应可能缺少Markdown格式，尝试增强格式化")
                                # 可能需要基本的格式化增强
                                lines = response_text.split('\n')
                                enhanced_lines = []
                                
                                # 简单地尝试识别可能的标题和列表
                                for line in lines:
                                    if re.match(r'^(.+?):$', line) and len(line) < 50:  # 可能是标题
                                        enhanced_lines.append(f"## {line}")
                                    else:
                                        enhanced_lines.append(line)
                                
                                # 记录增强结果
                                enhanced_text = '\n'.join(enhanced_lines)
                                logger.debug(f"增强格式化后的文本: {enhanced_text[:100]}...")
                            
                            # 如果有源文档，格式化显示
                            if sources:
                                logger.debug(f"检索到 {len(sources)} 个来源文档")
                                sources = [{
                                    'full_path': source.get('fault_code', 'N/A'),
                                    'aircraft_type': 'B737',  # 当前仅支持 B737
                                    'manual_type': 'FIM',     # 当前仅支持 FIM
                                    'language': 'N/A',
                                    'specific_section': source.get('description', 'N/A'),
                                    'chunk_index': source.get('fim_task', 'N/A')
                                } for source in sources]
                        else:
                            response_text = "错误：问答管道不可用。"
                            sources = []
                        end_query_time = time.time()

                    # 预处理 Markdown 内容以确保格式一致性
                    formatted_response = preprocess_markdown(response_text)
                    
                    # 检查是否包含特定Markdown元素
                    has_headings = any(line.strip().startswith('#') for line in formatted_response.split('\n'))
                    has_code = '`' in formatted_response
                    has_lists = any(line.strip().startswith(('* ', '- ', '+ ', '1. ')) for line in formatted_response.split('\n'))
                    
                    logger.debug(f"Markdown元素检查: 标题={has_headings}, 代码={has_code}, 列表={has_lists}")
                    
                    # 流式输出处理
                    try:
                        # 检查是否支持write_stream (Streamlit 1.25.0+)
                        stream_supported = hasattr(message_placeholder, "write_stream")
                        logger.debug(f"支持write_stream流式输出: {stream_supported}")
                        
                        if stream_supported:
                            # 使用write_stream方法进行流式输出
                            def stream_data():
                                # 按有意义的段落分割（标题、列表等）
                                current_segment = ""
                                buffer = []
                                lines = formatted_response.split('\n')
                                logger.debug(f"流式输出: 总行数 = {len(lines)}")
                                
                                for i, line in enumerate(lines):
                                    # 标题、列表项、代码块起始和结束应该作为段落边界
                                    is_boundary = (line.strip().startswith('#') or 
                                                line.strip().startswith('*') or 
                                                line.strip().startswith('1.') or
                                                line.strip().startswith('```'))
                                    
                                    # 空行也可能是段落边界
                                    is_blank = not line.strip()
                                    
                                    # 记录分段信息
                                    if is_boundary:
                                        logger.debug(f"流式输出: 在行 {i} 检测到段落边界: {line}")
                                    
                                    # 如果遇到边界，输出之前收集的内容
                                    if (is_boundary or is_blank) and buffer:
                                        segment = '\n'.join(buffer)
                                        if segment.strip():
                                            logger.debug(f"流式输出: 生成段落(长度={len(segment)})")
                                            yield segment + "\n\n"
                                            time.sleep(0.08)  # 较短的延迟，保持流畅
                                        buffer = []
                                    
                                    buffer.append(line)
                                
                                # 输出剩余内容
                                if buffer:
                                    segment = '\n'.join(buffer)
                                    if segment.strip():
                                        logger.debug(f"流式输出: 生成最后段落(长度={len(segment)})")
                                        yield segment
                        
                            # 流式输出
                            logger.debug("开始流式输出...")
                            message_placeholder.write_stream(stream_data)
                            
                            # 最后确保完整内容正确显示
                            logger.debug("流式输出完成，使用markdown渲染最终内容")
                            message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
                        else:
                            # 回退方法 - 逐步构建更新
                            logger.debug("不支持流式输出，使用回退方法")
                            segments = []
                            current_segment = []
                            lines = formatted_response.split('\n')
                            
                            for i, line in enumerate(lines):
                                # 检测段落边界
                                is_boundary = (line.strip().startswith('#') or
                                            line.strip() == '' or
                                            line.strip().startswith('*') or
                                            line.strip().startswith('```'))
                                
                                if is_boundary and current_segment:
                                    segments.append('\n'.join(current_segment))
                                    current_segment = []
                                
                                current_segment.append(line)
                            
                            # 添加最后一段
                            if current_segment:
                                segments.append('\n'.join(current_segment))
                            
                            logger.debug(f"回退方法: 分割为 {len(segments)} 个段落")
                            
                            # 逐段显示
                            for i, segment in enumerate(segments):
                                if i == 0:
                                    full_response = segment
                                else:
                                    full_response += "\n" + segment
                                
                                # 使用markdown渲染更新，确保格式正确
                                logger.debug(f"回退方法: 显示段落 {i+1}/{len(segments)}")
                                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                                time.sleep(0.1)
                    except Exception as e:
                        # 如果流式输出失败，简单显示完整内容
                        error_msg = f"流式显示出错: {str(e)}，使用标准显示"
                        logger.error(error_msg)
                        st.error(error_msg)
                        message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
                    
                    # 最后确保内容被正确渲染（避免部分情况下格式丢失）
                    logger.debug("最终渲染: 使用完整格式化内容")
                    message_placeholder.markdown(formatted_response, unsafe_allow_html=True)

                    # 记录会话状态
                    session_id = id(st.session_state)
                    logger.debug(f"会话ID: {session_id}")
                    
                    # 检查会话状态中的消息数量
                    if "messages" in st.session_state:
                        logger.debug(f"会话中的消息数量: {len(st.session_state.messages)}")
                    else:
                        logger.debug("会话中没有消息历史")

                    # Display sources in an expander below the answer
                    if sources:
                        with st.expander("查看来源", expanded=False):
                            for i, source in enumerate(sources):
                                file = source.get('full_path', 'N/A')
                                display_path = 'N/A'
                                if isinstance(file, str):
                                    try:
                                        display_path = Path(file).relative_to(Path.cwd()).as_posix()
                                    except ValueError:
                                        display_path = file

                                aircraft = source.get('aircraft_type', 'N/A')
                                manual = source.get('manual_type', 'N/A')
                                lang = source.get('language', 'N/A')
                                section = source.get('specific_section', 'N/A') or "N/A"
                                chunk_idx = source.get('chunk_index', 'N/A')
                                st.markdown(f"""
                                **{i + 1}.** `{display_path}` (Chunk: {chunk_idx})
                                > *机型:* `{aircraft}` | *手册:* `{manual}` | *语言:* `{lang}`
                                > *章节:* `{section[:60]}{'...' if len(section) > 60 else ''}`
                                """, unsafe_allow_html=True)
                            # Moved divider outside the inner loop, inside the expander
                            st.divider()

                    # Display query time
                    st.caption(f"查询耗时: {end_query_time - start_query_time:.2f} 秒")

                    # Add assistant response (and sources) to chat history
                    logger.debug("更新会话历史: 添加助手回复")
                    message_content_length = len(formatted_response)
                    logger.debug(f"添加到历史的消息长度: {message_content_length}")
                    logger.debug(f"添加到历史的来源数量: {len(sources)}")

                    # 记录最后一次操作前的会话状态
                    if "messages" in st.session_state:
                        prev_msg_count = len(st.session_state.messages)
                        logger.debug(f"更新前会话消息数量: {prev_msg_count}")

                    st.session_state.messages.append({"role": "assistant", "content": formatted_response, "sources": sources})
                    logger.debug(f"会话更新完成，当前消息数: {len(st.session_state.messages)}")

                    # 同步到当前会话
                    if "chat_sessions" in st.session_state and "current_session" in st.session_state:
                        if st.session_state.current_session in st.session_state.chat_sessions:
                            st.session_state.chat_sessions[st.session_state.current_session] = st.session_state.messages.copy()
                            # 保存更新后的会话状态
                            save_chat_sessions(st.session_state.chat_sessions)

                    # 记录会话中最后一条消息的内容摘要（前100个字符）
                    last_message = st.session_state.messages[-1]["content"]
                    logger.debug(f"会话最后消息摘要: {last_message[:100]}{'...' if len(last_message) > 100 else ''}")

                    # 检查历史消息的格式一致性
                    has_inconsistency = False
                    for i, msg in enumerate(st.session_state.messages):
                        if msg["role"] == "assistant" and "content" in msg:
                            # 检查消息内容是否包含markdown标记
                            content = msg["content"]
                            has_markdown = ('`' in content or '#' in content or '*' in content)
                            if not has_markdown and len(content) > 50:  # 只检查较长的无Markdown消息
                                logger.warning(f"消息 #{i} 可能缺少Markdown格式 (长度={len(content)})")
                                has_inconsistency = True

                    if has_inconsistency:
                        logger.warning("检测到会话历史中存在格式不一致的消息")
                    else:
                        logger.debug("会话历史中的所有消息格式一致")
        else:
            # This message shows if load_rag_pipeline returned None
            st.warning("RAG pipeline 未能成功加载，问答功能不可用。请检查启动时终端的错误信息和配置。", icon="⚠️")