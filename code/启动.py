
import os
import subprocess
import os
import torch
import streamlit as st

# 修复 torch.classes.__path__ 的问题
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
# 或者直接将其设置为空列表
# torch.classes.__path__ = []
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 切换到当前目录
os.chdir(current_dir)

# 创建并激活 Conda 环境
subprocess.run(["conda", "create", "-n", "plane_rag", "python=3.12", "-y"])
subprocess.run(["conda", "activate", "plane_rag"])

# 安装依赖
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# 启动 Streamlit 应用
subprocess.run(["streamlit", "run", "app.py"])