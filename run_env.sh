#!/bin/bash

# Transport环境启动脚本

echo "=========================================="
echo "Transport任务环境启动"
echo "=========================================="

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source /root/RL_Assignment/venv/bin/activate

# 检查Python版本
echo ""
echo "Python版本:"
python --version

# 检查已安装的包
echo ""
echo "已安装的关键包:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
python -c "import gym; print(f'  Gym: {gym.__version__}')"

# 检查VMAS
echo ""
echo "检查VMAS..."
python -c "import vmas; print(f'  VMAS: {vmas.__version__}')"

echo ""
echo "=========================================="
echo "环境已就绪！"
echo "=========================================="
echo ""
echo "运行测试："
echo "  python /root/RL_Assignment/test_transport.py"
echo ""
echo "激活虚拟环境："
echo "  source /root/RL_Assignment/venv/bin/activate"
echo ""
echo "退出虚拟环境："
echo "  deactivate"
echo ""