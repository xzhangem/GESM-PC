import sys
import os
from pathlib import Path
import importlib.util

# ==================== 路径设置 ====================
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "vision3d"))

# 强制清除之前错误的 loss 缓存
if 'loss' in sys.modules:
    del sys.modules['loss']
# ====================================================

# 显式加载本地 loss.py（绕过所有冲突）
loss_path = project_root / "loss.py"
spec = importlib.util.spec_from_file_location("loss", str(loss_path))
loss_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loss_module)

# 绑定到全局
EvalFunction = loss_module.EvalFunction
LossFunction = loss_module.LossFunction

print("✅ Successfully loaded local loss.py")
