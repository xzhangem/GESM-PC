import sys
import os
from pathlib import Path

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

from vision3d.engine import EpochBasedTrainer
from vision3d.utils.optimizer import build_optimizer, build_scheduler
from vision3d.utils.parser import add_trainer_args
from vision3d.utils.profiling import profile_cpu_runtime

# isort: split
from config import make_cfg
from dataset import train_valid_data_loader
#from loss import EvalFunction, LossFunction
from model import create_model


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        with profile_cpu_runtime("Data loader created"):
            train_loader, val_loader = train_valid_data_loader(cfg)
        self.register_loader(train_loader, val_loader)

        # model
        model = create_model(cfg)
        model = self.register_model(model)

        # optimizer
        optimizer = build_optimizer(model, cfg)
        self.register_optimizer(optimizer)
        scheduler = build_scheduler(optimizer, cfg)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = LossFunction(cfg)
        self.eval_func = EvalFunction(cfg)

        # select best model
        self.save_best_model_on_metric("loss", largest=False)
        self.save_best_model_on_metric("precision", largest=True)
        self.save_best_model_on_metric("recall", largest=True)

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        result_dict = self.eval_func(data_dict, output_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        result_dict = self.eval_func(data_dict, output_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    add_trainer_args()
    cfg = make_cfg()

    print("=== Full Config Dump ===")
    print(cfg)
    print("\n=== Data Config ===")
    import pprint
    pprint.pprint(dict(cfg.data))
    print("data keys:", list(cfg.data.keys()) if hasattr(cfg.data, 'keys') else "No keys")
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
