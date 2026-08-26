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

from vision3d.engine import SingleTester
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import add_tester_args, get_default_parser
from vision3d.utils.profiling import profile_cpu_runtime

# isort: split
from config import make_cfg
from dataset import test_data_loader
#from loss import EvalFunction
from model import create_model, NeuralGESM_PC
import torch

def add_custom_args():
    parser = get_default_parser()
    parser.add_argument(
        "--benchmark",
        choices=["4DMatch-F", "4DLoMatch-F"],
        required=True,
        help="test benchmark",
    )


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        # dataloader
        with profile_cpu_runtime("Data loader created"):
            data_loader = test_data_loader(cfg, self.args.benchmark)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        self.total_samples = len(data_loader)

        # evaluator
        self.eval_func = EvalFunction(cfg).cuda()

        self.loss_func = LossFunction(cfg).cuda()

        self.valid_metrics = {}      # 累加有效样本的指标
        self.valid_count = 0
        self.nan_count = 0

    def test_step(self, iteration, data_dict):
        if hasattr(self, 'total_samples') and self.total_samples > 0:
            print("[进度] 当前第 {} 个样本 / 共 {} 个样本".format(iteration, self.total_samples))

        data_dict["registration"] = True
        #output_dict = self.model(data_dict)
        #return output_dict

        #'''
        self.model.gesm_net = NeuralGESM_PC(self.cfg).cuda()
        torch.set_grad_enabled(True)
        self.model.gesm_net.train()   # 让 gesm_net 处于训练模式
        optimizer = torch.optim.Adam(self.model.gesm_net.siren.parameters(),lr=self.cfg.model.gesm.lr_test)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.2,           # 每次降低学习率的比例
                patience=1,          # 连续多少个 epoch loss 没改善才降 lr（建议比你写的 1 大一些）
                verbose=True,
                min_lr=1e-6)

        num_epochs = getattr(self.cfg.model.gesm, 'num_epochs_test', 300)

        src_corr = (data_dict["src_corr_points"]).float().cuda()
        tgt_corr = (data_dict["tgt_corr_points"]).float().cuda()

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # 前向
            output_dict = self.model(data_dict)

            #warped_src_corr, _, _ = self.model.gesm_net(src_corr)
            #output_dict["warped_src_corr"] = warped_src_corr
            #output_dict["tgt_corr"] = tgt_corr


            # 计算 loss（会自动包含 GESM loss，因为我们在 LossFunction 里加了）
            loss_dict = self.loss_func(data_dict, output_dict)
            #loss = loss_dict["loss"]
            if "registration_loss" in loss_dict:
                loss = loss_dict["registration_loss"]
            else:
                loss = loss_dict["loss"]

            gesm_loss = loss_dict["gesm_loss"]

            # 反向 + 更新
            loss.backward()
            optimizer.step()

            #scheduler.step(loss.item())

            # 可选：打印进度
            if (epoch + 1) % 100 == 0:
                scheduler.step(loss.item())
                print(f"[Test-time Opt] Epoch {epoch+1}/{num_epochs}, Loss: {gesm_loss.item():.6f}")

        # ==================== 优化完成后，最终前向得到结果 ====================

        self.model.eval()
        torch.set_grad_enabled(False)

        with torch.no_grad():
            output_dict = self.model(data_dict)

        return output_dict
        #'''


    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.eval_func(data_dict, output_dict)


        epe = result_dict.get("EPE")

        if epe is not None and (torch.isnan(torch.tensor(epe)) or torch.isinf(torch.tensor(epe))):
            self.nan_count += 1
            print(f"\n[跳过] 第 {iteration + 1} 个样本出现 NaN，已排除统计")

            return {}   # 返回空字典，不参与后续统计


        # 正常样本：累加
        self.valid_count += 1
        for key, value in result_dict.items():
            if isinstance(value, (int, float)):
                self.valid_metrics[key] = self.valid_metrics.get(key, 0.0) + value

        return result_dict

    def get_log_string(self, iteration, data_dict, output_dict, result_dict):
        shape_name = data_dict["shape_name"]
        src_frame = data_dict["src_frame"]
        tgt_frame = data_dict["tgt_frame"]
        message = f"{shape_name}, id0: {tgt_frame}, id1: {src_frame}"
        message += ", " + get_log_string(result_dict=result_dict)
        return message

    def after_test_epoch(self, summary_dict):
        print("\n" + "=" * 70)
        print("[最终统计结果]（已排除出现 NaN 的样本）")
        print(f"有效样本数 : {self.valid_count}")
        print(f"NaN样本数  : {self.nan_count}")
        print(f"总样本数   : {self.total_samples}")
        print("-" * 70)

        if self.valid_count > 0:
            print("平均指标（仅有效样本）:")
            for key, total_val in self.valid_metrics.items():
                avg = total_val / self.valid_count
                print(f"  {key:12s} : {avg:.6f}")
        else:
            print("没有有效样本！")

        print("=" * 70)


def main():
    add_tester_args()
    add_custom_args()
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run(strict_loading=False)


if __name__ == "__main__":
    main()
