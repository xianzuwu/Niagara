import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from datetime import datetime

class TensorboardLogger:
    def __init__(self, cfg, is_global_zero):
        self.writer = None
        self.logger = None
        
        if is_global_zero:
            self._setup_tensorboard(cfg)
            self._setup_logging(cfg)
            
            # 保存配置文件到 TensorBoard 的文本日志中
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            del cfg_dict["config"]
            
            # 过滤掉不能序列化的复杂对象
            filtered_config = {k: v for k, v in cfg_dict.items() if isinstance(v, (int, float, str, bool, list, dict))}
            self.logger.info(f"Config: {filtered_config}")

            if "SLURM_JOB_ID" in os.environ:
                SLURM_ID = os.environ['SLURM_JOB_ID']
                self.logger.info(f"SLURM job ID: {SLURM_ID}")

    def _setup_tensorboard(self, cfg):
        # 使用时间戳或其他关键参数生成实验命名
        experiment_name = f"{cfg.config.exp_name}_{cfg.run.random_seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log_dir = os.path.join("logs", experiment_name)
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs saved to {log_dir}")

    def _setup_logging(self, cfg):
        # 设置 Python logging 模块
        experiment_name = f"{cfg.config.exp_name}_{cfg.run.random_seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log_dir = os.path.join("logs", experiment_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'experiment.log')
        
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger()
        print(f"Logging to {log_file}")

    def log(self, values, step):
        if self.writer:
            for key, value in values.items():
                self.writer.add_scalar(key, value, step)

    def log_text(self, message):
        if self.logger:
            self.logger.info(message)

    def log_image(self, tag, image, step=0):
        if self.writer:
            # Image should be in the shape of (C, H, W), Tensor format
            if isinstance(image, torch.Tensor):
                self.writer.add_image(tag, image, step)

    def upload_file(self, key, filename):
        if self.logger:
            self.logger.info(f"File {filename} uploaded with key {key}")

def setup_logger(cfg, is_global_zero):
    return TensorboardLogger(cfg, is_global_zero)
