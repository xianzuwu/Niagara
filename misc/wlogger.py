import os
import logging
import wandb
from omegaconf import OmegaConf

class WandbLogger:
    def __init__(self, cfg, is_global_zero):
        self.run = None
        if is_global_zero:  # 只在 rank 0 进程上初始化 wandb
            self.run = self._setup(cfg)
            # 直接将配置文件保存到 wandb
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            del cfg_dict["config"]
            
            # 过滤掉不能序列化的复杂对象
            filtered_config = {k: v for k, v in cfg_dict.items() if isinstance(v, (int, float, str, bool, list, dict))}
            self.run.config.update(filtered_config)  # 自动记录配置文件

            if "SLURM_JOB_ID" in os.environ:
                SLURM_ID = os.environ['SLURM_JOB_ID']
                self.run.config.update({"SLURM_JOB_ID": SLURM_ID})
                logging.info(f"SLURM job ID: {SLURM_ID}")

    @staticmethod
    def _setup(cfg):
        # 自动生成命名规则，按时间戳或其他关键参数区分实验
        experiment_name = f"{cfg.config.exp_name}_{cfg.run.random_seed}"
        
        # 过滤配置字典，只传递可序列化的数据类型
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        filtered_config = {k: v for k, v in config_dict.items() if isinstance(v, (int, float, str, bool, list, dict))}

        run = wandb.init(
            project="SVSplat",
            name=experiment_name,
            config=filtered_config,  # 使用过滤后的配置
            mode="online" if not cfg.run.debug else "disabled"
        )
        return run


    def log(self, values, step):
        if self.run:  # 确保只有在 rank 0 时才调用 wandb.log
            wandb.log(values, step=step)

    def upload_file(self, key, filename):
        if self.run:  # 确保只有在 rank 0 时才上传文件
            wandb.save(filename)

    def log_image(self, key, image, step=0):
        if self.run:  # 确保只有在 rank 0 时才记录图像
            wandb.log({key: [wandb.Image(image)]}, step=step)


def setup_logger(cfg, is_global_zero):  # 增加 is_global_zero 参数用于 rank 控制
    return WandbLogger(cfg, is_global_zero)
