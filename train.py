import os
import time
import logging
import torch
import hydra
import torch.optim as optim

from ema_pytorch import EMA
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import seed_everything
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy

from misc.tlogger import setup_logger
from evaluation.evaluator import Evaluator
from datasets.util import create_datasets
from trainer import Trainer


def run_epoch(fabric,
              trainer,
              ema,
              train_loader,
              val_loader,
              optimiser,
              lr_scheduler,
              evaluator):
    """Run a single epoch of training and validation"""
    cfg = trainer.cfg
    trainer.model.set_train()

    if fabric.is_global_zero:
        logging.info("Training on epoch {}".format(trainer.epoch))

    for batch_idx, inputs in enumerate(train_loader):
        inputs["target_frame_ids"] = cfg.model.gauss_novel_frames
        start_time = time.time()
        
        losses, outputs = trainer(inputs)
        duration = time.time() - start_time

        optimiser.zero_grad(set_to_none=True)
        fabric.backward(losses["loss/total"])
        optimiser.step()
        if ema is not None:
            ema.update()
        
        step = trainer.step

        early_phase = batch_idx % trainer.cfg.run.log_frequency == 0 and step < 6000
        if fabric.is_global_zero:
            learning_rate = lr_scheduler.get_last_lr()[0]  # Ensure correct learning rate retrieval
            trainer.log_scalars("train", outputs, losses, learning_rate)

            trainer.log_time(batch_idx, duration, losses["loss/total"])

            late_phase = step % 5000 == 0
            if early_phase or late_phase:
                trainer.log("train", inputs, outputs)
            if step % cfg.run.save_frequency == 0 and step != 0:
                trainer.model.save_model(optimiser, step, ema)
            early_phase = (step < 6000) and (step % 500 == 0)
            if (early_phase or step % cfg.run.val_frequency == 0):
                model_eval = ema if ema is not None else trainer.model
                trainer.validate(model_eval, evaluator, val_loader, device=fabric.device)

        if (early_phase or step % cfg.run.val_frequency == 0):
            torch.cuda.empty_cache()
            
        trainer.step += 1
        lr_scheduler.step()

        
@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(output_dir)
    logging.info(f"Working dir: {output_dir}")

    torch.set_float32_matmul_precision('high')
    seed_everything(cfg.run.random_seed)

    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.train.num_gpus,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=cfg.train.mixed_precision
    )
    fabric.launch()
    fabric.barrier()

    logging.info("Loaded datasets")

    train_dataset, train_loader = create_datasets(cfg, split="train")

    trainer = Trainer(cfg, len(train_dataset))
    model = trainer.model

    optimiser = optim.Adam(model.parameters_to_train, cfg.optimiser.learning_rate)
    def lr_lambda(*args):
        threshold = cfg.optimiser.scheduler_lambda_step_size
        if trainer.step < threshold:
            return 1.0
        else:
            return 0.1
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    if cfg.train.ema.use and fabric.is_global_zero:
        ema = EMA(  
            model, 
            beta=cfg.train.ema.beta,
            update_every=cfg.train.ema.update_every,
            update_after_step=cfg.train.ema.update_after_step
        )
        ema = fabric.to_device(ema)
    else:
        ema = None

    if (ckpt_dir := model.checkpoint_dir()).exists():
        model.load_model(ckpt_dir, optimiser=optimiser)
    elif cfg.train.load_weights_folder:
        model.load_model(cfg.train.load_weights_folder)

    trainer, optimiser = fabric.setup(trainer, optimiser)
    train_loader = fabric.setup_dataloaders(train_loader)

    if fabric.is_global_zero:
        if cfg.train.logging:
            trainer.set_logger(setup_logger(cfg, fabric.is_global_zero))
        val_dataset, val_loader = create_datasets(cfg, split="val")
        evaluator = Evaluator()
        evaluator = fabric.to_device(evaluator)
    else:
        val_loader = None
        evaluator = None

    trainer.epoch = 0
    trainer.start_time = time.time()
    for trainer.epoch in range(cfg.optimiser.num_epochs):
        run_epoch(fabric, trainer, ema, train_loader, val_loader, optimiser, lr_scheduler, evaluator)

if __name__ == "__main__":
    main()