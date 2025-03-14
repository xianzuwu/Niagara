import os
import json
import hydra
import torch
import numpy as np

from tqdm import tqdm
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import torchvision.transforms as transforms

from models.model import GaussianPredictor, to_device
from datasets.util import create_datasets
from misc.util import add_source_frame_id

import lpips


class CustomLPIPS(lpips.LPIPS):
    def __init__(self, net='vgg'):
        super(CustomLPIPS, self).__init__(net=net)
        
        # Define normalization specific to VGG
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.layers = list(self.net.children())

    def normalize_tensor(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def extract_feats(self, x):
        feats = []
        x = self.normalize_tensor(x)  # Normalize image
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats

    def forward_return_layers(self, input1, input2):
        feats0 = self.extract_feats(input1)
        feats1 = self.extract_feats(input2)
        
        diffs = [(f1 - f2) ** 2 for f1, f2 in zip(feats0, feats1)]
        return diffs


class Evaluator:
    def __init__(self, crop_border=0):
        self.crop_border = crop_border
        self.lpips = CustomLPIPS(net='vgg')

    def to(self, device):
        self.lpips.to(device)

    def __call__(self, pred, gt):
        if self.crop_border > 0:
            pred = pred[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            gt = gt[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        results = {}
        lpips_val = self.lpips(pred, gt)
        results['lpips'] = lpips_val.mean().item()

        lpips_per_layer = self.lpips.forward_return_layers(pred, gt)
        results['lpips_layer'] = {}
        results['lpips_layer']['lpips_total'] = lpips_val.mean().item()
        results['lpips_layer']['lpips_per_layer'] = {f'layer_{i+1}': l.mean().item() for i, l in enumerate(lpips_per_layer)}
        return results


def get_model_instance(model):
    return model.ema_model if type(model).__name__ == "EMA" else model


def evaluate(model, cfg, evaluator, dataloader, device=None, save_vis=False):
    model_model = get_model_instance(model)
    model_model.set_eval()

    score_dict = {}
    match cfg.dataset.name:
        case "re10k" | "nyuv2":
            target_frame_ids = [1, 2, 3]
            eval_frames = ["src", "tgt5", "tgt10", "tgt_rand"]
            for fid, target_name in zip(add_source_frame_id(target_frame_ids), eval_frames):
                score_dict[fid] = {
                    "ssim": [],
                    "psnr": [],
                    "lpips": [],
                    "lpips_layer": {
                        "lpips_total": [],
                        "lpips_per_layer": {}
                    },
                    "name": target_name
                }
        case "kitti":
            if cfg.dataset.stereo:
                eval_frames = ["s0"]
                target_frame_ids = ["s0"]
                all_frames = add_source_frame_id(eval_frames)
            else:
                eval_frames = [1, 2]
                target_frame_ids = eval_frames
                all_frames = add_source_frame_id(target_frame_ids)
            for fid in all_frames:
                score_dict[fid] = {
                    "ssim": [],
                    "psnr": [],
                    "lpips": [],
                    "lpips_layer": {
                        "lpips_total": [],
                        "lpips_per_layer": {}
                    },
                    "name": fid
                }

    dataloader_iter = iter(dataloader)
    for k in tqdm([i for i in range(len(dataloader.dataset) // cfg.data_loader.batch_size)]):
        try:
            inputs = next(dataloader_iter)
        except Exception as e:
            if cfg.dataset.name == "re10k":
                if cfg.dataset.test_split in ["pixelsplat_ctx1", "pixelsplat_ctx2", "latentsplat_ctx1", "latentsplat_ctx2"]:
                    print(f"Failed to read example {k}")
                    continue
            raise e

        with torch.no_grad():
            if device is not None:
                to_device(inputs, device)
            inputs["target_frame_ids"] = target_frame_ids
            outputs = model(inputs)

        for f_id in score_dict.keys():
            pred = outputs[('color_gauss', f_id, 0)]
            gt = inputs[('color', f_id, 0)]

            # Add debug prints to check shapes and values
            print(f"Prediction shape: {pred.shape}, Ground truth shape: {gt.shape}")
            print(f"Prediction min: {pred.min()}, max: {pred.max()}")
            print(f"Ground truth min: {gt.min()}, max: {gt.max()}")

            out = evaluator(pred, gt)

            for metric_name, v in out.items():
                if metric_name == "lpips_layer":
                    score_dict[f_id]["lpips_layer"]["lpips_total"].append(v["lpips_total"])
                    for layer_name, layer_loss in v["lpips_per_layer"].items():
                        if layer_name not in score_dict[f_id]["lpips_layer"]["lpips_per_layer"]:
                            score_dict[f_id]["lpips_layer"]["lpips_per_layer"][layer_name] = []
                        score_dict[f_id]["lpips_layer"]["lpips_per_layer"][layer_name].append(layer_loss)
                else:
                    score_dict[f_id][metric_name].append(v)

            # Print LPIPS layer results for the current frame
            print(f"Current LPIPS per layer for frame {score_dict[f_id]['name']} (batch {k}):")
            for layer_name, layer_losses in score_dict[f_id]["lpips_layer"]["lpips_per_layer"].items():
                print(f"Layer {layer_name}: {layer_losses[-1]}")

    metric_names = ["psnr", "ssim", "lpips"]
    score_dict_by_name = {}
    for f_id in score_dict.keys():
        score_dict_by_name[score_dict[f_id]["name"]] = {}
        for metric_name in metric_names:
            if score_dict[f_id][metric_name]:  # Check if the list is not empty
                score_dict[f_id][metric_name] = sum(score_dict[f_id][metric_name]) / len(score_dict[f_id][metric_name])
                score_dict_by_name[score_dict[f_id]["name"]][metric_name] = score_dict[f_id][metric_name]
            else:
                # Handle cases with no data (optional: print warning or set default value)
                print(f"Warning: No valid data for {metric_name} in frame {f_id}")

        score_dict_by_name[score_dict[f_id]["name"]]["lpips_layer"] = {}
        layers = score_dict[f_id]["lpips_layer"]["lpips_per_layer"].keys()
        for layer_name in layers:
            layer_losses = score_dict[f_id]["lpips_layer"]["lpips_per_layer"][layer_name]
            if layer_losses:
                score_dict_by_name[score_dict[f_id]["name"]]["lpips_layer"][layer_name] = sum(layer_losses) / len(layer_losses)

        lpips_total_losses = score_dict[f_id]["lpips_layer"]["lpips_total"]
        if lpips_total_losses:
            score_dict_by_name[score_dict[f_id]["name"]]["lpips_layer"]["lpips_total"] = sum(lpips_total_losses) / len(lpips_total_losses)

    for metric in metric_names:
        vals = [score_dict_by_name[f_id][metric] for f_id in eval_frames if f_id in score_dict_by_name and metric in score_dict_by_name[f_id]]
        if vals:
            print(f"{metric}: {np.mean(np.array(vals))}")
        else:
            print(f"Warning: No data available for metric {metric}")

    return score_dict_by_name

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    print("current directory:", os.getcwd())
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(output_dir)
    print("Working dir:", output_dir)

    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 1
    model = GaussianPredictor(cfg)
    device = torch.device("cuda:0")
    model.to(device)
    if (ckpt_dir := model.checkpoint_dir()).exists():
        model.load_model(ckpt_dir, ckpt_ids=0)
    
    evaluator = Evaluator(crop_border=cfg.dataset.crop_border)
    evaluator.to(device)

    split = "test"
    save_vis = cfg.eval.save_vis
    dataset, dataloader = create_datasets(cfg, split=split)
    score_dict_by_name = evaluate(model, cfg, evaluator, dataloader, 
                                  device=device, save_vis=save_vis)
    print(json.dumps(score_dict_by_name, indent=4))
    if cfg.dataset.name == "re10k":
        with open("metrics_{}_{}_{}.json".format(cfg.dataset.name, split, cfg.dataset.test_split), "w") as f:
            json.dump(score_dict_by_name, f, indent=4)
    with open("metrics_{}_{}.json".format(cfg.dataset.name, split), "w") as f:
        json.dump(score_dict_by_name, f, indent=4)
    
    lpips_layers_file = "lpips_layers_{}_{}.json".format(cfg.dataset.name, split)
    with open(lpips_layers_file, "w") as f:
        json.dump({name: scores["lpips_layer"] for name, scores in score_dict_by_name.items()}, f, indent=4)
    
if __name__ == "__main__":
    main()