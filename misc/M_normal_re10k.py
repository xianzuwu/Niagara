import os
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import threading
from queue import Queue
import cv2
import numpy as np

def worker(queue, model, pbar, pad_info):
    while True:
        item = queue.get()
        if item is None:
            break
        image_path, normal_save_path = item
        try:
            # 加载图像（保持原始尺寸）
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)

            # 预处理输入
            image = cv2.resize(image, (1064, 616))  # 模型需要的输入尺寸
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            image_tensor = image_tensor.cuda()  # 假设你使用GPU

            # 使用 Metric3Dv2 模型进行推理
            with torch.no_grad():
                output_dict = model(image_tensor)

            # 提取法线图
            if 'prediction_normal' in output_dict:
                pred_normal = output_dict['prediction_normal'][:, :3, :, :]
                pred_normal = pred_normal.squeeze()

                # 根据 padding 信息调整法线图的尺寸
                pred_normal = pred_normal[:, pad_info[0]: pred_normal.shape[1] - pad_info[1], pad_info[2]: pred_normal.shape[2] - pad_info[3]]

                # 将法线图转换为可视化的格式
                pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
                pred_normal_vis = (pred_normal_vis + 1) / 2  # 将法线图从 [-1, 1] 映射到 [0, 1]

                # 保存法线图
                normal_save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(normal_save_path), (pred_normal_vis * 255).astype(np.uint8))

        except Exception as e:
            print(f"处理 {image_path} 时出错：{e}")
        finally:
            pbar.update(1)
            queue.task_done()

def main():
    dataset_root = Path('/home/aizhenxin/Projects/aizhenxin/Re10K')
    split_names = ['train', 'test']

    # 本地加载模型（确保模型已经下载或存储在本地）
    normal_predictor = torch.hub.load('local_metric3d_repo', 'metric3d_vit_giant2', pretrain=True)

    pad_info = [32, 32, 32, 32]  # 假设的padding信息，调整为实际需要

    task_queue = Queue()
    total_tasks = 0

    for split in split_names:
        split_path = dataset_root / split
        normal_save_dir = dataset_root / 'M_normals' / split

        for root, dirs, files in os.walk(split_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_path = Path(root) / file
                    relative_path = image_path.relative_to(split_path)
                    normal_save_path = normal_save_dir / relative_path
                    if not normal_save_path.exists():
                        task_queue.put((image_path, normal_save_path))
                        total_tasks += 1

    pbar = tqdm(total=total_tasks, desc="Processing images")
    num_threads = 64
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(task_queue, normal_predictor, pbar, pad_info))
        t.start()
        threads.append(t)

    task_queue.join()

    for _ in threads:
        task_queue.put(None)
    for t in threads:
        t.join()

    pbar.close()

if __name__ == '__main__':
    main()