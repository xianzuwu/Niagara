import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import threading
from queue import Queue

def worker(queue, model, pbar):
    while True:
        item = queue.get()
        if item is None:
            break
        image_path, normal_save_path = item
        try:
            # 加载图像（保持原始尺寸）
            image = Image.open(image_path).convert('RGB')

            # 使用 StableNormal 模型预测法线图
            normal_map = model(image)

            # 保存法线图
            normal_save_path.parent.mkdir(parents=True, exist_ok=True)
            normal_map.save(normal_save_path)
        except Exception as e:
            print(f"处理 {image_path} 时出错：{e}")
        finally:
            pbar.update(1)
            queue.task_done()

def main():
    # 设置您的数据集路径
    dataset_root = Path('/home/aizhenxin/Projects/aizhenxin/Re10K')  # 替换为您的数据集路径
    split_names = ['train', 'test']  # 根据您的数据集划分调整

    # 在主程序中加载模型
    normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)

    # 准备任务队列
    task_queue = Queue()

    # 统计总任务数
    total_tasks = 0

    for split in split_names:
        split_path = dataset_root / split
        normal_save_dir = dataset_root / 'normals1' / split

        # 获取所有未处理的图像文件的路径列表
        for root, dirs, files in os.walk(split_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_path = Path(root) / file
                    # 构建对应的法线图保存路径
                    relative_path = image_path.relative_to(split_path)
                    normal_save_path = normal_save_dir / relative_path
                    if not normal_save_path.exists():
                        task_queue.put((image_path, normal_save_path))
                        total_tasks += 1

    # 创建进度条
    pbar = tqdm(total=total_tasks, desc="Processing images")

    # 创建线程池
    num_threads = 64  # 根据您的CPU核心数和需求调整
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(task_queue, normal_predictor, pbar))
        t.start()
        threads.append(t)

    # 等待所有任务完成
    task_queue.join()

    # 停止所有线程
    for _ in threads:
        task_queue.put(None)
    for t in threads:
        t.join()

    # 关闭进度条
    pbar.close()

if __name__ == '__main__':
    main()
