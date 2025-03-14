import os
import sys
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def main(gpu_id, data_partition, total_partitions):
    # Set the GPU used by the current process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')  


    normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)


    dataset_root = Path('/home/aizhenxin/Projects/aizhenxin/Re10K')  
    split_names = ['train', 'test']  

    image_paths = []
    normal_save_paths = []

    for split in split_names:
        split_path = dataset_root / split
        normal_save_dir = dataset_root / 'normals1' / split


        for root, dirs, files in os.walk(split_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_path = Path(root) / file
                    relative_path = image_path.relative_to(split_path)
                    normal_save_path = normal_save_dir / relative_path
                    if not normal_save_path.exists():
                        image_paths.append(image_path)
                        normal_save_paths.append(normal_save_path)

    total_images = len(image_paths)
    images_per_partition = total_images // total_partitions
    start_idx = data_partition * images_per_partition
    if data_partition == total_partitions - 1:
        end_idx = total_images
    else:
        end_idx = start_idx + images_per_partition

    image_paths_partition = image_paths[start_idx:end_idx]
    normal_save_paths_partition = normal_save_paths[start_idx:end_idx]

    pbar = tqdm(total=len(image_paths_partition), desc=f"GPU {gpu_id} processing images")

    for image_path, normal_save_path in zip(image_paths_partition, normal_save_paths_partition):
        try:
            image = Image.open(image_path).convert('RGB')

            with torch.no_grad():
                normal_map = normal_predictor(image)

            normal_save_path.parent.mkdir(parents=True, exist_ok=True)
            normal_map.save(normal_save_path)
        except Exception as e:
            print(f"process {image_path} error：{e}")
        finally:
            pbar.update(1)

    pbar.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_images.py <gpu_id> <data_partition>")
        sys.exit(1)

    gpu_id = int(sys.argv[1])  
    data_partition = int(sys.argv[2])  
    total_partitions = 8  # total gpu account

    main(gpu_id, data_partition, total_partitions)