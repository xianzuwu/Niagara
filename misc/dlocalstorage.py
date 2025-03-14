import os
import shutil
import logging
import tarfile
from pathlib import Path
import torch.distributed as dist  # 引入分布式库

project_name = "monosplat"

def get_local_dir():
    tmp = os.environ["TMPDIR"] if "TMPDIR" in os.environ else "/tmp"
    if "SLURM_JOB_ID" in os.environ:
        sub_dir = f"{project_name}/{os.environ['SLURM_JOB_ID']}"
    else:
        sub_dir = project_name
    tmp = os.path.join(tmp, sub_dir)
    return Path(tmp)

def local_storage_path(filename):
    return get_local_dir() / Path(filename).name

def copy_to_local_storage(filename, rank=None):
    storage = get_local_dir()
    os.makedirs(storage, exist_ok=True)
    new_filename = local_storage_path(filename)
    filename = Path(filename)

    # 只有主进程(rank 0)执行文件复制
    if rank is not None and rank != 0:
        # 非主进程直接返回目标路径，不执行复制
        return new_filename
    
    # 主进程执行文件复制
    if not new_filename.is_file() or \
        filename.stat().st_size != new_filename.stat().st_size:
        logging.info(f"Copying {str(filename)} to {str(new_filename)} ...")
        shutil.copyfile(filename, new_filename)
        logging.info(f"Finished copying.")

    # 所有进程在这里等待，确保文件复制完成
    if dist.is_initialized():
        dist.barrier()

    return new_filename

def extract_tar(fn, unzip_dir, rank=None):
    unzip_dir.mkdir(exist_ok=True, parents=True)

    # 只有主进程(rank 0)执行解压操作
    if rank is not None and rank == 0:
        logging.info(f"Unpacking {fn} to {unzip_dir} ...")
        with tarfile.open(fn) as tf:
            tf.extractall(unzip_dir, filter='fully_trusted')
        logging.info(f"Finished unpacking.")
        fn.unlink()
        logging.info(f"Deleted {str(fn)}.")
    
    # 所有进程在这里等待，确保解压完成
    if dist.is_initialized():
        dist.barrier()
