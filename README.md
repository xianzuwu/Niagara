<!-- [![arXiv](https://img.shields.io/badge/arXiv-2406.04343-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/)
[![ProjectPage](https://img.shields.io/badge/Project_Page-Niagara-blue?logoColor=blue)](https://ai-kunkun.github.io/Niagara_page/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow)](https://huggingface.co/Xianzu/Niagara)  -->
<p align="center">

  <h2 align="center">🐃Niagara<br><span style="font-size: 0.58em;">
   Normal-Integrated Geometric Affine Field for Scene Reconstruction from a Single View</h2>
  <p align="center">
    <a href="https://xianzuwu.github.io/"><strong>Xianzu Wu</strong></a>
    ·
    <a href="https://ai-kunkun.github.io/"><strong>Zhenxin Ai</strong></a>
    ·
    <a href="https://leehomyc.github.io/"><strong>Harry Yang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=HX0BfLYAAAAJ&hl=en"><strong>Sernam Lim</strong></a>
    ·
    <a href="https://wp.lancs.ac.uk/vl/"><strong>Jun Liu</strong></a>
    ·
    <a href="https://huanwang.tech/"><strong>Huan Wang</strong></a>
    <br>
  </p>


<p align="center">
  <img src="assets/teaser_video.gif" alt="animated" />
</p>

<p align="center">
  </br>
    <a href="https://arxiv.org/">
      <img src='https://img.shields.io/badge/Paper-Arxiv-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://ai-kunkun.github.io/Niagara_page/'>
      <img src='https://img.shields.io/badge/Project-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href="https://github.com/xianzuwu/Niagara">
      <img src='https://img.shields.io/badge/Code-Github-blue?style=for-the-badge&logo=github&logoColor=white&labelColor=181717' alt='Code Github'></a> 
      <br>
      <a href="https://huggingface.co/Xianzu/Niagara">
      <img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow' alt='Huggingface'></a>
  </p>
  
</p>

## 🦉 ToDo List
- [x] 📢`15.03.2025`: release code and paper
- [ ] Release Complete Checkpoint.

## 🎉 Key Result

<table style="font-size: 9px; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="padding: 4px;">Method</th>
      <th style="padding: 4px;">PSNR (5f)</th>
      <th style="padding: 4px;">SSIM (5f)</th>
      <th style="padding: 4px;">LPIPS (5f)</th>
      <th style="padding: 4px;">PSNR (10f)</th>
      <th style="padding: 4px;">SSIM (10f)</th>
      <th style="padding: 4px;">LPIPS (10f)</th>
      <th style="padding: 4px;">PSNR (u[-30,30]f)</th>
      <th style="padding: 4px;">SSIM (u[-30,30]f)</th>
      <th style="padding: 4px;">LPIPS (u[-30,30]f)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/facebookresearch/synsin"><strong>Syn-Sin</strong></a></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>22.30</td>
      <td>0.740</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://single-view-mpi.github.io/"><strong>SV-MPI</strong></a> </td>
      <td>27.10</td>
      <td>0.870</td>
      <td>-</td>
      <td>24.40</td>
      <td>0.812</td>
      <td>-</td>
      <td>23.52</td>
      <td>0.785</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://github.com/Brummi/BehindTheScenes"><strong>BTS</strong></a> </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>24.00</td>
      <td>0.755</td>
      <td>0.194</td>
    </tr>
    <tr>
      <td><a href="https://github.com/szymanowiczs/splatter-image"><strong>Splatter Image</strong></a></td>
      <td>28.15</td>
      <td>0.894</td>
      <td>0.110</td>
      <td>25.34</td>
      <td>0.842</td>
      <td>0.144</td>
      <td>24.15</td>
      <td>0.810</td>
      <td>0.177</td>
    </tr>
    <tr>
      <td><a href="https://github.com/vincentfung13/MINE"><strong>MINE</strong></a></td>
      <td>28.45</td>
      <td>0.897</td>
      <td>0.111</td>
      <td>25.89</td>
      <td>0.850</td>
      <td>0.150</td>
      <td>24.75</td>
      <td>0.820</td>
      <td>0.179</td>
    </tr>
    <tr>
      <td><a href="https://github.com/eldar/flash3d"><em>Flash3D</em></a></td>
      <td><em>28.46</em></td>
      <td><em>0.899</em></td>
      <td><em>0.100</em></td>
      <td><em>25.94</em></td>
      <td><em>0.857</em></td>
      <td><em>0.133</em></td>
      <td><em>24.93</em></td>
      <td><em>0.833</em></td>
      <td><em>0.160</em></td>
    </tr>
    <tr>
      <td><strong>Ours</strong></td>
      <td><strong>29.00</strong></td>
      <td><strong>0.904</strong></td>
      <td><strong>0.099</strong></td>
      <td><strong>26.30</strong></td>
      <td><strong>0.862</strong></td>
      <td><strong>0.131</strong></td>
      <td><strong>25.28</strong></td>
      <td><strong>0.836</strong></td>
      <td><strong>0.156</strong></td>
    </tr>
  </tbody>
</table>
<strong> Novel view synthesis comparison on the RealEstate10K dataset. </strong>Following <a href="https://github.com/eldar/flash3d"><strong>Flash3D</strong></a>, we evaluate our method on the in-domain novel view synthesis task. As seen, our model consistently outperforms existing methods across different frame counts (f as frames,5 frames, 10 frames, u[-30,30] frames), in terms of PSNR, SSIM, and LPIPS. (The <strong>best</strong> results are in bold, and the <em>second best</em> is slanting typeface. )<br>
<p align="center" style="margin: 1rem 0;">
  <!-- 第一个按钮：Project Page 颜色 -->
  <a href="https://ai-kunkun.github.io/Niagara_page/"
     style="
       display: inline-block;
       background-color: #CF6F1C;
       color: #fff;
       padding: 0.5rem 1rem;
       text-decoration: none;
       border-radius: 4px;
       margin-right: 1rem;
     ">
    Check out more visual results
  </a>
</p>
<p align="center" style="margin: 1rem 0;">
  <!-- 第二个按钮：Arxiv 颜色 -->
  <a href="https://ai-kunkun.github.io/Niagara_page/"
     style="
       display: inline-block;
       background-color: #A0D468;
       color: #fff;
       padding: 0.5rem 1rem;
       text-decoration: none;
       border-radius: 4px;
     ">
    Check out more numerical results
  </a>
</p>

# 🚀Setup

## 🛠️Create a python environment

Niagara has been trained and tested with the followings software versions:

- Python 3.10
- Pytorch 2.2.2
- CUDA 11.8
- GCC 11.2 (or more recent)

Begin by installing CUDA 11.8 and adding the path containing the `nvcc` compiler to the `PATH` environmental variable.
Then the python environment can be created either via conda:

```sh
conda create -y python=3.10 -n niagara
conda activate niagara
```

or using Python's venv module (assuming you already have access to Python 3.10 on your system):

```sh
python3.10 -m venv .venv
. .venv/bin/activate
```

Finally, install the required packages as follows:

```sh
pip install -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## 🛠️Add 3DGS python environment
```sh
git clone diff-gaussian-rasterization @ git+https://github.com/eldar/diff-gaussian-rasterization-w-pose@main
git submodule add diff-gaussian-rasterization @ git+https://github.com/eldar/diff-gaussian-rasterization-w-pose@main third_party/diff-gaussian-rasterization-w-pose
git submodule update --init --recursive
```

## 📝 Download training data

### 🧩RealEstate10K dataset

For downloading the RealEstate10K dataset we base our instructions on the [Behind The Scenes](https://github.com/Brummi/BehindTheScenes/tree/main?tab=readme-ov-file#-datasets) scripts.
First you need to download the video sequence metadata including camera poses from https://google.github.io/realestate10k/download.html and unpack it into `data/` such that the folder layout is as follows:

```
data/RealEstate10K/train
data/RealEstate10K/test
```

Finally download the training and test sets of the dataset with the following commands:

```sh
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m train
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m test
```

This step will take several days to complete. Finally, download additional data for the RealEstate10K dataset.
In particular, we provide pre-processed COLMAP cache containing sparse point clouds which are used to estimate the scaling factor for depth predictions.
The last two commands filter the training and testing set from any missing video sequences.

```sh
sh datasets/dowload_realestate10k_colmap.sh
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s train
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s test
```

### 🧩KITTI dataset
For downloading the KITTI dataset, we base our instructions on the [versatran01](https://gist.github.com/versatran01/19bbb78c42e0cafb1807625bbb99bd85) scripts.
```sh
cd kitti_raw
wget -nc -i kitti_archives.txt
```
This step will take in some time to complete. Finally, the KITTI download data you need to extract.
```sh
unzip "*drive*.zip" "*/*/image*"
unzip "*drive*.zip" "*/*/oxts*"
unzip "*calib*.zip"
```

### 🧩Download and evaluate the pretrained model

We provide model weights that could be downloaded and evaluated on RealEstate10K test set:

```sh
python -m misc.download_pretrained_models -o exp/re10k_v2
sh evaluate.sh exp/re10k_v2
```
Huggingface login (国内需要先 export HF_ENDPOINT=https://hf-mirror.com)
```
huggingface-cli login 
# Then input your huggingface token for authentication
```
## 🏃‍♂️ Run the Code
### 🧩Training

In order to train the model on RealEstate10K dataset execute this command:
```sh
python train.py \
  +experiment=layered_re10k \
  model.depth.version=v1 \
  train.logging=false 
```

For multiple GPU, we can run with this command:
```sh
bash train.sh
```
### 🧩inference
```sh
bash evaluate.sh
```
## 📑BibTeX
```

```

## 📖Acknowledgement

A large portion of codes in this repo is based on [Flash3D](https://github.com/eldar/flash3d), some of the code is borrowed from:

+ [MVDream](https://github.com/bytedance/MVDream)
+ [TensoRF](https://github.com/apchenstu/TensoRF)
+ [TriplaneGaussian](https://github.com/VAST-AI-Research/TriplaneGaussian)
+ [Unidepth](https://github.com/lpiccinelli-eth/UniDepth)
+ [StableNormal](https://github.com/Stable-X/StableNormal)

