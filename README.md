## Volumetric Wireframe Parsing from Neural Attraction Fields

> Volumetric Wireframe Parsing from Neural Attraction Fields
> 
> Nan Xue, Bin Tan, Yuxi Xiao, Liang Dong, Gui-Song Xia, Tianfu Wu
> 
> Preprint

<!-- insert the teaser -->
<img src='docs/teaser-neat.png'/>

## Installation 
### Cloning the Repository
```
git clone https://github.com/cherubicXN/neat.git --recursive
```
### Pytorch 1.13.1 + CUDA 11.7 (Ubuntu 22.04 LTS)
```bash
conda create -n neat python=3.10
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install gputil gitpython pyhocon tqdm matplotlib plotly opencv-python scikit-image trimesh
```

## Data Preparation
### DTU dataset
### BlendedMVS dataset
