# Omni Aggregation Networks for Lightweight Image Super-Resolution (OmniSR)
## Accepted by CVPR2023

**The official repository with Pytorch**

Our paper can be downloaded from [[Arxiv]]().

Try OmniSR in Colab [ <a href="https://colab.research.google.com/github/Francis0625/OmniSR/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/github/Francis0625/OmniSR/blob/main/demo.ipynb)

## Attention

***We are archiving our code and awaiting approval for code public access!***

## Installation
**Clone this repo:**
```bash
git clone https://github.com/Francis0625/OmniSR.git
cd OmniSR
```
**Dependencies:**
- PyTorch 1.7.0
- Pillow 8.3.1; Matplotlib 3.3.4; opencv-python 4.5.3; Faiss 1.7.1; tqdm 4.61.2; Ninja 1.10.2

All dependencies for defining the environment are provided in `environment/omnisr_env.yaml`.
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/) (you may need to modify `omnisr_env.yaml` to install PyTorch that matches your own CUDA version following [https://pytorch.org/](https://pytorch.org/)):
```bash
conda env create -f ./environment/omnisr_env.yaml
```

## Training

## Inference with a pretrained OmniSR model

## Results

![performance](https://user-images.githubusercontent.com/18433587/227410356-6b69906b-416d-4d07-8127-41b08ab79c7a.PNG)


## To cite our paper

## Related Projects

## Citation
If this work helps your research, please cite the following paper:

```
@inproceedings{omni_sr,
  title      = {Omni Aggregation Networks for Lightweight Image Super-Resolution},
  author     = {Wang, Hang and Chen, Xuanhong and Ni, Bingbing and Liu, Yutian and Liu jinfan},
  booktitle  = {Conference on Computer Vision and Pattern Recognition},
  year       = {2023}
}
```
