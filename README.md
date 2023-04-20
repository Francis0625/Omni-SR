# Omni Aggregation Networks for Lightweight Image Super-Resolution (OmniSR)
## Accepted by CVPR2023

**The official repository with Pytorch**

Our paper can be downloaded from [[Arxiv]]().
 

## Installation
**Clone this repo:**
```bash
git clone https://github.com/Francis0625/OmniSR.git
cd OmniSR
```
**Dependencies:**
- PyTorch>1.10
- OpenCV
- Matplotlib 3.3.4 
- opencv-python 
- pyyaml
- tqdm
- numpy
- torchvision

## Preparation

- Download pretrained models, and copy them to ```./train_logs/```:

|  Settings   | CKPT name | CKPT url|
|  ----  | ----  | --- |
| DIV2K $\times 2$  | OmniSR_X2_DIV2K.zip | [baidu cloud](https://pan.baidu.com/s/1dJhTlhloaiYn9yImk6pa1Q) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/18lSvJq9CGCwDomkas2gh8K6UOq8qRLIw/view?usp=sharing)|
| DF2K $\times 2$  | OmniSR_X2_DF2K.zip | [baidu cloud](https://pan.baidu.com/s/1IK_bzB5gp2tK67zF-VV4Lg) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/12EvHRof0-kA2Wt_BzfFJBK1J0jbzfz-4/view?usp=sharing)| 
| DIV2K $\times 3$  | OmniSR_X3_DIV2K.zip | [baidu cloud](https://pan.baidu.com/s/19J5uONEOYWxAbEMWIF9qDA) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/1Rwg6o-RGC-TEiyVSVT9FS1iHjx5n948h/view?usp=sharing)|
| DF2K $\times 3$  | OmniSR_X3_DF2K.zip | [baidu cloud](https://pan.baidu.com/s/1mXL7AOUwyC91UDcEWFCh2Q) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/198R2c3nlyhL4FxMJSC_gccyL3O1gH_K6/view?usp=sharing)| 
| DIV2K $\times 4$  | OmniSR_X4_DIV2K.zip | [baidu cloud](https://pan.baidu.com/s/1ovxRa4-wOKZLq_nO6hddsg) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/1VoPUw0SRnCPAU8_R5Ue15bn2gwSBr97g/view?usp=sharing)|
| DF2K $\times 4$  | OmniSR_X4_DF2K.zip | [baidu cloud](https://pan.baidu.com/s/1KwO_shGLeais9Jne_cCINQ) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/17rJXJHBYt4Su8cMDMh-NOWMBdE6ki5em/view?usp=sharing)|

- Download benchmark images, and copy them to ```./benchmark/```: [baidu cloud](https://pan.baidu.com/s/1HsMtfjEzj4cztaF2sbnOMg) (passwd: sjtu)

 
## Evaluate Pretrained Models
### Example: evaluate the model trained with DF2K@X4:

- Step 1, the following cmd will report a performance evaluated with python script, and generated images are placed in ```./SR```

```
python -v "OmniSR_X4_DF2K" -s 994 -t tester_Matlab --test_dataset_name "Urban100"
```
- Step2, please execute the ```Evaluate_PSNR_SSIM.m``` script in the root directory to obtain the results reported in the paper. Please modify ```L8: methods = {'OmniSR_X4_DF2K'};``` to match the model name evaluated above.

## Training


## Visualization

![performance](./doc/imgs/vis.png)

## Results

![performance](https://user-images.githubusercontent.com/18433587/227410356-6b69906b-416d-4d07-8127-41b08ab79c7a.PNG)

## Related Projects

## To cite our paper
If this work helps your research, please cite the following paper:

```
@inproceedings{omni_sr,
  title      = {Omni Aggregation Networks for Lightweight Image Super-Resolution},
  author     = {Wang, Hang and Chen, Xuanhong and Ni, Bingbing and Liu, Yutian and Liu jinfan},
  booktitle  = {Conference on Computer Vision and Pattern Recognition},
  year       = {2023}
}
```
