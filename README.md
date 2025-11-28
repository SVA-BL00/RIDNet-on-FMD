# Real Image Denoising with Feature Attention Applied to Real Fluorescence Microscopy Images
This repository is a fork of Real Image Denoising with Feature Attention (RIDNet). RIDNet was first introduced in the following paper

[Saeed Anwar](https://saeed-anwar.github.io/), [Nick Barnes], "Real Image Denoising with Feature Attention", [ICCV (Oral), 2019](https://arxiv.org/abs/1904.07396) 

Find the original README.md file in the respective repository.


## Contents
1. [Introduction](#introduction)
2. [Network](#network)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
RIDNet proposes a network to succesfully denoise artificial and real noise in a single-stage, blind model. While older models usually employ 2 subnets or could only properly handle spatially invariant noise, RIDNet applies modular architecture and feature attention at its core, and performs significantly better than other models.
To test the model's generality, we are denoising Fluorence Real Fluorescence Microscopy Images (RMD) and comparing its blind effectiveness vs a version of the model fine tuned specifically for RMD.

[Test images for RMD](https://drive.google.com/drive/folders/1FSMr4uGLzJs3ZhT7ntflCcVrvlwsx9aq?usp=sharing)

[Train and validate images for RMD](https://drive.google.com/drive/folders/1Z6psZh2tLZs3uK2wKquyPr_6uy8j1xxd?usp=sharing)

<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/Front.PNG">
</p>
Sample results on a real noisy face image from RNI15 dataset.

## Network
![Network](/Figs/Net.PNG)
The architecture of the proposed network. Different green colors of the conv layers denote different dilations while the smaller
size of the conv layer means the kernel is 1x1. The second row shows the architecture of each EAM.

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/FeatureAtt.PNG">
</p>
The feature attention mechanism for selecting the essential features.


## Train
**Will be added later**

## Test
### Quick start
1. Download the trained models for our paper and place them in '/TestCode/experiment'.

    The real denoising model can be downloaded from [Google Drive](https://drive.google.com/open?id=1QxO6KFOVxaYYiwxliwngxhw_xCtInSHd) or [here](https://icedrive.net/0/e3Cb4ifYSl). The total size for all models is 5MB.

2. Cd to '/TestCode/code', run the following scripts.

    **You can use the following script to test the algorithm**

    ```bash
    #RIDNET
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model RIDNET --n_feats 64 --pre_train ../experiment/ridnet.pt --test_only --save_results --save 'RIDNET_RNI15' --testpath ../LR/LRBI/ --testset RNI15
    ```


## Results
**All the results for RIDNET can be downloaded from GoogleDrive from [SSID](https://drive.google.com/open?id=15peD5EvQ5eQmd-YOtEZLd9_D4oQwWT9e), [RNI15](https://drive.google.com/open?id=1PqLHY6okpD8BRU5mig0wrg-Xhx3i-16C) and [DnD](https://noise.visinf.tu-darmstadt.de/submission-detail). The size of the results is 65MB** 

### Quantitative Results
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/DnDTable.PNG">
</p>
The performance of state-of-the-art algorithms on widely used publicly available DnD dataset in terms of PSNR (in dB) and SSIM. The best results are highlighted in bold.

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/SSIDTable.PNG">
</p>
The quantitative results (in PSNR (dB)) for the SSID and Nam datasets.. The best results are presented in bold.

For more information, please refer to our [paper](https://arxiv.org/abs/1904.07396)

### Visual Results
![Visual_PSNR_DnD1](/Figs/DnD.PNG)
A real noisy example from DND dataset for comparison of our method against the state-of-the-art algorithms.

![Visual_PSNR_DnD2](/Figs/DnD2.PNG)
![Visual_PSNR_Dnd3](/Figs/DnD3.PNG)
Comparison on more samples from DnD. The sharpness of the edges on the objects and textures restored by our method is the best.

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/RNI15.PNG">
</p>
A real high noise example from RNI15 dataset. Our method is able to remove the noise in textured and smooth areas without introducing artifacts

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/SSID.PNG">
</p>
A challenging example from SSID dataset. Our method can remove noise and restore true colors

![Visual_PSNR_SSIM_BI](/Figs/SSID3.PNG)
![Visual_PSNR_SSIM_BI](/Figs/SSID2.PNG)
Few more examples from SSID dataset.

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{anwar2019ridnet,
  title={Real Image Denoising with Feature Attention},
  author={Anwar, Saeed and Barnes, Nick},
  journal={IEEE International Conference on Computer Vision (ICCV-Oral)},
  year={2019}
}

@article{Anwar2020IERD,
  author = {Anwar, Saeed and Huynh, Cong P. and Porikli, Fatih },
    title = {Identity Enhanced Image Denoising},
    journal={IEEE Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year={2020}
}
```
## Acknowledgements
This code is built on [DRLN (PyTorch)](https://github.com/saeed-anwar/DRLN)
