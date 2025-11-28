# Real Image Denoising with Feature Attention Applied to Real Fluorescence Microscopy Images
This repository is a fork of Real Image Denoising with Feature Attention (RIDNet). RIDNet was first introduced in the following paper

[Saeed Anwar](https://saeed-anwar.github.io/), [Nick Barnes], "Real Image Denoising with Feature Attention", [ICCV (Oral), 2019](https://arxiv.org/abs/1904.07396) 

Find the original README.md file in the respective repository.


## Introduction
RIDNet proposes a network to succesfully denoise artificial and real noise in a single-stage, blind model. While older models usually employ 2 subnets or could only properly handle spatially invariant noise, RIDNet applies modular architecture and feature attention at its core, and performs significantly better than other models.
To test the model's generality, we are denoising Fluorence Real Fluorescence Microscopy Images (RMD) and comparing its blind effectiveness vs a version of the model fine tuned specifically for RMD.

<p align="center">
  <img width="600" src="https://github.com/SVA-BL00/RIDNet-on-FMD/blob/master/tables/Confocal_FISH_1_x50.png">
</p>

[Test images for RMD](https://drive.google.com/drive/folders/1FSMr4uGLzJs3ZhT7ntflCcVrvlwsx9aq?usp=sharing)

[Train and validate images for RMD](https://drive.google.com/drive/folders/1Z6psZh2tLZs3uK2wKquyPr_6uy8j1xxd?usp=sharing)


## Network
![Network](/Figs/Net.PNG)
The architecture of the proposed network. Different green colors of the conv layers denote different dilations while the smaller
size of the conv layer means the kernel is 1x1. The second row shows the architecture of each EAM.

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/FeatureAtt.PNG">
</p>
The feature attention mechanism for selecting the essential features.


## Train

This version uses the ridnet.pt as a starting point and fine tunes it with FMDTrain
and FMDVal. FMDTrain and FMDVal are in the same folder. Each type of subdataset contains 1000 data pairs, for validation we use the first 200 and for training we use the last 800.


```
python main.py --model RIDNET \
  --pre_train ridnet.pt \
  --save FMD_finetune \
  --dir_data dir/files/ \
  --data_train FMDTrain \
  --data_test FMDVal \
  --n_train 9600 \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-5 \
  --noise_g 50 \
  --n_colors 3 \
  --patch_size 64 \
  --print_every 5 \
  --n_threads 0
```

## Test
### Quick start
1. Download the trained models for our paper and place them in '/TestCode/experiment'.

    The real denoising model can be downloaded from [Google Drive](https://drive.google.com/open?id=1QxO6KFOVxaYYiwxliwngxhw_xCtInSHd) or [here](https://icedrive.net/0/e3Cb4ifYSl). The total size for all models is 5MB.

    [The FMD fine tune can be found here](https://drive.google.com/file/d/1YS189pDk90r9ev5R3EIlrxLzigc0AxOn/view?usp=sharing)

2. Cd to '/TestCode/code', run the following scripts.

    **You can use the following script to test the algorithm**

    ```
    !python main.py --model RIDNET \
      --pre_train /route/ridnet.pt \
      --test_only \
      --save FMD_test_extract \
      --dir_data /route/testing \
      --data_test FMDTest \
      --noise_g 50 \
      --n_colors 3 \
      --n_threads 0 \
      --save_results
    ```

    ```
    !python main.py --model RIDNET \
      --pre_train /route/model_latest.pt \
      --test_only \
      --save FMD_test_finetune \
      --dir_data /route/testing \
      --data_test FMDTest \
      --noise_g 50 \
      --n_colors 3 \
      --n_threads 0 \
      --save_results
    ```

## How to use? / Explanation

When you wish to add a new dataset to test or train, copy a similar dataset to yours, from
inside data. You can define how your dataset looks like, the format, and how to retrieve the
groundtruth and the noisy version. This version of RIDNet is hardcoded on 3 channels, so to
avoid bugs, grayscale images must be passed to 3 channels. This is the case with FMD. 
For examples, take a look at any fmd dataset in TestCode/code/data/

If testing or training is to be done in Google Colab, set the number of threads to 0 due to 
Colab's own limitations. Make sure that when you're training on a new dataset, you add 
different datasets for training and validating.


The repository is set up so the dataset object created will load the noisy/gt images, and 
return the lr_tensor, hr_tensor, filename. Dataloader manages the multi-threading part of
the project, loading batches in parallel. The trainer runs the training loop with the 
lr_tensor, hr_tensor, filename, and idx_scale (added by dataloader). It feeds it into
the model, calculates weights and loss. RIDNet uses an optimizer and a loss, found in the 
/loss folder. The model/ folder is where the main core of the project is located, with the
modular structure previously defined in ridnet, and all of the math operations and
convolutions are defined in ops.py.

## Results
**[RIDNet fine-tuned (ours) for FMD images](https://drive.google.com/drive/folders/1Nh7oFHP-52iLdlQseVludMRtIp1Cz8f6?usp=drive_link)**

**[RIDNet pre-trained (default) for FMD images](https://drive.google.com/drive/folders/1NV-nFaftL2gPcNDpbnKcCxqy-FOO2tX0?usp=sharing)**

The noisy images have the suffix                      _LR
The ground truth images have the suffix               _HR
The denoised images by the model have the suffix      _SR


### Results
<p align="center">
  <img width="600" src="https://github.com/SVA-BL00/RIDNet-on-FMD/blob/master/tables/Confocal_BPAE_B_1_x50.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/SVA-BL00/RIDNet-on-FMD/blob/master/tables/Confocal_BPAE_G_1_x50.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/SVA-BL00/RIDNet-on-FMD/blob/master/tables/Confocal_MICE_2_x50.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/SVA-BL00/RIDNet-on-FMD/blob/master/tables/TwoPhoton_BPAE_B_2_x50.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/SVA-BL00/RIDNet-on-FMD/blob/master/tables/WideField_BPAE_G_4_x50.png">
</p>

For more examples, check out the image folder with the results linked previously or the tables/


## Citation

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
