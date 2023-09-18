<div align="center">

<p align="left">
  <img src="../imgs/Logos/HI_Title.png" >
</p>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python-3776AB?&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.0-EE4C2C?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Pytorch Lightning 2.0-792EE5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://albumentations.ai/"><img alt="Albumentations" src="https://img.shields.io/badge/Albumentations 1.3 -cc0000"></a>
<a href="https://hydra.cc/"><img alt="L: Hydra" src="https://img.shields.io/badge/Hydra 1.3-89b8cd" ></a>

<a href="https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html"><img alt="Tensorboard" src="https://img.shields.io/badge/Logging-Tensorboard-FF6F00"></a>
<a href="https://black.readthedocs.io/en/stable"><img alt="L: Hydra" src="https://img.shields.io/badge/Code Style-Black-black" ></a>

---
<img alt="L: SemSeg" src="https://img.shields.io/badge/Task-Semantic Segmentation-rgb(68, 1, 84)">
<img alt="L: MultLabel" src="https://img.shields.io/badge/Task-Multilabel Segmentation-rgb(68, 1, 84)">
<img alt="L: InstSeg" src="https://img.shields.io/badge/Task-Instance Segmentation-rgb(68, 1, 84)">

<a href="https://arxiv.org/pdf/1904.04514.pdf"><img alt="L: HRNet" src="https://img.shields.io/badge/Model -High Resolution Network (HRNet)-rgb(33, 145, 140)" ></a>
<a href="https://arxiv.org/pdf/1909.11065.pdf"><img alt="L: OCR" src="https://img.shields.io/badge/Model -Object Contextual Representation (OCR)-rgb(33, 145, 140)" ></a>
<a href="https://arxiv.org/pdf/2005.10821.pdf"><img alt="L: MS" src="https://img.shields.io/badge/Model -Hierarchical Multi Scale Attention (MS OCR)-rgb(33, 145, 140)" ></a>
<a href="https://arxiv.org/pdf/1706.05587.pdf"><img alt="L: DL" src="https://img.shields.io/badge/Model -DeepLabv3 -rgb(33, 145, 140)" ></a>
<a href="https://arxiv.org/pdf/1411.4038.pdf"><img alt="L: FCN" src="https://img.shields.io/badge/Model -Fully Convolutional Network (FCN) -rgb(33, 145, 140)" ></a>
<a href="https://arxiv.org/pdf/1505.04597.pdf"><img alt="L: UNET" src="https://img.shields.io/badge/Model -UNet -rgb(33, 145, 140)" ></a>
<a href="https://arxiv.org/pdf/1703.06870.pdf"><img alt="L: MRCNN" src="https://img.shields.io/badge/Model - MASK RCNN -rgb(33, 145, 140)" ></a>

<a href="https://www.cityscapes-dataset.com/"><img alt="L: CS" src="https://img.shields.io/badge/Dataset-Cityscapes -rgb(253, 231, 37)" ></a>
<a href="https://cs.stanford.edu/~roozbeh/pascal-context/"><img alt="L: VOC" src="https://img.shields.io/badge/Dataset-PASCAL VOC 2010 Context -rgb(253, 231, 37)" ></a>
<a href="http://sceneparsing.csail.mit.edu/"><img alt="L: ADE20K" src="https://img.shields.io/badge/Dataset-ADE20K -rgb(253, 231, 37)" ></a>


---
</div>

This repository contains an easy-to-use and flexibly customizable framework for training semantic
segmentation models.
The focus was put on being usable out of the box, without being a black box and giving the possibility to be adapted to individual projects.
Therefore, this repository is designed in a modular way, to be extended with additional models and datasets, as well as other optimizers, schedulers,
metrics, loss functions and data augmentation pipelines.
In addition, popular packages such as Pytorch Lightning, Hydra and Albumentations were used 
to enable features, among others, such as multi-GPU, device independent and mixed precision training as well as
easy job configuration and easy construction of data augmentation pipelines.
Several architectures like [High-Resolution Network (HRNet)](https://arxiv.org/pdf/1904.04514.pdf), 
[Object Contextual Representation (OCR)](https://arxiv.org/pdf/1909.11065.pdf), 
[Hierarchical Multi-Scale Attention (MS OCR)](https://arxiv.org/pdf/2005.10821.pdf), 
[FCN](https://arxiv.org/pdf/1411.4038.pdf), 
[DeepLabv3](https://arxiv.org/pdf/1706.05587.pdf),
[UNet](https://arxiv.org/pdf/1505.04597.pdf) and [Mask RCNN](https://arxiv.org/pdf/1703.06870.pdf) are already
supported, as well as benchmark datasets like [Cityscapes](https://www.cityscapes-dataset.com/) (coarse and
fine), [PASCAL VOC2010 Context](https://cs.stanford.edu/~roozbeh/pascal-context/) and [ADE20K](http://sceneparsing.csail.mit.edu/).
Additionally, [Region Mutual Information (RMI)](https://arxiv.org/pdf/1910.12037.pdf) loss is included.
All this together provides the ability to compare different state-of-the-art (SOTA) segmentation models under
same conditions on different datasets.

The following contains information about how to [set up the data](#setting-up-the-data)
and [run the code](#running-code).
A comparison between different SOTA approaches (HRNet, OCR,OCR+ASPP, MS OCR) on the Cityscapes and
PASCAL VOC Context datasets is shown in the [experiments](#experiments) section.
For an advanced use of this framework, the [***config/*
folder**](/config#walkthrough-the-config-jungle) contains a full explanation of all available
configurations and how to customize the code to your needs.

## Table of Contents
- [Overview](#overview)
- [How To Run](#how-to-run)
  - [Requirements](#requirements)
  - [Setting up the Data](#setting-up-the-data)
    - [Cityscapes](#cityscapes)
    - [Cityscapes_Coarse](#cityscapes_coarse)
    - [PASCAL Context](#pascal-context)
  - [Download Pretrained Weights](#download-pretrained-weights)
  - [Running Code](#running-code)
    - [Baseline](#baseline)
    - [Selecting a Model](#selecting-a-model)
    - [Selecting a Dataset](#selecting-a-dataset)
    - [Changing Hyperparmeters](#changing-hyperparmeters)
    - [Changing Loss Function(s)](#changing-loss-functions)
    - [Logging and Checkpointing](#logging-and-checkpointing)
    - [Run Testing/Validation](#run-testingvalidation)
    - [Additional Tools](#additional-tools)
- [Experiments](#experiments)
  - [Cityscapes](#cityscapes-1)
  - [Pascal VOC2010 Context](#pascal-voc2010-context)

# Overview

**!!!Currently in progress numbers are generated with an older version of the repo**

**Cityscapes**

Overview about the results on the **Cityscapes val** set.
The best result from three runs (mean intersection over union, mIoU) is reported.
A more detailed analysis is given in the [experiments](#cityscapes-1) section.

| Model                | Baseline | RMI loss  | Paddle weights | Mapillary pretrained | using Coarse Data | Mapillary + Coarse Data + RMI |
|----------------------|:--------:|:---------:|:--------------:|:--------------------:|:-----------------:|:-----------------------------:|
| HRNET                |  81.44   |   81.89   |     81.74      |      **83.02**       |       82.03       |               -               |
| OCR                  |  81.37   |   82.08   |     81.89      |      **83.37**       |       82.24       |               -               |
| OCR + ASPP           |  81.53   | **82.20** |       -        |          -           |         -         |               -               |
| MS OCR [0.5, 1.]     |  81.49   |   82.59   |     82.18      |        83.63         |       82.26       |           **84.80**           |
| MS OCR [0.5, 1., 2.] |  82.30   |   82.88   |     82.79      |        84.31         |       82.95       |           **85.31**           |



**Pascal VOC**

Overview of the results on the **PASCAL VOC2010 Context** set.
The best result from three runs (mean intersection over union, mIoU) is reported.
A more detailed analysis is given in the [experiments](#pascal-voc2010-context) section.
Models are tested with and without multiscale testing.

|       Model        | Baseline | PaddleClass | PaddleClass + RMI | Baseline * | PaddleClass * | PaddleClass + RMI * |
|:------------------:|:--------:|:-----------:|:-----------------:|:----------:|:-------------:|:-------------------:|
|       HRNet        |  50.61   |    54.19    |       54.55       |   53.21    |     56.31     |        56.67        |
|        OCR         |  51.68   |    55.98    |       57.20       |   53.94    |     58.18     |        59.34        |
|  MS OCR [0.5, 1]   |  52.00   |    56.03    |       57.54       |   54.24    |     58.10     |        59.58        |
| MS OCR [0.5, 1, 2] |  53.71   |    57.50    |       58.97       |   54.81    |     58.68     |        59.94        |

**Multi Scale Testing with Scales [0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.] and flipping*

### References

This repository adopts code from the following sources:

- **HRNet** ( High-Resolution Representations for Semantic
  Segmentation, [paper](https://arxiv.org/pdf/1904.04514.pdf)
  , [source code](https://github.com/HRNet/HRNet-Semantic-Segmentation))
- **OCR** (Object Contextual Representation, [paper](https://arxiv.org/pdf/1909.11065.pdf)
  , [source code](https://github.com/HRNet/HRNet-Semantic-Segmentation))
- **OCR + ASPP** (Combines OCR with an ASPP
  module, [source code](https://github.com/NVIDIA/semantic-segmentation/tree/main/network))
- **MS OCR** (Hierarchical Multi-Scale Attention for Semantic
  Segmentation, [paper](https://arxiv.org/pdf/2005.10821.pdf)
  , [source code](https://github.com/NVIDIA/semantic-segmentation/tree/main/network))
- **UNet** ([paper](https://arxiv.org/pdf/1505.04597.pdf), [source code](https://github.com/milesial/Pytorch-UNet))
- **RMI** (Region Mutual Information Loss for Semantic
  Segmentation, [paper](https://arxiv.org/pdf/1910.12037.pdf)
  , [source code](https://github.com/ZJULearning/RMI))
- **DC+CE** (combination from Dice and Cross Entropy Loss), **TOPK**, **TOPK+DC** are all from
  nnUNet ([paper](https://www.nature.com/articles/s41592-020-01008-z)
  , [source code](https://github.com/MIC-DKFZ/nnUNet))
- **Lightning-Hydra Template**: [source code](https://github.com/ashleve/lightning-hydra-template)
