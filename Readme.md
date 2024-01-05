<div align="center">

<p align="left">
  <img src="imgs/Logos/HI_Title.png" >
</p>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python-3776AB?&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.0-EE4C2C?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.0-792EE5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://albumentations.ai/"><img alt="Albumentations" src="https://img.shields.io/badge/Albumentations 1.3 -cc0000"></a>
<a href="https://hydra.cc/"><img alt="L: Hydra" src="https://img.shields.io/badge/Hydra 1.3-89b8cd" ></a>

<a href="https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html"><img alt="Tensorboard" src="https://img.shields.io/badge/Logging-Tensorboard-FF6F00"></a>
<a href="https://black.readthedocs.io/en/stable"><img alt="L: Hydra" src="https://img.shields.io/badge/Code Style-Black-black" ></a>

---
<img alt="L: SemSeg" src="https://img.shields.io/badge/Task-Semantic Segmentation-rgb(33, 145, 140)">
<img alt="L: MultLabel" src="https://img.shields.io/badge/Task-Multilabel Segmentation-rgb(33, 145, 140)">
<img alt="L: InstSeg" src="https://img.shields.io/badge/Task-Instance Segmentation-rgb(33, 145, 140)">

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

**DACL Challenge:** The documentation for the Challenge on [Semantic Bridge Damage Segmentation](https://eval.ai/web/challenges/challenge-page/2130/overview) can be found [here](docs/DACL_Challenge.md).

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

# How To Run

## Requirements

Install the needed packages by the following command. You might need to adapt the cuda versions for
torch and torchvision specified in *requirements.txt*.
Find a pytorch installation guide for your
system [here](https://pytorch.org/get-started/locally/#start-locally).

````shell
pip install -r requirements.txt
````

When using SegNeXt you have to install [mmseg](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

````
pip install -U openmim
mmsegmentation
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
mim install mmdet
pip install regex
pip install ftfy
````


Among others, this repository is mainly built on the following packages.
You may want to familiarize yourself with their basic use beforehand.

- **[Pytorch](https://pytorch.org/)**: The machine learning/deep learning library used in this
  repository.
- **[Pytorch Lightning](https://www.pytorchlightning.ai/):**
  An open source framework for organizing Pytorch code and supporting machine learning development.
  It is automating most of the training loop and offers many useful features like mixed precision
  training.
  Lightning also makes code scalable to any hardware(CPU, GPU, TPU) without changing the code, as
  well as distributed training on multiple GPUs.
- **[Hydra](https://hydra.cc/docs/intro/):** Framework to simplify the development, organization,
  and configuration of machine learning experiments.
  It provides the ability to dynamically create a hierarchical configuration by composition and
  override it through config files and the command line.
- **[Albumentations](https://albumentations.ai):** Package for fast and flexible data augmentation
  in Semantic Segmentation (Albumentations is not limited to segmentation, but only that is used in
  this repository).
  Albumentations provides a lot of augmentations that can be used. Also random operations (e.g.
  random cropping) can be applied directly to images and masks.
- **[Torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)**: Collection of multiple PyTorch
  metrics and with a good integrity in Pytorch Lightning.
  Using torchmetrics makes it possible to use custom or predefined metrics and synchronization these
  between multiple devices.
- **[Black](https://black.readthedocs.io/en/stable/)**: Code Formatter used in this Repo. Check out the [official docs](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html) on how to use Black or [here](https://akshay-jain.medium.com/pycharm-black-with-formatting-on-auto-save-4797972cf5de) on how to integrate Black into PyCharm.


### Setting up the Data

How to set up each dataset is described [here](/config/dataset/)

### Download Pretrained Weights

Pretrained weights for HRNet-based models (HRNet, OCR, MS OCR) are available on ImageNet, PadddleClas and Mapillary.
Download the preferred weights (direct download links below) and put them in the *pretrained/*
folder.

- **ImageNet** weights: [source](https://github.com/HRNet/HRNet-Image-Classification#imagenet-pretrained-models), [direct download](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk)
- **PaddleClas**
  weights: [source](https://github.com/HRNet/HRNet-Image-Classification#imagenet-pretrained-models), [direct download](https://github.com/HRNet/HRNet-Image-Classification/releases/download/PretrainedWeights/HRNet_W48_C_ssld_pretrained.pth)
- **Mapillary**
  weights: [source](https://github.com/NVIDIA/semantic-segmentation#download-weights), [direct download](https://drive.google.com/file/d/1Whz--rurtBoIsfF-t3YB9NEt3GT6hBQI/view?usp=sharing)

## Running Code

The following is a **Quickstart** guide on how to run the code.
To adopt the configuration you can edit the */config/training.yaml* file directly or use the hydra
commandline syntax.
As you will see, the basic syntax of how to run and adopt the code is simple.
The crucial part is to know which parameters you can configure and how.
Therefore, the [*config/* folder](/config) explains in detail how the configuration is composed and
which parameters it contains.
For training with predefined setting on the benchmark dataset you can run:
````shell
python training.py experiment=Cityscapes/base
python training.py experiment=VOC2010_Context/base
python training.py experiment=ADE20K/base
````

## Basic Configure

The default settings and hyperparameters are defined in *config/training.yaml* and are overwritten by the 
corresponding *config/experiment/<experiment.name>.yaml* for the predefined experiments.
To run one of the predefined experiments in the *config/experiment* folder run:
````shell
python training.py experiment=<experiment.name>
python training.py experiment=Cityscapes/base
````

For further customization the basic hyperparameters (**epochs**, **lr**, **batch_size**, **val_batch_size**, **num_workers**) needed for training can be changed from the commandline as shown below:
````shell
python training.py epochs=400 batch_size=6 val_batch_size=6 num_workers=10 lr=0.001
````

All available options for configuring experiments are shown below, depending on if you want to train a semantic segmentation, multilabel segmentation or instance segmentation model.
They can be selected as shown here:
````shell
python training.py model=hrnet dataset=Cityscapes/base metric=mean_IoU trainer=SemSeg lossfunction=[CE,CE] optimizer=SGD lr_scheduler=polynomial
````

More details on how to configure and how to customize them can be found here.

### Semantic Segmentation

| **model**      | **dataset**                         | **metric**      | **trainer** | **lossfunction** | **optimizer** | **lr_scheduler**        |
|----------------|-------------------------------------|-----------------|-------------|------------------|---------------|-------------------------|
| hrnet          | Cityscapes/base                     | mean_IoU        | SemSeg      | CE               | SGD           | polynomial              |
| hrnet_ocr      | Cityscapes/coarse                   | mean_IoU_Class  |             | wCE              | MADGRAD       | polynomial_epoch        |
| hrnet_ocr_ms   | Citycapes/cross_validation          | mean_Dice       |             | RMI              | ADAMW         | polynomial_warmup       |
| hrnet_ocr_aspp | Citycapes/sampling                  | mean_Dice_Class |             | wRMI             |               | polynomial_epoch_warmup |
| FCN            | Citycapes/sampling_cross_validation |                 |             | DC               |               |                         |
| DeepLabv3      | VOC2010/base                        |                 |             | DC_CE            |               |                         |
| UNet           |                                     |                 |             |                  |               |                         |

### Multilabel (Semantic) Segmentation

| **model**      | **dataset** | **metric**                | **trainer** | **lossfunction** | **optimizer** | **lr_scheduler**        |
|----------------|-------------|---------------------------|-------------|------------------|---------------|-------------------------|
| hrnet          | ---         | mean_IoU_Class_Multilabel | SemSegML    | BCEwL            | SGD           | polynomial              |
| hrnet_ocr      |             |                           |             | BCE              | MADGRAD       | polynomial_epoch        |
| hrnet_ocr_ms   |             |                           |             | mlDC             | ADAMW         | polynomial_warmup       |
| hrnet_ocr_aspp |             |                           |             | mlJL             |               | polynomial_epoch_warmup |
| FCN            |             |                           |             |                  |               |                         |
| DeepLabv3      |             |                           |             |                  |               |                         |
| UNet           |             |                           |             |                  |               |                         |

### Instance Segmentation

| **model** | **dataset** | **metric** | **trainer** | **lossfunction** | **optimizer** | **lr_scheduler**        |
|-----------|-------------|------------|-------------|------------------|---------------|-------------------------|
| Mask_RCNN | PennFudan   | MAP        | InstSeg     | ---              | SGD           | polynomial              |
|           |             |            |             |                  | MADGRAD       | polynomial_epoch        |
|           |             |            |             |                  | ADAMW         | polynomial_warmup       |
|           |             |            |             |                  |               | polynomial_epoch_warmup |

## Logging and Checkpointing

The logging structure of the output folder is depicted below.
The ``logging.logdir=<some.folder.dir>`` argument defines the logging folder (*"logs/"* by default).
For a better overview, experiments can also be named by ``logging.name="my_experiment"`` ("run"
by default).
The parameters which are logged are: Hyperparameters (like epochs, batch_size, initial lr,...), 
metrics, loss (train+val), time (train+val) and learning rate as well as example predictions 
during validation  (can be disabled by setting ``logging.num_example_predictions=0``).
Checkpointing can be disabled/enabled by ``pl_trainer.enable_checkpointing= <True or False>``.
To resume training from a checkpoint use the *continue_from* argument.
If you want to finetune from a checkpoint (only load the weights) use the *finetune_from* argument.
Both should lead to a model checkpoint, which is a *.ckpt file.

````shell
#Continue from Checkpoint + Example
python training.py +continue_from=<path.to.ckpt>
python training.py +continue_from=logs/.../checkpoints/last_epoch_74.ckpt
#Finetune from Checkpoint + Example
python training.py +finetune_from=<path.to.ckpt>
python training.py +finetune_from=logs/.../checkpoints/last_epoch_74.ckpt
````

````
loggin.logdir                               # logs/ by default
    └── dataset.name                        # Name of the used Dataset
        └──model.name                       # Name of the used Model
           └──logging.name+experiment_overrides          # Parameters that have been overwritten from the commandline
              └──date                       # Date as unique identifier
                  ├── checkpoints/          # If checkpointing is enabled this contains the best and the last epochs checkpoint
                  ├── .hydra/               # Contains hydra config files
                  ├── testing/              # (Optional) only when model is tested - contains testing results
                  ├── event.out...          # Tensorboard log
                  ├── training.log          # Console logging file
                  └── hparams.yaml          # Resolved config
````

Since [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#run-tensorboard)
is used for logging, you can view the logged results by the following command.
Tensorboard will include all logs (*.tfevents.* files) found in any subdirectory of *--logdir*.
This means by the level of the selected *--logdir* you can define which experiments (runs) should be
included into the tensorboard session.

````shell
tensorboard --logdir=<some.dir>
# example for viewing a single experiment
tensorboard --logdir=/home/.../logs/Cityscapes/hrnet/run_.../2022-04-21_10-25-30
# example for viewing all runs in .../run_..
tensorboard --logdir=/home/.../logs/Cityscapes/hrnet/run_...
````
**Node**: Tensorboard does not show the images for all epochs (like the example predictions or confusion matrix) by default.
If you need to see all images use ``tensorboard --logdir=<some.dir> --samples_per_plugin images=<num_epochs>``

## Testing and Inference

For testing and predicting a separate script exist
Both require a ckpt_dir with points to a checkpoint as described in [ckpt_dir](#ckptdir). 
Also, the test time augmentation setting defined in the configs is used for both scripts, see [tta](#test-time-augmentation-tta) for more details.

**Node**: For testing and predicting with your model, checkpointing hast to be enabled during
training (``python training.py .... pl_trainer.enable_checkpointing=True`` or set to True in the config file.

### Testing

For testing your trained model use the testing.py script.
If you don't have a test set ensure you dataset return the validation set instead (done by the predefined dataset classes).
For testing, the configuration used during training is reconstructed and composed with the arguments given by the command line.
You can change parameter and settings from the commandline using hydra in the same way as for the training.
In the config the tta argument (see *config/training.yaml*) can be used to define the testing behaviour and test time augmentations.
Since the configuration is composed of different configs, it can 
happen that hydra raises error because some attributes from the commandline didn't exist at the 
current config. If this happens add a "++" in front of the argument (e.g. `` ... ++example=True``) or 
load the missing config group manually (e.g. ``... model=mymodel MODEL.specific_parameter=X``).

````shell
python testing.py ckpt_dir=<path.to.ckpt>
````

**Node**: For testing the same config is reconstructed which was used during training.
This means if you train and test on different devices you have to adapt the environment
parameter(``python testing.py ckpt_dir=<some.dir> environment=some_env``).

### Predicting/Inference

To generate prediction from trained model the *predict.py* script can be used.
The input of the script has to be an input folder, an output folder as well as the as a model checkpoint.
When the *--save_probabilities* flag is given, additionally the softmax predictions are stored as a *.npz* file

````shell
python predict.py -i <path.to.img.folder> -o <path.to.output.folder> ckpt_dir=<path.to.ckpt>
python predict.py -i <path.to.img.folder> -o <path.to.output.folder> ckpt_dir=<path.to.ckpt> --save_probabilities
````

For **semantic segmentation** models the class index map is saved to the output folder.
For **multilabel segmentation** a binary map is saved for each class (a *_i* postfix is added to the input name for each class i).
For **instance segmentation** a class index map (same as for semantic segmentation) is saved as well an instance map.
The *--save_probabilities* flag does not work for instance segmentation

**Note**: The Dataloader in prediction.py only can deal with .png, .jpg and .tif images (and everything which is readable by cv2.imread()) 
If you have other datatypes or require custom data loading, you may have to adopt the load_img() methode.

### ckpt_dir

The ckpt_dir is needed for testing as well for predicting and should point to the output directory which is created during training (contains the logs and the checkpoint folder).
````shell
python testing.py -i  ckpt_dir=".../logs/Cityscapes/hrnet/run__experiment_Cityscapes.base/2023-09-13_17-07-11"
````
When using the cross validation dataset also an model ensemble can be used. In this case the ckpt dir should point to the folder which contains the different fold folders (fold_0,fold_1,...)
````shell
python testing.py -i  ckpt_dir=".../logs/Cityscapes_CV/hrnet/run__experiment_Cityscapes.base"
````

### Test Time Augmentation (tta)

For configuring test time augmentations (tta) use the *tta* entry in the config. 
By default, the scales [0.5, 0.75, 1, 1.25, 1.5, 1.75] (tta.scales) are used as well a horizontal flipping (tta.hflip).
To disable tta set: `tta.scales=[1] tta.hflip=False tta.vflip=False`.

````shell
# Disable TTA
python testing.py ckpt_dir=<...> tta.scales=[1] tta.hflip=False tta.vflip=False
python predict.py -i=<...> -o=<...> ckpt_dir=<...> tta.scales=[1] tta.hflip=False tta.vflip=False
````

## Additional Tools

The [tools/](/tools) folder contains some additional and useful tools for developing and experimenting. 
It is not guaranteed that these tools will work for all kind of use-cases, datasets and datatypes but even then
they can be used as a starting point and can be adapted with a few changes. 
The Scripts and tools are shortly listed in the following and explained in more detail in [tools/](/tools).

- **show_data.py**: Load and Visualize the pytorch dataset which is defined in the dataset config.
  - dataset: Name of the dataset config (see [here](#selecting-a-dataset))
  ````shell
  pyhton tools/show_data.py dataset=<dataset.name>
  pyhton tools/show_data.py dataset=Cityscapes
  ````

- **show_prediction.py**: Show the predictions of a trained model. Basically has the same syntax 
as the [validation/testing](#run-validationtesting), but visualizes the result instead of calculating 
metrics.
  - ckpt_dir: path to the checkpoint which should be used
  ````shell
  python tools/show_prediction.py ckpt_dir=<path>
  python tools/show_prediction.py ckpt_dir=ckpt_dir="/../Semantic_Segmentation/logs/VOC2010_Context/hrnet/run_/2022-02-15_13-51-42"
  ````

- **dataset_stats.py**: Getting some basic stats and visualizations about the dataset like: mean and std for each channel, appearances and ratio of classes and potential class weights.  
  - dataset: Name of the dataset config (see [here](#selecting-a-dataset))
  ````shell
  python tools/dataset_stats_old.py dataset=<dataset.name>
  pyhton tools/dataset_stats_old.py dataset=Cityscapes
  ````


# Acknowledgements

<p align="left">
  <img src="imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="imgs/Logos/DKFZ_Logo.png" width="500"> 
</p>

This Repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).