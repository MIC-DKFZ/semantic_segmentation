# How To Run

## Requirements

Install the needed packages by the following command. You might need to adapt the cuda versions for
torch and torchvision specified in *requirements.txt*.
Find a pytorch installation guide for your
system [here](https://pytorch.org/get-started/locally/#start-locally).

````shell
pip install -r requirements.txt
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

#### Cityscapes

<details><summary>Click to expand/collapse</summary>
<p>

Download the Cityscapes dataset from [here](https://www.cityscapes-dataset.com/downloads/).
You have to create an account and afterward download: *leftImg8bit_trainvaltest.zip* (11GB)  and 
*gtFine_trainvaltest.zip* (241MB).
Unzip them and put them into a folder, the structure of the folder should now look like this:

````
cityscapes
    ├── leftImg8bit_trainvaltest
    │   └── leftImg8bit
    │       ├── train
    │       │   ├── aachen
    │       │   └── ...  
    │       └── val
    │           ├── frankfurt
    │           └── ...  
    └── gtFine_trainvaltest
        └── gtFine
            ├── test
            │   └── ...
            ├── train
            │   └── ...        
            └── val
                └── ...

````

The cityscapes dataset contains 34 classes by default but only 19 of them are used in practices.
To avoid doing this conversion at each training step, it is done in a preprocessing step.
To do this preprocessing run the following code with adjusting the data_path to the location which
contains the *leftImg8bit_trainvaltest* and *gtFine_trainvaltest* folders.
This will create a new mask for each data sample with the converted class labeling which will be
merged into the folder/data structure of the cityscapes dataset.

````
python src/data_processing/process_cityscapes.py --data_path="/home/.../Datasets/cityscapes"
````

After downloading and setting up the data, the last step is to adjust the path in the configuration.
Open the file of the environment you are using (by default *config/environment/local.yaml*) and
adopt the cityscapes path to the location of the folder where your *gtFine_trainvaltest* and 
*leftImg8bit_trainvaltest* are.
For this example this would look like this:

````yaml
config/environment/local.yaml
  ─────────────────────────────
...
paths:
  cityscapes: /home/.../Datasets/cityscapes
````

</p>
</details>

#### Cityscapes_Coarse

<details><summary>Click to expand/collapse</summary>
<p>

The cityscapes dataset provides 20k additional coarse labeled images.
This is an extension to cityscapes rather than a separate dataset, so [cityscapes](#cityscapes)
should be set up first.
Download the cityscapes_coarse dataset
from [here](https://www.cityscapes-dataset.com/downloads/) (*leftImg8bit_trainextra.zip (44GB)*
and *gtCoarse.zip (1.3GB)*) and unzip them into the same folder as your cityscapes data.
You then should end up with this:

````
cityscapes
    ├── leftImg8bit_trainvaltest
    │   └── leftImg8bit
    │       └── ...
    ├── gtFine_trainvaltest
    │   └── gtFine
    │       └── ...
    ├── leftImg8bit_trainextra
    │   └── leftImg8bit
    │       └── ...
    └── gtCoarse
        └── gtCoarse
            └── ...
````

Afterward process the cityscapes_coarse dataset in the same way as it was done for cityscapes by:

````shell
python src/data_processing/process_cityscapes_coarse.py --data_path="home/.../Datasets/cityscapes"
````

</p>
</details>

#### PASCAL Context

<details><summary>Click to expand/collapse</summary>
<p>

Click [here](https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz) for directly
downloading the labels or do it manually by downloading the file *trainval.tar.gz (30.7 MB file)*
from [PASCAL-Context](https://cs.stanford.edu/~roozbeh/pascal-context/#download).
Click [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar) for
directly downloading the images or do it manually by downloading the file 
*training/validation data (1.3GB tar file)*
from [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html#devkit).
Unzip both files and put them into a folder.
The structure of you folders should look like this:

````
Datasets
    ├── VOCtrainval_03-May-2010/VOCdevkit/VOC2010
    │   ├── Annotations
    │   ├── ImageSets
    │   └── ...
    └── trainval
        └──trainval
            ├── *.mat
            └── ...
````

Since the VOC2010 dataset contains a lot of unnecessary stuff (unnecessary for this repo), only the
required data is extracted and merged with the transformed label data from *trainval/*.
Run the following script which creates a new folder structure with only the relevant and transformed
data.

````shell
python datasets/VOC2010_Context/process_VOC2010_Context.py home/.../Datasets/
````

Afterwards a new dataset is created and the data from *trainval* and *VOCtrainval_03-May-2010*  is
not further needed.
The new dataset looks like this:

````
Datasets
    └── VOC2010_Context
        ├── Annotations
        │   ├── train
        │   └── val
        └── Images
            ├── train
            └── val
````

After downloading and setting up the data, the last step is to adjust the path in the configuration.
Open the file of the environment you are using (by default *config/environment/local.yaml*) and
adopt the VOC2010_Context path to the location of the folder where your *Images* and *Annotations*
are.
For this example this would look like this:

````yaml
config/environment/local.yaml
  ─────────────────────────────
...
paths:
  VOC2010_Context: /home/.../Datasets/VOC2010_Context
````

</p>
</details>

#### ADE20K

<details><summary>Click to expand/collapse</summary>
<p>

Click [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) for directly downloading the dataset or do it manually from [here](http://sceneparsing.csail.mit.edu/)
Unzip the folder and you then should end up with this:

````
ADEChallengeData2016
    ├── annotations
    │   ├── training
    │   └── validation
    └── images
        ├── training
        └── validation
````

Afterward process the ade20k dataset in the following way:

````shell
python src/data_processing/process_ade20k.py --data_path="home/.../Datasets/ADEChallengeData2016"
````
After downloading and setting up the data, the last step is to adjust the path in the configuration.
Open the file of the environment you are using (by default *config/environment/local.yaml*) and
adopt the ADE20K path to the location of the folder where your *images* and *annotations* are.
For this example this would look like this:

````yaml
config/environment/local.yaml
  ─────────────────────────────
...
paths:
  ADE20K: home/.../Datasets/ADEChallengeData2016
````

</p>
</details>

http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
http://sceneparsing.csail.mit.edu/


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
| hrnet_ocr_ms   |             |                           |             | BCE_PL           | ADAMW         | polynomial_warmup       |
| hrnet_ocr_aspp |             |                           |             |                  |               | polynomial_epoch_warmup |
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
