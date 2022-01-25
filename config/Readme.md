TODO:
- Data Augmentation
- Maybe initialization using hydra

# Walkthrough the Config Jungle

In this repository [Hydra](https://hydra.cc/) is used for configuring and managing experiments.
Therefore configuration files and there handling are of major importance which is why they get explained in more detail in the following.
First the basic functionally of Hydra is explained shortly. 
Since Hydra uses the [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/) package to handle .yaml files, also Omegaconf and YAML are introduced shortly.
Below a walkthrough through all available configuration in this repository and there use is give.

## Basics of Hydra

<details><summary>Click to expand/collapse</summary>
<p>

[Hydra](https://hydra.cc/) automatically loads and composes different configuration files and allows to dynamically overriding values at runtime via the command line.
In Hydra *.yaml* files are used to specify configurations. 
In this repository the *config/baseline.yaml* can be seen as the main file and from there further configs are composed.
Each subfolder in *config/* is a [config group](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/), which contains a separate config file for each alternative inside.
For example the config group *model* is located in the *config/model* subfolder with a separate .yaml file for each available model (hrnet.yaml, hrnet_ocr.yaml, ...).
The individual config files contain model/dataset/etc. specific parameters, like the number of channels in a layer of the model or the number of classes in a dataset.
Having a separate config files for each model/dataset/etc. makes it easy to switch between them and arbitrary combine different config files form different config groups.
Additionally, this ensures that only the relevant parameters are loaded into the job configuration.
Hydra builds the job configuration by composing the config files from the different config groups.
Basically, exactly one config file from each config group is used in this process.
To tell hydra how to compose the job configuration, a [default list](https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/) is used, which specifies which configuration file from which configuration group should be used and in which order they are composed.
The default list is defined in *config/baseline.yaml* and looks like this:
````yaml
baseline.yaml
─────────────────────────────
defaults:
  - _self_
  - callbacks: default
  - data_augmentations: default
  - model: hrnet
  - dataset: Cityscapes
  - environment: local
````
The configs of each config group are merged from top to bottom, where later groups can overwrite the parameters of earlier groups.
Besides the order the default lists also sets default values for the config groups.
This means if not changed in this case the parameters defined in *baseline.yaml*,..., *model/hrnet.yaml* and *datasets/Cityscapes.yaml* are used.
To change the used config file of a config group, the corresponding entry in the default list can be changed in the *baseline.yaml*, or the entry can be overwritten from the commandline.
The [commandline syntax](https://hydra.cc/docs/advanced/override_grammar/basic/#working-with-your-shell) of Hydra is straight forward and elements can be changed, added or removed in the following way.
Thereby this syntax is the same for single parameters like *batch_size* but also for config files from config groups like *model*.
All available options to change for parameters and config groups are shown below in the *Configure the Configuration* part.
````shell
python main.py  parameter_to_change=<new_value>  +parameter_to_add=<a_value>  ~parameter_to_delete
#Example for single parameters
python main.py  batch_size=3 +extra_lr=0.001 ~momentum
#Example for config groups
python main.py  model=hrnet_ocr +parameters=basic ~environment   
````
This was only a short introduction how to use hydra to work with this repository.
For more information on Hydra, check out the official docs or one of the following sources, which provide some nice insights into Hydra
([source1](https://github.com/lkhphuc/lightning-hydra-template),
[source2](https://www.sscardapane.it/tutorials/hydra-tutorial/) and
[source3](https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b)).

</p>
</details>

## OmegaConf in a Nutshell

<details><summary>Click to expand/collapse</summary>
<p>

Hydra uses the package [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/) to handle *.yaml* files. 
An introduction to yaml is give below.
OnegaConf gives a lot of possibilities to work with yaml files, but since hydra manages this for you in the background you do not need much of it for a basic use.
If you need further functionality, for example if you manually want to load or save files look 
at the official [OmegaConf docs](https://omegaconf.readthedocs.io/en/2.1_branch/).
The [**Access and Manipulation**](https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-manipulation) of the cfg in python is straight forward:
````yaml
example.yaml
─────────────────────────────
Parameters:
  lr: 0.01
  epochs: 100
  whatever: 
  - 42
  - ...
````
````python3
main.py
─────────────────────────────
from omegaconf import OmegaConf
...
#for the example manually load the cfg, normally done by hydra automatically
cfg = OmegaConf.load("example.yaml") 

#acess over object and dictionary style
lr = cfg.Parameters.lr
lr = cfg["Parameters"]["lr"]

#Manipulation in the same way
cfg.Parameters.epochs = 300
cfg["Parameters"]["epochs"] = 300

##same goes for accessing lists
x = cfg.Parameters.whatever[0]
````
[**Variable interpolation**](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) is another important concept of Hydra and Omegaconf.
When defining config files the situation will occur that variables from other config files are needed.
For example for defining the last layer of a model, the number of classes, which is defined in the specific dataset configs, may be needed.
Therefore, variable interpolation is used, which can be seen as a link to a position in the config, that resolved at runtime.
Therefore, the variable is resolved from the dataset which used the current job and no conflicts occur between different dataset configs and the model config.
Variable interpolation is done with the following syntax:``${path.to.another.node.in.the.config}``.
and in that case the value will be the value of that node
````yaml
dataset/a_dataset.yaml
─────────────────────────────
dataset:
  num_classes: 24
````
````yaml
model/a_model.yaml
─────────────────────────────
num_output_classes: ${dataset.number_classes}      #num_output_classes will have the value 24 at runtime
````
</p>
</details>

## YAML in a Nutshell

<details><summary>Click to expand/collapse</summary>
<p>

This is only a short introduction to YAML and only shows its basic syntax. This should be enough for defining you own yaml files but if you need more informations they can be found [here](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html) for example.

Some  **Basic Assignments** are shown here:
````yaml
example.yaml
─────────────────────────────
#Comments in yaml
number: 10                  # Simple value, works for int and float.
string: Text|"Text"         #Strings, Quotation marks are not neccesarily required if the value is text(a string).
empty: None| |Empty|Null
explicit_Type: !!float 1   # Explicitly defined type. works as well for other types like str etc.
missing_vale: ???          # Missing required value. The  has to be given e.g. from the commandline.
optional opt_value:        # Optional Value. Can be empty or ???, and will only be considered if it has a value.
value2: ${number}          # Value interpolation (takes the value of attribute number, in this case 10). $ indicates reference and {} is required.
value3: "myvalue ${number}"  # String interpolation, same as value interpolation just with string output.
booleans: on|off|yes|no|true|false|True|False|TRUE|FALSE    #multiple possibilities to define boolean values.
````
**List** are defined in the following way:
````yaml
#LISTS
alist:
- elem1                   #elements need to be on the same indentation level
- elem2                   # there needs to be a space between dash and element
- ...
samelist: [elem1, elem2, ...]
````
**Dictionaries** are defined in the following way:
````yaml
adict:
  key1: val1                    #keys must be indented
  key2: val2                    #there has to be a space between colon and value
  ...
samedict: {key1: val1, key2: val2, ...}
````
For more complex files you will end up with lists of dictionaries and dictionaries of list and mixtures of both. But basically thats it!
</p>
</details>

# Configure the Configuration

### Basic Hyperparameters
The following hyperparameters are supported and can be changed in the *baseline.yaml* directly or can be overwritten from the command line as shown below:
 - **epochs:** number of epochs for training.
 - **batch_size:** defines the batch size during training (per GPU). 
 - **val_batch_size:** defines the batch size during validation and testing (also per GPU). Is set to batch_size if not specified.
 - **num_workers:** number of workers for the dataloaders.
 - **lr:** initial learning rate for training.
 - **wd:** weight decay (optimizer parameter)
 - **momentum:** momentum (optimizer parameter)
 - **optimizer**: defines the optimizer to use. Currently only Stochastic Gradient Descent (SGD) is supported (hence default: ``optimizer="sgd"``)
   - "sgd": [Stochastic Gradient Descent](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
 - **lr_scheduler**: defines the lr scheduler to use. By default ``lr_scheduler="poly"`` is used
   - "poly": Polynomial lr scheduler over the number of steps: *(1-current_step/max_step)^0.9*
   - "poly_epoch": Polynomial lr scheduler over number of epochs: *(1-current_epoch/max_epoch)^0.9*
 ```` shell
 python main.py epochs=100 batch_size=7 val_batch_size=3 num_workers=4 lr=0.001 wd=0.0001 momentum=0.8 
 python main.py lr_scheduler=poly_epoch optimizre=sgd
  ```` 
### Model
Currently, the following models are supported, and they can be selected as shown below. By default hrnet is used.
- **hrnet**: [High-Resolution Network (HRNet)](https://arxiv.org/pdf/1904.04514.pdf). Segmentation model with a single output.
- **hrnet_ocr**: [Object-Contextual Representations (OCR)](https://arxiv.org/pdf/1909.11065.pdf). 
A HRNet backbone with an OCR head. 
The model has two outputs, a primary and an auxiliary one.
- **hrnet_ocr_aspp**: Additionally including an ASPP module into the ORC model. Again the model has two outputs.
- **hrnet_ocr_ms**: [Hierarchical Multiscale Attention Network](https://arxiv.org/pdf/2005.10821.pdf). Extends ORC with multiscale and attention. 
The model has 4 outputs: primary, auxiliary, high_scale_prediction, low_scale_prediction
```shell
python main.py model=hrnet
python main.py model=hrnet_ocr
python main.py model=hrnet_ocr_aspp
python main.py model=hrnet_ocr_ms
```
Besides the selection of the models other parameters are provided and can be enabled/disabled as shown below. By default both are True
- **MODEL.PRETRAINED**: Indicate if pretrained weights (on ImageNet) should be used.
- **MODEL.INIT_WEIGHTS**: Indicate if weights should be Initialized from a normal distribution.

````shell 
python main.py MODEL.PRETRAINED=false MODEL.INIT_WEIGHTS=false
````

### Dataset

Currently, the following datasets are supported, and they can be selected as shown below. By default, the cityscapes dataset is used.
- **Cityscapes**: [Cityscapes dataset](https://www.cityscapes-dataset.com/) with using fine annotated images. Contains 19 classes and 2.975 training and 500 validation images.
- **Cityscapes_coarse**: [Cityscapes dataset](https://www.cityscapes-dataset.com/) with using coarse annotated training images. Contains 19 classes and ~20.000 training images. 
For validation the 500 fine annotated images from Cityscape are used.
- **Cityscapes_fine_coarse**: [Cityscapes dataset](https://www.cityscapes-dataset.com/) with using coarse and fine annotated training images. Contains 19 classes and ~23.000 training images. 
For validation the 500 fine annotated images from Cityscape are used.
- **VOC2010_Context**: [PASCAL Context](https://cs.stanford.edu/~roozbeh/pascal-context/) dataset, which is an extension for the [PASCAL VOC2010 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/) and contains additional segmentation masks. 
Currently, only the 60 class setting is supported. 
It contains 5.105 training and 4.998 validation images.
```shell
python main.py dataset=Cityscapes
python main.py dataset=Cityscapes_coarse
python main.py dataset=Cityscapes_fine_coarse
python main.py dataset=VOC2010_Context
```

### Lossfunction

There are two parameters to define the functionality of the lossfunction. 
The *lossfunction* parameter is used to define one or multiple lossfunctions.
The *lossweights* parameter is used to weight the different losssfunctions.
Both are explained in more detail in the following and can be overwritten from the commandline as shown below:
 - **lossfunction:** defines the lossfunction to be used and can be set by: ``lossfunction="CE"`` for using Cross Entropy Loss.
If the model has multiple output a list of lossfunctions can be passed, where the order inside the list corresponds to the order of the models outputs.
For example: ``lossfunction=["RMI","CE"]`` if the RMI loss should be used for the primary model output and Cross Entropy for the secondary output. 
The following losses are supported and can be selected by using the corresponding name/abbreviation:
   - **CE**: [Cross Entropy Loss](https://pytorch.org/docs/1.9.1/generated/torch.nn.CrossEntropyLoss.html)
   - **wCE**: CE with using weighting of classes
   - **RMI**: [Region Mutual Information Loss for Semantic Segmentation](https://arxiv.org/pdf/1910.12037.pdf)
   - **wRMI**: slightly adopted RMI with using weighting of classes
   - **DC**: Dice Loss
   - **DC_CE**: Combination of Dice and Cross Entropy Loss
   - **TOPK**: TOPK loss
   - **TOPK_CE**: Combination of TOPK and Cross Entropy Loss
 - **lossweight**: For multiple losses it may be usefull to weight the losses differently.
Therefore pass a list of weights where the length correspond to the number of losses/model outputs. 
For two outputs this can be done in the following way: ``lossweight=[1, 0.4]`` to weight the primary loss by 1 while the second output is weighted less with 0.4.
If not specified no weighting is done.
By default ``lossweight=[1, 0.4, 0.05, 0.05]`` is used.
```shell
python main.py lossfunction=wCE lossweight=1                    #For one output like for HRNet
python main.py lossfunction=[RMI, CE] lossweight=[1,0.4]          #Two outputs like OCR and OCR+ASPP
python main.py lossfunction=[wRMI, wCE, wCE, wCE] lossweight=[1, 0.5, 0.1, 0.05]  #Four outputs like OCR+MS
```
Consider the number of outputs of each model for **defining the correct number of losses in the right order**. 
If the number of given lossfunctions/lossweights is higher than the number of model outputs that's no problem and only the first corresponding lossfunction/lossweight is used.
For the supported models the number of outputs looks like this:
- hrnet:  1 output
- hrnet_ocr: 2 outputs *[primary, auxiliary]*
- hrnet_ocr_aspp: 2 outputs *[primary, auxiliary]*
- hrnet_ocr_ms: 4 outputs *[primary, auxiliary, high_scale_prediction, low_scale_prediction]*


### Pytorch Lightning Trainer
Since Pytorch Lightning is used as training framework, with the trainer class as central unit, 
there is also the possibility to give arguments to the trainer from the config.
The *pl_trainer* entry in the baseline.yaml is used for this purpose.
By default this looks like the following and arguments can be overwritten/added/removed as shown below:
```` yaml
baseline.yaml
------------------
pl_trainer:
  precision: 16
  sync_batchnorm: True
  benchmark: True
````
````shell
#Overwriting
python main.py pl_trainer.precision=32 pl_trainer.benchmark=false
#Adding
python main.py +fast_dev_run=True +pl_trainer.reload_dataloaders_every_n_epochs=2 
#Removing
python main.py ~pl_trainer.precision 
````
A full list of all available options of the Pytorch Lightning Trainer class can be seen in the [Lightning docs](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api). \
Some arguments are defined inside the code and can't be overwritten from the config file. 
These parameters are not intended to be changed, if you still want to adapt them you can do this in *main.py* at the bottom of the *training_loop* function.
The effected parameters are: 
- *max_epochs*: are set inside the config by *epochs*
- *gpus*: all available gpus are used
- *callbacks*: callbacks are defined from the config in *config/callbacks*
- *logger*: tb_logger is used by default
- *strategy*: ddp if multiple gpus are used, else None

### Environment

If you run code on different devices (e.g. on your local machine and a gpu-cluster) it can make sense to group all environment specific settings, e.g. paths or hyperparameters like the batch size, to enable easy switching between them. 
Different environments are stored in the *conifg/environment/* folder and can be used in the following way. 
To add you own environment look at the customization chapter. By default ``environment=local``.


````shell
python main.py environment=cluster
python main.py environment=local
````

### Data Augmentations
####################
MISSING
####################

