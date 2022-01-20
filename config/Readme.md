#Walktrough the config jungle

In this repository [Hydra](https://hydra.cc/) is used for configuring and managing experiments.
Therefore configuration files are of major importance which is why they get explained in more detail in the following.
First the basic functionally of Hydra is explained shortly, afterwards the options how to configure the experiments which are provided in this repository are explained.



## Basics
- Command line Syntax
- Default List
- References/Overrides


Hydra automatically loads and composes the different configuration files and allows to dynamically overriding values at runtime via the command line.
The commandline syntax of Hydra is straight forward and elements can be changed, added or removed in the following way:
````shell
python main.py  parameter_to_change=<new_value>  +parameter_to_add=<a_value>  ~parameter_to_delete
#Example
python main.py  lr=0.1  +momentum=0.9  ~prunung   
````
Hydra uses *.yaml* files to specify a configuration. 
In this repository the *baseline.yaml* can be seen as the main file and from there further configs are composed.
Each model(also dataset, environment, etc) has an own config file which contain model specific parameters
The subfolder *models/* contains the configurations for different models, the subfolder *datasets/* contains the configurations for different datasets and so on for the remaining subfolders where the names speak for themselves.
Hydra composes the final config from the default list(in *baseline.yaml*) which looks like this:

### 
````yaml
baseline.yaml
_____________________
defaults:
  - _self_
  - callbacks: default
  - data_augmentations: default
  - models: hrnet
  - datasets: Cityscape_19
  - environment: local
````
One can change the used model by passing:```python main.py models=hrn_ocr```(and analogues for the remaining the other ones).
This tells Hydra to load the *hrn_ocr.yaml* config from the *models/* folder.


The need for different configuration files arises from the fact that different models and datasets require different parameters and only those relevant to the current experiment should be loaded.
Therefore, each model or dataset has its own config and Hydra then merges the configs corresponding to the experiment.

The default list instructs Hydra how to build the output config

Hydra is used for managing the config files. So maybe make youself familar with Hydra before reading further.
If you want to adopt thinks you should condiser the order of hydras default list which looks like the following:

later packages will overwrite the previous ones. For example in the baseline.yaml a batch_size=6 is defined. 
If you set batch_size=3 in the enviroment package(local.yaml) this one will be used for training,
This gives you the freedom to define a baseline(baseline.yaml) and change relevant thinks which are dataset, model or environment specific.


The different packages are used for defining different thinks. 
Whereby the names speak for themselves.
The most defauls parameters are defined in baseline.yaml. Callbacks and data_augmentations are defined in the corresponding packages.
Model parameters are specified in models and analogues datasert parameters are specified in datasets
The environment package is used for changing environment dependent parameters to make it easier to switch between differen enviroments like a local machine and a gpu-cluster.

## Configure the Configuration
### Models
Currently, the following models are supported, and they can be selected as shown below. By default hrnet is used.
- **hrnet**: [High-Resolution Network (HRNet)](https://arxiv.org/pdf/1904.04514.pdf). Segmentation model with a single output.
- **hrnet_ocr**: [Object-Contextual Representations (OCR)](https://arxiv.org/pdf/1909.11065.pdf). 
A HRNet backbone with an OCR head. 
The model has two outputs, a primary and an auxiliary one.
- **hrnet_ocr_aspp**: Additionally including an ASPP module into the ORC model. Again the model has two outputs.
- **hrnet_ocr_ms**: [Hierarchical Multiscale Attention Network](https://arxiv.org/pdf/2005.10821.pdf). Extends ORC with multiscale and attention. 
The model has 4 outputs: primary, auxiliary, high_scale_prediction, low_scale_prediction
```
python main.py model=hrnet
python main.py model=hrnet_ocr
python main.py model=hrnet_ocr_aspp
python main.py model=hrnet_ocr_ms
```
Besides the selection of the models other parameters are provided and can be enabled/disabled as shown below. By default both are True
- **MODEL.PRETRAINED**: Indicate if pretrained weights (on ImageNet) should be used.
- **MODEL.INIT_WEIGHTS**: Indicate if weights should be Initialized from a normal distribution.

```python main.py MODEL.PRETRAINED=false MODEL.INIT_WEIGHTS=false```


### Datasets

Currently, the following datasets are supported, and they can be selected as shown below. By default, the cityscapes dataset is used.
- **Cityscapes**: [Cityscapes dataset](https://www.cityscapes-dataset.com/) with fine annotated images. Contains 19 classes and 2.975 training and 500 validation images.
- **Cityscapes_coarse**: [Cityscapes dataset](https://www.cityscapes-dataset.com/) with coarse annotated images. Contains 19 classes and ~20.000 training images. 
For validation the 500 fine annotated images form Cityscape are used.
- **VOC2010_Context**: [PASCAL Context](https://cs.stanford.edu/~roozbeh/pascal-context/) dataset, which is an extension for the [PASCAL VOC2010 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/) and contains additional segmentation masks. 
Currently, only the 60 class setting is supported. 
It contains 5.105 training and 4.998 validation images.
```
python main.py dataset=Cityscapes
python main.py dataset=Cityscapes_coarse
python main.py dataset=VOC2010_Context
```

### Lossfunction

There are two parameters to define the functionality of the lossfunction. 
The first one is lossfunction wich defines the used lossfunction and lossweights which weight the different losses.

   
   Consider the number of outputs of each model for defining the correct number of losses: 

   **Number of Outputs per model:** hrnet: 1 -- hrnet_ocr: 2 -- hrnet_ocr_aspp: 2 -- hrnet_ocr_ms: 4

 - **lossfunction:** defines the lossfunction to be used and can be set by: ``lossfunction="CE"`` for using Cross Entropy Loss.
If the model has multiple output a list of lossfunctions can be passed, where the order inside the list corresponds to the order of the models outputs.
For example: ``lossfunction=["RMI","CE"]`` if the RMI loss should be used for the primary model output and Cross Entropy for the secondary output. 
The following losses are supported and can be selected by using the corresponding name/abbreviation:
   - "CE": Cross Entropy Loss
   - "wCE": Cross Entropy Loss with weighting classes
   - "RMI": ...
   - "wRMI": ...
   - "DC": ...
   - "DC_CE": ..
   - "TOPK": ...
   - "TOPK_CE": ..
 - **lossweight**: For multiple losses it may be usefull to weight the losses differently.
Therefore pass a list of weights where the length correspond to the number of losses/model outputs. 
By default: ``lossweight=[1, 0.4, 0.05, 0.05]``

### Basic Training parameters
The following training parameters can be changed in the *baseline.yaml* directly or can be overwriten from the command line in the following way: 
`` python main.py parameter = new_value`` 
 - **epochs:** number of epochs for training.
 - **batch_size:** defines the batch size during training. 
 - **val_batch_size:** defines the batch size during validation and testing. Is set to batch_size if not specified.
 - **num_workers:** number of workers for the dataloaders.
 - **lr:** initial learning rate for training.
 - **wd:** weight decay (optimizer parameter)
 - **momentum:** momentum (optimizer parameter)
For two outputs this can be done in the following way: ``lossweight=[1, 0.4]`` to weight the primary loss by 1 while the second output is weighted less with 0.4.
If not specified no weighting is done.
 - **optimizer**: defines the optimizer to use. Currently only sgd is supported (default: ``optimizer="sgd"``)
   - "sgd": stochastic gradient Descent
 - **lr_scheduler**: defines the lr scheduler to use. Set ``lr_scheduler="poly"`` by default
   - "poly": Polynomial lr scheduler over number of steps
   - "poly_epoch": Polynomial lr scheduler over number of epochs

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
There is also an environment option. If you debug your code on your local machine but want to train it on another server or cluster you have some environment specific parameters.
For example you environments will have different paths or you want a smaller batch_size on you local machine you can set this thinks dependent on which environment you are using.
For doing thinks like this you can define different enviromnets in the *environment/* folder and use them in the following way:
``python main.py environment=cluster``

# Hydra in an Nutshell
Some additional sources which have more information and helped me at the beginning are
[here](https://github.com/lkhphuc/lightning-hydra-template),
[here](https://www.sscardapane.it/tutorials/hydra-tutorial/) and
[here](https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b).

The Default list

Overwriting




## OmegaConf in a Nutshell
Hydra uses the package [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/) to handle *.yaml* files. 
A introudcution to yaml is give below.
OnegaConf gives a lot of possibilities to work with yaml files, but since hydra manages this for you in the background you do not need much of it for a basic use.
If you need further functionality, for example if you manually want to load or save files look 
at the official [OmegaConf docs](https://omegaconf.readthedocs.io/en/2.1_branch/).
The **Access and Manipulation** of the cfg in python is straight forward:
````yaml
example.yaml
----------------
Parameters:
  lr: 0.01
  epochs: 100
  whatever: 
  - 42
  - ...
````
````python3
main.py
----------------
import OmegaConf
...
#for the example manually load the cfg, normally done by hydra automatically
cfg = OmegaConf.load("example.yaml") 

#acess over object and dictionary style
lr=cfg.Parameters.lr
lr=cfg["Parameters"]["lr"]

#Manipulation in the same way
cfg.Parameters.epochs = 300
cfg["Parameters"]["epochs"] = 300

##same goes for accessing lists
x=cfg.Parameters.whatever[0]
````

## YAML in a Nutshell

This is only a short introduction to YAML and only shows its basic syntax. This should be enough for defining you own yaml files but if you need more informations they can be found [here](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html) for example.

Some  **Basic Assignments** are shown here:
````yaml
example.yaml
-----------
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
