<p align="left">
  <img src="../imgs/Logos/HI_Title.png" >
</p>

# Walkthrough of the Config Jungle

In this repository [Hydra](https://hydra.cc/) is used for configuring and managing experiments.
Therefore, configuration files and their handling are of major importance, which is why they are
explained in more detail below.
First, the basic functionality of Hydra will be briefly explained.
At first glance, the use of Hydra may make the configuration more complicated and confusing, but
this will quickly disappear if you familiarize yourself with it a bit.
The advantage that Hydra provides is the ease of managing experiments and to easily add new models
or datasets (and more) without changing the base code.
Since Hydra uses the [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/) package to
handle *.yaml* files, Omegaconf and YAML are also briefly introduced.
## Hydra Basics

<details><summary>Click to expand/collapse</summary>
<p>

[Hydra](https://hydra.cc/) automatically loads and composes different configuration files and allows
to dynamically override values at runtime via the command line.
In Hydra, *.yaml* files are used to set configurations.
In this repository the *config/training.yaml* can be seen as the main file from which other
configurations are composed (for training).
Each subfolder in *config/* is
a [config group](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/), which
contains a separate config file for each alternative inside.
For example the config group *model* is located in the *config/model* subfolder with a separate *
.yaml* file for each available model (hrnet.yaml, hrnet_ocr.yaml, ...).
The individual config files contain model/dataset/etc. specific parameters, such as the number of
channels in a layer of the model or the number of classes in a dataset.
Having a separate config file for each model/dataset/etc. makes it easy to switch between them and
arbitrarily combine different config files from different config groups.
Additionally, this ensures that only the relevant parameters are loaded into the job configuration.
Hydra creates the [job configuration](https://hydra.cc/docs/1.0/configure_hydra/job/) by composing
the configuration files from the different configuration groups.
Basically, exactly one config file from each config group is used in this process
(as an exception, a config group can be declared as optional, this will then only be used if it is
explicitly defined).
To tell hydra how to compose the job configuration,
a [default list](https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/) is used, which
specifies which configuration file from which configuration group should be used and in which order
they are composed.
The default list in this repository is defined in *config/training.yaml* and looks like this:

````yaml
training.yaml
  ─────────────────────────────
defaults:
  - _self_
  - trainer: SemSeg             # Which Trainer to use
  - metric: mean_IoU            # Metric configuration
  - model: hrnet                # Model
  - dataset: Cityscapes         # Dataset
  - data_augmentation: only_norm  # Data Augmentation
  - optimizer: SGD              # Optimizer
  - lr_scheduler: polynomial    # Learning rate scheduler
  - callbacks: default          # Callbacks
  - logger: tensorboard         # Logger
  - experiment/default          # Load Default Setting and Hyperparameters
  - optional experiment:        # (Optional) load another experiment configuratioj
  - environment: local          # Environment
````

The configs of each config group are merged from top to bottom, where later groups can overwrite the
parameters of earlier groups.
In addition to the order, the default list also sets default values for the configuration groups.
This means if not changed, the parameters defined in *experiment/default.yaml*,..., *datasets/Cityscapes.yaml*
and *model/hrnet.yaml* are used in this case.
To change the used config file of a config group, the corresponding entry in the default list can be
changed in the *training.yaml*, or the entry can be overwritten from the commandline.
Hydra's [commandline syntax](https://hydra.cc/docs/advanced/override_grammar/basic/#working-with-your-shell)
is straight forward and elements can be changed, added or removed in the following ways.
Thereby this syntax is the same for single parameters like *batch_size* as well as for config files
from config groups like *model*.
All available options to change for parameters and config groups are shown below in
the [Configure the Configuration](#configure-the-configuration) part.

````shell
python training.py  parameter_to_change=<new_value>  +parameter_to_add=<a_value>  ~parameter_to_delete
#Example for single parameters
python training.py  batch_size=3 +extra_lr=0.001 ~momentum
#Example for config groups
python training.py  model=hrnet_ocr +parameter_group=default ~environment   
````

Another important concept of Hydra is the ability
to [instantiate objects](https://hydra.cc/docs/advanced/instantiate_objects/overview/).
This enables to fully define classes in the config files and then instantiate them in the code.
An example for both is shown below.
The reason for doing this is that it is possible to add new optimizers, models, datasets etc. from
the config without having to change the base code.
This makes this repository easy to change and flexible to extend without having to search yourself
through the implementation.
For example, to use or define another optimizer in the example below, only the optimizer entry in
the *example.yaml* has to be changed.

````yaml
example.yaml
  ─────────────────────────────
# Generall syntax
name:
  _target_: path.to.class
  arg1:     some_argument
  arg2:     ...
# Example for defining a torch-optimizer
optimizer:
  _target_:     torch.optim.SGD
  lr:           0.01
  momentum:     0.9
  weight_decay: 0.005
# Another example for defining a custom class object
metric:
  _target_:    src.metric.ConfusionMatrix
  num_classes: 24
````

````python3
example.py
─────────────────────────────
my_optimizer = hydra.utils.instantiate(cfg.optimizer)
my_metric = hydra.utils.instantiate(cfg.metric)
````

This was only a short introduction how to use hydra to work with this repository.
For more information on Hydra, check out the official docs or one of the following sources, which
provide some nice insights into Hydra
([source1](https://github.com/lkhphuc/lightning-hydra-template),
[source2](https://www.sscardapane.it/tutorials/hydra-tutorial/),
[source3](https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b)
and
[source4](https://github.com/ashleve/lightning-hydra-template)).

</p>
</details>

## OmegaConf in a Nutshell

<details><summary>Click to expand/collapse</summary>
<p>

Hydra uses the package [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/) to handle *
.yaml* files.
OnegaConf gives a lot of possibilities to work with *.yaml* files, but since hydra manages this for
you in the background you do not need much of it for a basic use.
If you need further functionality, for example if you manually want to load or save files look
at the official [OmegaConf docs](https://omegaconf.readthedocs.io/en/2.1_branch/).
The [**Access and
Manipulation**](https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-manipulation) of
the cfg in python is straight forward:

````yaml
example.yaml
─────────────────────────────
Parameters:
  lr:     0.01
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
# For the example load the cfg manually, which is normally done automatically by hydra
cfg = OmegaConf.load("example.yaml")

# Access over object and dictionary style
lr = cfg.Parameters.lr
lr = cfg["Parameters"]["lr"]

# Manipulation in the same way
cfg.Parameters.epochs = 300
cfg["Parameters"]["epochs"] = 300

# The same goes for accessing lists
x = cfg.Parameters.whatever[0]
````

[**Variable
interpolation**](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) is
another important concept of Hydra and Omegaconf.
When defining config files the situation will occur that variables from other config files are
needed.
For example for defining the last layer of a model, the number of classes, which is defined in the
specific dataset configs, may be needed.
Therefore, variable interpolation is used, which can be seen as a link to a position in the config,
that is resolved at runtime.
Therefore, the variable is resolved from the dataset which used the current job and no conflicts
occur between different dataset configs and the model config.
Variable interpolation is done with the following syntax:``${path.to.another.node.in.the.config}``.
and in that case the value will be the value of that node.

````yaml
dataset/a_dataset.yaml
─────────────────────────────
  #@package _global_
...
dataset:
  num_classes: 24
````

````yaml
model/a_model.yaml
─────────────────────────────
  #@package _global_
...
num_output_classes: ${dataset.number_classes} # num_output_classes will have the value 24 at runtime
````

</p>
</details>

## YAML in a Nutshell

<details><summary>Click to expand/collapse</summary>
<p>

This is only a short introduction to YAML and only shows its basic syntax. This should be enough for
defining your own yaml files, but if you need more information they can be
found [here](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html) for
example.
The following examples are for Yaml in combination with OmegaConf and may not work for yaml alone.

Some  **Basic Assignments** are shown here:

````yaml
example.yaml
─────────────────────────────
# Comments in yaml
number: 10                   # Simple value, works for int and float.
string: Text|"Text"          # Strings, Quotation marks are not necessarily required.
empty: None| |Empty|Null
explicit_Type: !!float 1     # Explicitly defined type. works as well for other types like str etc.
missing_vale: ???            # Missing required value. The  has to be given e.g. from the commandline.
optional opt_value:          # Optional Value. Can be empty or ???, and will only be considered if it has a value.
value2: ${number}            # Value interpolation (takes the value of attribute number, in this 
                             # case 10). $ indicates reference and {} is required.
value3: "myvalue ${number}"  # String interpolation, same as value interpolation just with string output.
booleans: on|off|yes|no|true|false|True|False|TRUE|FALSE    #multiple possibilities to define boolean values.
````

**List** are defined in the following way:

````yaml
alist:
  - elem1                      # Elements need to be on the same indentation level
  - elem2                      # There needs to be a space between dash and element
  - ...
samelist: [ elem1, elem2, ... ]               # The same list can also be defined with this syntax

val_interpolation: ${alist[0]}                # Get the value of alist at position 0
````

**Dictionaries** are defined in the following way:

````yaml
adict:
  key1: val1                    # Keys must be indented
  key2: val2                    # There has to be a space between colon and value
  ...                           # Each key may occur at most once
samedict: { key1: val1, key2: val2, ... }     # The same dict can also be defined with this syntax

val_interpolation: ${adict.key1}              # Get the value of adict at key1
````

For more complex files you will end up with lists of dictionaries and dictionaries of list and
mixtures of both. But basically that's it!

</p>
</details>

# Configure the Configuration

In the following, each configuration group and some other features are explained, for more details and how to customize take a look in the readme in each config group folder.


- **dataset**: Define the folder structure and file format. (note: this is not a torch dataset - torch datasets are defined in the training_scheme group.)
- **img_loader**: How to load the image files.
- **label_handler**: How to load the label files, but also how to apply transforms or visualize the data.
- **training_scheme**: Corresponds to a pytorch dataset class. Which and how images and labels are loaded and provided for training.
- **model**: Pytorch Model
- **trainer**: Defines the training loop
- **metric**: How to validate the models performance compared to the GT.
- **optimizer**: torch optimizer
- **lr_scheduler**: learning rate scheduler
- **augmentation**: Albumentations augmentation pipeline
- **experiment**: To compose config groups and overwrite parameters to plan experiments
- **environment**: To enable the use of different devices, adopt path and setting for individual devices.



## Experiments, Hyperparameters and Pytorch Lightning Trainer

How to deal with parameters which are not part of any config group:

<details><summary>Configure</summary>
<p>

#### Experiments

Individual data sets, models, etc. are defined in the corresponding parameter groups, the experiment files are used to define their combination. This avoids the need for manual coniguration via the commandline for frequently performed experiments.
Therefore, the standard list is overwritten as follows:

````yaml
# @package _global_
#define dataset and model
defaults:
  - override /dataset: Cityscapes
  - override /model: hrnet
````

#### Hyperparameters


The default hyperparameters are defined in *config/experiment/default.yaml*. 
For the specific datasets they are overwriten from *config/experiment/<dataset.name>.yaml*
The following hyperparameters are supported and can be changed in the *.yaml*-files directly or
can be overwritten from the command line as shown below.

- **epochs:** number of epochs for training.
- **batch_size:** defines the batch size during training (per GPU).
- **val_batch_size:** defines the batch size during validation and testing (also per GPU). Is set to
  batch_size if not specified.
- **num_workers:** number of workers for the dataloaders.
- **lr:** initial learning rate for training.

 ```` shell
 python training.py epochs=100 batch_size=7 val_batch_size=3 num_workers=4 lr=0.001
  ```` 

#### Pytorch Lightning Trainer

Since Pytorch Lightning is used as training framework, with the trainer class as central unit,
some additional parameters can be defined by passing them to the Pytorch Lightning Trainer.
The *pl_trainer* entry in the config is used for this purpose.
By default, this looks like the following and arguments can be overwritten/added/removed as shown
below:

```` yaml
experiment/default.yaml
------------------
pl_trainer:                     # parameters for the pytorch lightning trainer
  max_epochs: ${epochs}         # parsing the number of epochs which is defined as a hyperparameter
  gpus: -1                      # defining the used GPUs - in this case using all available GPUs
  precision: 16                 # using Mixed Precision
  benchmark: True               # using benchmark for faster training
````

````shell
# Overwriting
python training.py pl_trainer.precision=32 pl_trainer.benchmark=false
# Adding
python training.py +fast_dev_run=True +pl_trainer.reload_dataloaders_every_n_epochs=2 
# Removing
python training.py ~pl_trainer.precision 
````

A full list of all available options of the Pytorch Lightning Trainer class can be seen in
the [Lightning docs](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api). \
Some arguments are defined inside the code and can't be overwritten from the config file.
These parameters are not intended to be changed, if you still want to adapt them you can do this
in *training.py* in the *training_loop* function.
The effected parameters are:

- *callbacks*: callbacks are defined in *config/callbacks*, so add your callbacks there
- *logger*: tb_logger is used by default
- *strategy*: ddp if multiple gpus are used, else None
- *sync_batchnorm*: sync_batchnorm is True if multiple gpus are used, else False

</p>
</details>

<details><summary>Customize</summary>
<p>

Experiment Configurations and Hyperparameters can be added or changed in *.yaml*-files or from the commandline.
For different experiments, a group of parameters may need to be adjusted at once.
To not have to change them manually each time there is an optional *experiment* config group to
easily switch between different experiment settings.
Create *experiment/my_config.yaml* and insert all parameters or settings that differ from the default.yaml
into it.
A dummy and how this can be used it shown below:

````yaml
config/experiment/my_config.yaml
─────────────────────────────
# @package _global_
defaults:
  - override /data_augmentation: randaugment_hflip
    
batch_size: 6
val_batch_size: 4
epochs: 175
lossfunction: RMI
...
# Also fo Pytorch Lightning Trainer Arguments
pl_trainer:
  precision: 32
  ...               
````

````shell
python training.py experiment=my_config
````

</p>
</details>


## Loss Function

Lossfunctions need to be defined outside of hydra:

<details><summary>Configure</summary>
<p>

There are two parameters to define the functionality of the loss function.
The *lossfunction* parameter is used to define one or multiple loss functions.
The *lossweights* parameter is used to weight the different losss functions.
Both are explained in more detail in the following and can be overwritten from the commandline as
shown below:

- **lossfunction:** defines the loss function to be used and can be set by: ``lossfunction="CE"``
  for using Cross Entropy Loss.
  If the model has multiple outputs a list of loss functions can be passed, where the order inside
  the list corresponds to the order of the model outputs.
  For example: ``lossfunction=["RMI","CE"]`` if the RMI loss should be used for the primary model
  output and Cross Entropy for the secondary output.
  The following losses are supported and can be selected by using the corresponding
  name/abbreviation:
    - **CE**: [Cross Entropy Loss](https://pytorch.org/docs/1.9.1/generated/torch.nn.CrossEntropyLoss.html)
    - **wCE**: CE with using weighting of classes
    - **RMI**: [Region Mutual Information Loss for Semantic Segmentation](https://arxiv.org/pdf/1910.12037.pdf)
    - **wRMI**: slightly adopted RMI with using weighting of classes
    - **DC**: Dice Loss
    - **DC_CE**: Combination of Dice and Cross Entropy Loss
    - **TOPK**: TOPK loss
    - **TOPK_CE**: Combination of TOPK and Cross Entropy Loss
- **lossweight**: In the case of multiple losses, it may be useful to weight the losses differently.
  Therefore pass a list of weights where the length correspond to the number of losses/model
  outputs.
  For two outputs this can be done in the following way: ``lossweight=[1, 0.4]`` to weight the
  primary loss by 1 while the second output is weighted less with 0.4.
  If not specified no weighting is done.
  By default ``lossweight=[1, 0.4, 0.05, 0.05]`` is used.

```shell
python training.py lossfunction=wCE lossweight=1                    # For one output like for HRNet
python training.py lossfunction=[RMI, CE] lossweight=[1,0.4]        # Two outputs like OCR and OCR+ASPP
python training.py lossfunction=[wRMI, wCE, wCE, wCE] lossweight=[1, 0.5, 0.1, 0.05]  # Four outputs like OCR+MS
```

Consider the number of outputs of each model for **defining the correct number of losses in the right order**.
If the number of given loss functions/lossweights is higher than the number of model outputs that's
no problem and only the first corresponding lossfunctions/lossweights are used.
For the supported models the number of outputs is listed [here](../config#loss-function)

</p>
 </details>


<details><summary>Customize</summary>
<p>

The loss function in defined using the *get_loss_function_from_cfg* function in *utils/lossfunction*
.
Inside the function your have access to everything what is defined in the cfg.
To add a custom loss function just add the following onto the bottom of the function:

````py 
elif name_lf == "MYLOSS":
        ...                  #do whatever you need
        loss_function = MyLoss(...)
````

The loss function will be called in the following way:
````lossfunction(y_pred, y_gt) ````
with ````y_pred.shape = [batch_size, num_classes, height, width] ````
and ````y_gt.shape = [batch_size, height, width]````.
If you need the data in another format you can use for example *lambda functions* (look at the
definition of "DC_CE" loss in the get_loss_function_from_cfg).

</p>
</details>


# Acknowledgements

<p align="left">
  <img src="../imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="../imgs/Logos/DKFZ_Logo.png" width="500"> 
</p>

This Repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).
