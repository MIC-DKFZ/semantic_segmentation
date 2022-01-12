#Walktrough the config jungle

Hydra is used to ... therefore the config files are of main importance.

Hydra is used for managing the config files. So maybe make youself familar with Hydra before reading further.
If you want to adopt thinks you should condiser the order of hydras default list which looks like the following:
````yaml
baseline.yaml
------------------------
defaults:
  - _self_
  - callbacks: default
  - data_augmentations: default
  - models: hrnet
  - datasets: Cityscape_19
  - environment: local
````
later packages will overwrite the previous ones. For example in the baseline.yaml a batch_size=6 is defined. 
If you set batch_size=3 in the enviroment package(local.yaml) this one will be used for training,
This gives you the freedom to define a baseline(baseline.yaml) and change relevant thinks which are dataset, model or environment specific.


The different packages are used for defining different thinks. 
Whereby the names speak for themselves.
The most defauls parameters are defined in baseline.yaml. Callbacks and data_augmentations are defined in the corresponding packages.
Model parameters are specified in models and analogues datasert parameters are specified in datasets
The environment package is used for changing environment dependent parameters to make it easier to switch between differen enviroments like a local machine and a gpu-cluster.


### Models
you can switch between each of the supported models by:
```python main.py models=hrn_ocr```. In this case the hrn_ocr model is used. Additional parameters which are *MODEL.PRETRAINED* and *MODEL.INIT_WEIGHTS* which are set to true by default.
They can be disabled by```python main.py models=MODEL.PRETRAINED=false MODEL.INIT_WEIGHTS=false``` if you want to train from scratch.
The currently supported models are: *hrnet*, *hrnet_ocr*, *hrnet_ocr_aspp*, *hrnet_ocr_ms*

### Datasets
changing between different datasets works in the same way as for models by:
```python main.py models=Cityscape_Coarse```. The currently supported datasets are *Cityscape* and *Cityscape_Coarse*

### Basic Training parameters
the following training parameters can be changed in the *baseline.yaml* directly or can be overwriten from the command line in the following way: 
`` python main.py parameter = new_value`` 
 - **epochs:** number of epochs for training
 - **batch_size:** defines the batch size during training. 
 - **val_batch_size:** defines the batch size during validation and testing. By default it is set to the same value as batch_size, so just ignore this if you want the same batch size for training and validation.
 - **num_workers:** number of workers for the dataloaders
 - **lr:** initial learning rate for training
 - **wd:** weight decay (optimizer parameter)
 - **momentum:** momentum (optimizer parameter)
 - **lossfunction:** defines the lossfunction which should be used. Give the name of the lossfunction like: ``lossfunction="CE"`` for using Cross Entropy Loss.
If you have multiple model outputs, pass a list of lossfunctions, where the order inside the list corresponds to the order of the models outputs. 
For example: ``lossfunction=["RMI","CE"]`` if you want to use RMI loss for the primary model output and Cross Entropy for the secondary output. 
Use the following name/abbreviation for getting the corresponding loss.
   - "CE": Cross Entropy Loss
   - "wCE": Cross Entropy Loss with weighting classes
   - "RMI": ...
   - "wRMI": ...
   - "DC": ...
   - "DC_CE": ..
   - "TOPK": ...
   - "TOPK_CE": ..
 - **lossweight**: if you use multiple model outputs you probably want to weight them differently. 
For two outputs this can be done in the following way: ``lossweight=[1, 0.4]``. In this case the primary loss is weighted by 1 while the second output is weighted less with 0.4.
If not specified no weighting is done.
 - **optimizer**: defines the optimizer to use. Currently only sgd is supported (default: ``optimizer="sgd"``)
   - "sgd": stochastic gradient Descent
 - **lr_scheduler**: defines the lr scheduler to use. Set ``lr_scheduler="poly"`` by default
   - "poly": Polynomial lr scheduler over number of steps
   - "poly_epoch": Polynomial lr scheduler over number of epochs

### Pytorch Lightning Trainer
Since Pytorch Lightning is used as training framework, with the trainer class as central unit, 
there has to be the possibility to give arguments to the trainer from the config.
The *pl_trainer* entry in the baseline.yaml is used for this purpose.
By default this looks like this
```` yaml
baseline.yaml
------------------
pl_trainer:
  precision: 16
  sync_batchnorm: True
  benchmark: True
````
As you can see some entries are already in use while a lot of other options are not used.
Look at the Pytorch Lightning Trainer class for all available options ([here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api)).
Overwriting a already set parameter is done in the following way:
````
python main.py pl_trainer.precision=32 pl_trainer.benchmark=false
````
For adding a new parameter is done in the same way, just add and *+* in front of the argument like this:
````
python main.py +fast_dev_run +pl_trainer.reload_dataloaders_every_n_epochs=2 
````
Some argument are defined inside the code and can't be overwritten from the config file. 
These parameters are not intedet to be changed, if you still want to adapt them you can do this in *main.py* at the bottom of the *training_loop* function.
The effected parameters are: 
- *max_epochs*: are set inside the config
- *gpus*: all available gpus are used
- *callbacks*: callbacks are defined from the config
- *logger*: tb_logger is used by default
- strategy: ddp if multiple gpus are used, else None

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




## OmegaConf
Hydra uses the package [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/) to handle .yaml files. A introudcution to yaml is give below.
OnegaConf gives a lot of possibilities to work with yaml files, but since hydra manages this for you in the backround you do not need much of it for a basic use.
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
#for the example manually load the cfg, normaly done by haydra automatically
cfg = OmegaConf.load("example.yaml") 

#Acess over object and dictionary style
lr=cfg.Parameters.lr
lr=cfg["Parameters"]["lr"]

#Manipulation in the same way
cfg.Parameters.epochs = 300
cfg["Parameters"]["epochs"] = 300

##same goes for acessing lists
answer=cfg.Parameters.whatever[0]
````

## YAML in an Nutshell
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
