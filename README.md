
# !!!IN PROGRESS!!!


# Requirements
## Dependecies
Install the needed packages by:
````shell
pip install -r requirements.txt
````
This Repo is build up on the following packages. Maybe you should make yourselve fimilar with their basic usage fist.
- **Pytorch Lightning:** 
- **Hydra:** development and planing tool
- **Albumentation:** Package used for dataaugmentation



# How To Run
### Datasetup
#### Cityscape
Download the Cityscape dataset from [here](https://www.cityscapes-dataset.com/login/). 
You have to create an account and afterward download: *leftImg8bit_trainvaltest.zip* (11GB)  and *gtFine_trainvaltest.zip* (241MB).
Unzip them and put them into a folder, the structure of the folder should now look like this:

````
Datasets/cityscapes
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
The cityscapes dataset contain 34 classes by default but only 19 of them are used in practices.
To avoid doing this convertion during training this is done in a preprocessing step.
To do this preprocessing run the following code with adjusting the datapath to the location which contains the *gtFine_trainvaltest* folder. 
This will create a new img for each datasamble with the converted class labeling which will be merged into the folder/data structure of the cityscapes dataset.
````
python creating_CS_19_labels.py home/.../Datasets/cityscapes
````
After downloading and setting up the datathe path in the config has to be adjusted.
Open the file *config/datasets/Cityscape_19.yaml* and seacrch for DATASET.ROOT adopt this path to the location of the folder where your *gtFine_trainvaltest* and *leftImg8bit_trainvaltest* are.
In the above Example set:
````yaml
config/datasets/Cityscape_19.yaml
─────────────────────────────────
DATASET:
  ROOT: home/.../Datasets/cityscapes
````

### Running Code
you can directly train a model by just executing:
```` 
python main.py
````
which woule train the baseline experiment which is to train HRNET on the Cityscape Dataset with default settings.
To adopt the basic config setting you can edit the */config/baseline.yaml* file directly or use the hydra commandline syntax. 
To train a ocr model on another dataset without using pretrained weights and with an adopted lr and number of epochs you can run the following:
```` 
python main.py models=hrnet_ocr datasets=Pascal lr=0.005 epochs=120 MODEL.PRETRAINED=false
````
Another example with adopting the lossfunction and editing the pytorch lightning trainer. 
In this case the main loss is changed as well as the precision parameter of the lightning trainer and accumulated gradiend batches are added to the trainer.
````
python main.py main_loss="RMI" pl_trainer.precision=32 +pl_trainer.accumulate_grad_batches=2
````

As you can see the basic syntax how to run the code is simple. 
The crucial thing is to know which parameters you can overwrite and how.
Therefore the *config/* folder explains in detail how the configuration is composed and which parameters it contains.
LINK

# Customizing

## Model
Defining a custom model is done in two steps, first defining your custom pytorch model and afterwards setting up its config file.
1. **Defining your Pytorch Model**, thereby the following thinks have to be considered:
   - put your *modelfile* into the *models/* folder
   - Your file has to contain a *get_seg_model* function which gets the config(cfg) and returns your model.
     In this fcuntion you load your model model and may intialize it with pretrained weight or do whatever you want. The function should look like this:
   ````py
   def get_seg_model(cfg):
        #you can get everthink you need from the config, e.g. the number of classes, like this:
        num_classes=cfg.DATASET.NUM_CLASSES
        ...
        model=MyModel(num_classes, ...)
        ...
        return model 
    ````
   - **Model Output**: Your model should **return a dict** which contain all the models outputs. The naming can be arbitrary.
   For example if you have one ourput return as follows: ``return {"out": model_prediction}``. If you have multiple output to it analogues:
``return {"main": model_prediction, "aux": aux_out}``.
It should be noted that the **order of the outputs is relevant**. Only the first output is used for updating the metric during validation.
Futher the order of the outputs should match the order of your losses in *lossfunction* and the weights in *lossweights*.(see *config/* for more details on that)
   
2. **Setting up your model config**
   - Create a *custom_model.yaml* file in *config/models/*. For the content of the *.yaml* file adopt the following dummy.
   ````yaml 
   #@package _global_
   #MODEL IS USED TO STORE INFORMATION WHICH ARE NEEDED FOR YOUR MODEL  
   MODEL:
      #REQUIRED MODEL ARGUMENTS
      NAME: MyModel            #Name of the file in models/ which contains you get_seg_model function
                               # In this case the get_seg_mode() funtion is in models/MyModel.py
      #YOUR ARGUMENTS, FOR EXAMPLE SOMETHINNK LIKE THAT
      PRETRAINED: True         # you could want a parameter to indicate if pretrained weights should be used or not 
      PRETRAINED_WEIGHTS: /pretrained/weights.pth  # give the path to the weights      
    ````
## Dataset
Defining a custom dataset is done in two steps, first defining your custom pytorch dataset and afterwards setting up its config file.
1. **Defining your pytorch Dataset**, therby consider that the following structure is required (mainly pytorch basic):
   - \__init__(self, custom_args, split, transforms):
     - *custom_args*: your custom input arguments (for example data_root etc.). They will be given to your dataset from the config file (see below).
     - *split*: one of the following strings: \["train","val","test"]. To define if train, validation or test set should be returned.
     - *transforms*: Albumentation transformations
   - \__getitem__(self, idx):
     - getting some index and should the output should look similat to: *return img, mask* 
     - with ````img.shape = [c, height, width]```` and ````mask.shape = [height, width]````, where *c* is the number of channels. For example *c=3* if you use RGB data.
   - \__len(self)__:
     - return the number of samples in your dataset, somehtink like: *return len(self.img_files)*
2. **Setting up your dataset config** 
   - Create a *custom_dataset.yaml* file in *config/datasets/*. For the content of the *.yaml* file adopt the following dummy:
   ````yaml 
   #@package _global_
   ### dataset is used to initialize your custom dataset, 
   ### _target_: should point to your dataset class
   ### afterwards you can handle your custom input arguments which are used to initialize the dataset
   dataset:
     _target_: datasets.MyDataset.dataset_class
     root: ${DATASET.DATA_ROOT}     #data root as example input (in this case a reference to what is defined in DATASET) 
     input1: ...                    #All your other input arguments
     input2: ...
   ### DATASET is used to store information about the dataset which are needed during training
   DATASET:
     ## REQUIRED DATASER ARGUMENTS
     NAME:            #Used for the logging directory
     NUM_CLASSES:     #Needed for defining the model and the metrics
     IGNORE_INDEX:    #Needed for the lossfunction, if no ignore indes set to -100 or another number which do no occur in your dataset
     ## CUSTOM INPUT ARGUMENTS, SOMETHINK LIKE:
     DATA_ROOT: /home/user/datasets/mydata  
     ## OPTIONAL, BUT NEEDED IF POLY LR SCHEDULER IS USED
     Size:
        TRAIN: 1234 # Size of your training dataset
     ## OPTIONAL, BUT NEEDED IF WEIGHTED LOSSFUNCTIONS ARE USED
     CLASS_WEIGHTS: [ 0.9, 1.1, ...]
     ##OPTIONAL, ONLY NEEDED FOR NICER LOGGING
     CLASS_LABELS:
        - class1
        - class2 ...
   ````
## Lossfunction
The lossfunction in defined using the *get_loss_function_from_cfg* function in *utils/lossfunction*.
Inside the the function your have acess to everthink what you defined inside your config using *cfg.myparameter*.
To add a custom lossfunction just add the following onto the buttom of the function:
````py 
elif LOSSFUNCTION == "MYLOSS":
        ...                  #do whatever you need
        loss_function = MyLoss(...)
````
The lossfunction will be called in the following way:
````lossfunction(y_pred, y_gt) ```` with ````y_pred.shape = [batch_size, num_classes, height, width] ```` and ````y_gt.shape = [batch_size, height, width]````.
If you need the data in another format you can use for example *lambda functions* (look at the definition of "DC_CE" loss in the get_loss_function_from_cfg).

## Optimizer
The optimizer is defined using the *get_optimizer_from_cfg* function in *utils/optimizer*. The inputs of the function are the models *parameters* as well the complete *cfg*. 
An custom optimizer can be added to *get_optimizer_from_cfg* by:
````py    
elif cfg.optimizer == "MYOPTIMIZER:
        ...                  #do whatever you need
        return My_Optimizer(...)
````
## LR Scheduler
The lr scheduler is defined using the *get_scheduler_from_cfg* function in *utils/lr_scheduler*. 
The input of the function are the opimizer, max steps and the cfg.
To add a custom lr scheduler you have to define the scheduler and its config in the following way:

````py
elif cfg.lr_scheduler == "MYSCHEDULER":
    ...                  #do whatever you need
    lr_scheduler = My_Scheduler(...)
    lr_scheduler_config = {"scheduler": lr_scheduler, 'interval': 'step', 'frequency': 1,
                           "monitor": "metric_to_track"}
````
lr_scheduler is your custom scheduler. The config is needed to tell Pytorch Lightning how to call your scheduler.
For example the *interval* parameter can set to *step* or *epoch*, and accordingly a lr_scheduler.step() is executed after each step or only at the end of the epoch.


