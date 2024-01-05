# Dataset

The Dataset config group is used to define the folder structures and the file formats.

---
## Setting up the Data

### Cityscapes

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

### Cityscapes_Coarse

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

### PASCAL Context

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

### ADE20K

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

### DACL10K

<details><summary>Click to expand/collapse</summary>
<p>

Download the data from [here](https://eval.ai/web/challenges/challenge-page/2130/overview).
Afterward run the following script. 
The script converts the masks from json (given as polygons) into individual binary image for each class with {image_name}_{class_index}.png.
Additionally, the images and masks are resized to the size 512x512.

````shell
python src/data_processing/process_dacl.py --root="/home/.../dacl10k_v2_devphase" --output="/home/.../dacl10k_dataset_512"
````

The last step is to adopt the path parameter.
Open the file of the environment you are using (by default *config/environment/local.yaml*) and
adopt the dacl10k path to the location of the folder where your *images* and *annotations* are.
For this example this would look like this:

````yaml
config/environment/local.yaml
  ─────────────────────────────
...
paths:
  dacl10k: home/.../dacl10k_dataset_512
````

</p>
</details>

## Customize

<details><summary>Click to expand/collapse</summary>
<p>

To define your own dataset create a yaml file named after your dataset, copy and fill out the content of _template_.yaml.
It is basically about giving information about the dataset structure (folder and file format).

````yaml
### Template for defining a Dataset
### '???': is required
### 'optional': can be deleted if not required or keep default
### otherwise: change to your needs
name: MyDataset             # (Unique) Name of the Dataset is needed for appropriate logging
num_classes: ???            # Number of classes inside the dataset (ignore index is not a class!).
segmentation_type: semantic # Which type of segmentation task to be solved (one of: semantic | multilabel | instance).
class_labels:               # Class names, is needed because it makes some things (logging, ...) easier and more comfortable.
  - class_name_1
  - class_name_2
  - ...
ignore_index:               # (optional) Ignore index of the dataset if used (otherwise keep 255).
class_weights:              # (optional) Weighting of classes int the loss function. Required when a such lossfunction is used.
root: ???                   # Root to the Dataset. Can be hard coded or defined in the environment.yaml (e.g. ${paths.MyDataset}).
img_folder: ???             # Folder which contains the training images (Can also be a folder structure).
label_folder: ???           # Folder which contains the training labels (Can also be a folder structure).
img_folder_val: ???         # Folder which contains the validation images (Can be Null when doing Cross Validation).
label_folder_val: ???       # Folder which contains the validation labels (Can be Null when doing Cross Validation).
img_folder_test:            # (optional) Folder which contains the test images (If Null validation data is used for testing).
label_folder_test:          # (optional) Folder which contains the test labels (If Null validation data is used for testing).
dtype: ".png"               # Data Type of the image files, the '.' is always required (.png | .jpg | .tif | .tiff).
dtype_mask: ".png"          # Data Type of the label files, the '.' is always required (.png | .tif | .tiff).
img_prefix: ""              # (optional) Prefix of the image files to distinguish from other files in the folder.
img_postfix: ""             # (optional) Postfix of the image files to distinguish from other files in the folder.
label_prefix: ""            # (optional) Prefix of the label files to distinguish from other files in the folder.
label_postfix: ""           # (optional) Postfix of the label files to distinguish from other files in the folder.
````

</p>
</details>