# DACL Challenge

## Installation

Follow the instructions in the readme and install all required packages.

## Data Processing 
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

## Training
````shell
python training.py experiment=dacl10k/cv fold=0
python training.py experiment=dacl10k/cv fold=1
python training.py experiment=dacl10k/cv fold=2
python training.py experiment=dacl10k/cv fold=3
python training.py experiment=dacl10k/cv fold=4
````

## Inference
**Note**: All images in the input dir have to be resized to 512x512.

````shell
python predict.py -i path.input.images -o path.output.folder ckpt_dir= .../logs/dacl10k_CV/Maxvit512_xlarge/run__experiment_dacl10k.cv/
````

