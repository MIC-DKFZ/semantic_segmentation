# Dataset



<details><summary>Configure</summary>
<p>

Currently, the following datasets are supported, and they can be selected as shown [here](../config#selecting-a-dataset). 
By default, the cityscapes dataset is used.

- **Cityscapes**: [Cityscapes dataset](https://www.cityscapes-dataset.com/) with using fine
  annotated images. Contains 19 classes and 2.975 training and 500 validation images.
- **Cityscapes_coarse**: [Cityscapes dataset](https://www.cityscapes-dataset.com/) with using coarse
  annotated training images. Contains 19 classes and ~20.000 training images.
  For validation the 500 fine annotated images from Cityscape are used.
- **Cityscapes_fine_coarse**: [Cityscapes dataset](https://www.cityscapes-dataset.com/) with using
  coarse and fine annotated training images. Contains 19 classes and ~23.000 training images.
  For validation the 500 fine annotated images from Cityscape are used.
- **VOC2010_Context**: [PASCAL Context](https://cs.stanford.edu/~roozbeh/pascal-context/) dataset,
  which is an extension for
  the [PASCAL VOC2010 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/) and contains
  additional segmentation masks.
  It contains 5.105 training and 4.998 validation images.
  This dataset contains 59 classes. For the 60 class setting see below.
- **VOC2010_Context_60**: The **VOC2010_Context** dataset with an additional background class,
  resulting in a total of 60 classes.

</p>
 </details>

<details><summary>Customize</summary>
<p>

Defining a custom dataset is done in two steps, first defining your custom pytorch dataset and
afterwards setting up its config file.

1. **Defining your pytorch Dataset**, thereby consider that the following structure is required (
   mainly pytorch basic) and see the dummy below:
    - \__init__(self, custom_args, split, transforms):
        - *custom_args*: your custom input arguments (for example data_root etc.). They will be
          given to your dataset from the config file (see below).
        - *split*: one of the following strings: \["train","val","test"]. To define if train,
          validation or test set should be returned.
        - *transforms*: Albumentations transformations
    - \__getitem__(self, idx):
        - getting some index and should the output should look similar to: *return img, mask*
        - with ````img.shape = [c, height, width]```` and ````mask.shape = [height, width]````,
          where *c* is the number of channels. For example *c=3* if you use RGB data.
    - \__len(self)__:
        - return the number of samples in your dataset, something like: *return len(self.img_files)*
   ````py
   class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self,root,split,transforms):
        # get your data for the corresponding split
        if split=="train":
             self.imgs = ...
             self.masks = ...
        if split=="val" or split=="test":       # if you have dont have a test set use the validation set
             self.imgs = ...
             self.masks = ...
        
        # save the transformations
        self.transforms=transforms

    def __getitem__(self, idx):
        # reading images and masks as numpy arrays
        img =cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR order

        mask=cv2.imread(self.masks[idx],-1)

        # thats how you apply Albumentations transformations
        transformed = self.transforms(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']
        
        return img, mask.long()

    def __len__(self):
        return len(self.imgs)
   ````
2. **Setting up your dataset config**
    - Create a *custom_dataset.yaml* file in *config/datasets/*. For the content of the *.yaml* file
      adopt the following dummy:

   ````yaml 
   #@package _global_
   # dataset is used to initialize your custom dataset, 
   # _target_: should point to your dataset class
   # afterwards you can handle your custom input arguments which are used to initialize the dataset
   dataset:
     _target_: datasets.MyDataset.Custom_dataset 
     root: /home/.../Datasets/my_dataset     #the root to the data as an example input
     #root: ${paths.my_dataset}               #the root if defined in config/environment/used_env.yaml
     input1: ...                    #All your other input arguments
     input2: ...
   # DATASET is used to store information about the dataset which are needed during training
   DATASET:
     # Required dataser arguments
     NAME:            # Used for the logging directory
     NUM_CLASSES:     # Needed for defining the model and the metrics
     # (Optional) but needed if ignore index should be used
     IGNORE_INDEX:    # Needed for the loss function, if no ignore indes set to 255 or another number
                      # which do no occur in your dataset 
     # (Optional) needed if weighted lossfunctions are used
     CLASS_WEIGHTS: [ 0.9, 1.1, ...]                
     # (Optional) can be used for nicer for logging 
     CLASS_LABELS:
        - class1
        - class2 ...
   ````
3. **Train on your Dataset**
   ````shell
    python training.py dataset=custom_dataset     # to select config/dataset/custom_dataset.yaml
    ````
