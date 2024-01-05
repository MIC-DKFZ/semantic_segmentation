# Experiment

The Experiment config group is used to plan your experiments by composing the corresponding config groups (dataset, model, ...) and override parameters.

## Configure

## Customize

<details><summary>Click to expand/collapse</summary>
<p>

To define your own experiment create a yaml file named after your experiment, copy and fill out the content of _template_.yaml.
Edit, add or remove the entries to your needs. You can override everything from the config/training.yaml 
````yaml
#@package _global_
# Everything which is defined in config/training.yaml can be overwriten here.
# This is a template containing the most important argument, delete them if not needed
defaults:
  - override /dataset:                              # one of config/dataset/*, e.g. Cityscapes
  - override /training_scheme: TrainVal             # one of config/training_scheme/*
  - override /model:                                # one of config/model/*
  - override /metric: mIoU                          # one of config/metric/*
  - override /optimizer: SGD                        # one of config/optimizer/*. (requires to adopt lr, weight_decay)
  - override /augmentation@augmentation.train:      # one of config/augmentations/*
  - override /augmentation@augmentation.val:        # one of config/augmentations/*
  - override /augmentation@augmentation.test:       # one of config/augmentations/*

# Configure the augmentations
augmentation:
  cfg:
    scale_limit: [-0.5, 1.0]  # Scale limits, if scaling augmentations are used - 0 = no scaling
    crop_size: [512, 1024]    # Crop/patch size, used for: cropping operations; patch-wise inference; sampling dataset
    mean: [ 0.485, 0.456, 0.406 ]     # mean for normalization
    std: [ 0.229, 0.224, 0.225 ]      # std for normalization
    pad_size: ${augmentation.cfg.crop_size}   # Size for Padding, used for padding operations
    pad_mode: 0               # Mode for Padding: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101
    pad_val: 0                # Value to pad the image with if padding is used
    pad_mask: ${dataset.ignore_index} # Value to pad the mask with if padding is used
    N: 3                      # Needed when using Randaugment, defines how many operations are used
    M: 3                      # Needed when using Randaugment, defines the magnitude of the operations
````


</p>
</details>