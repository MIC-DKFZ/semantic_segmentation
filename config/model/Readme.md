# Model

## Download Pretrained Weights

<details><summary>Configure</summary>
<p>
Pretrained weights for HRNet-based models (HRNet, OCR, MS OCR) are available on ImageNet, PadddleClas and Mapillary.
Download the preferred weights (direct download links below) and put them in the *pretrained/*
folder. PaddleClas weights are used by default.

- **ImageNet** weights: [source](https://github.com/HRNet/HRNet-Image-Classification#imagenet-pretrained-models), [direct download](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk)
- **Paddle**
  weights: [source](https://github.com/HRNet/HRNet-Image-Classification#imagenet-pretrained-models), [direct download](https://github.com/HRNet/HRNet-Image-Classification/releases/download/PretrainedWeights/HRNet_W48_C_ssld_pretrained.pth)
- **Mapillary**
  weights: [source](https://github.com/NVIDIA/semantic-segmentation#download-weights), [direct download](https://drive.google.com/file/d/1Whz--rurtBoIsfF-t3YB9NEt3GT6hBQI/view?usp=sharing)

For selecting the different weighs set (only for HRNet, OCR, MS OCR):

```shell
python training.py model.pretrained = False
python training.py model.pretrained_on = ImageNet
python training.py model.pretrained_on = Paddle
python training.py model.pretrained_on = Mapillary
```

</p>
 </details>

## Models

<details><summary>Configure</summary>
<p>

Currently, the following models are supported, by default hrnet is used. 

- **hrnet**: [High-Resolution Network (HRNet)](https://arxiv.org/pdf/1904.04514.pdf). Segmentation
  model with a single output.
- **hrnet_ocr**: [Object-Contextual Representations (OCR)](https://arxiv.org/pdf/1909.11065.pdf).
  A HRNet backbone with an OCR head.
  The model has two outputs, a primary and an auxiliary one.
- **hrnet_ocr_aspp**: Additionally including an ASPP module into the ORC model. Again the model has
  two outputs.
- **hrnet_ocr_ms**: [Hierarchical Multiscale Attention Network](https://arxiv.org/pdf/2005.10821.pdf).
  Extends ORC with multiscale and attention.
  The model has 4 outputs: primary, auxiliary, high_scale_prediction, low_scale_prediction
    - ``model.model.cfg.MODEL.MSCALE_INFERENCE`` is used to enable/disable the use of multiple scales (only during
      inference and validation), which is False by default.
    - ``model.model.cfg.N_SCALES`` defines the scales which are used during *MSCALE_INFERENCE*, by default *= [0.5, 1.0, 2.0]*
- **FCN**: including torchvision's FCN  ([docs]((https://pytorch.org/vision/stable/models.html#fully-convolutional-networks)), [paper](https://arxiv.org/pdf/1411.4038.pdf)).
Besides the arguments described in the [torchvision docs](https://pytorch.org/vision/stable/generated/torchvision.models.segmentation.fcn_resnet101.html#torchvision.models.segmentation.fcn_resnet101) you can specify the following arguments:
  - ``model.model.backbone`` can be resnet50 or resnet101, to define which version of the model should be used. resnet101 by default.
- **DeepLab**: including torchvision's DeepLabv3 ([docs]((https://pytorch.org/vision/stable/models.html#deeplabv3)), [paper]()). 
Besides the arguments described in the [torchvision docs](https://pytorch.org/vision/stable/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.deeplabv3_resnet101) you can specify the following arguments:
  - ``model.model.backbone`` can be resnet50 or resnet101, to define which version of the model should be used. resnet101 by default.
- **UNet**: Implementation of UNet ([paper](https://www.nature.com/articles/s41592-020-01008-z)
  , [source code](https://github.com/MIC-DKFZ/nnUNet)). No pretrained weights are available

</p>
 </details>

## Customize
<details><summary>Customize</summary>
<p>

Defining a custom model is done in two steps, first defining your custom pytorch model and
afterwards setting up its config file.

1. **Defining your Pytorch Model**, thereby the following thinks have to be considered:
    - **Model Input**: The input of the model will be a torch.Tensor of shape [batch_size, channels, height, width]).
    - **Model Output**: It is recommended that your model **return a dict** which contain all the models outputs.
      The naming can be arbitrary but the ordering matters. 
      For example if you have one output return as follows: ``return {"out": model_prediction}``. If
      you have multiple output to it analogues:
      ``return {"main": model_prediction, "aux": aux_out}``.
      The output of the model can also be a single Tensor, a list or a tuple, but in this case the output is converted into dict automatically.
      It should be noted that in each case the **order of the outputs is relevant**. Only the first output is
      used for updating the metric during validation or testing.
      Further the order of the outputs should match the order of your losses in *lossfunction* and
      the weights in *lossweights*.(see [Lossfunction](#loss-function) for more details on that)

2. **Setting up your model config**
    - Create a *custom_model.yaml* file in *config/model/*. Therby the name of the file defines how the model can be select over hydras commandline syntax.
   For the content of the *.yaml* file adopt the following dummy.

````yaml
name: ModelName           # Required for logging
arg1: ...                 # Store some custom stuff
pretrained: True          # e.g. if pretrained model should be used
model:
   _target_: models.my_model.get_model     # if you want to use a getter function to load weights 
                                           # or initialize you model
   #_target_: models.my_model.Model        # if you want to load the Model directly
   num_classes:  ${dataset.num_classes}    # example arguments, for example the number of classes
   pretrained: ${model.pretrained}         # of if pretrained weights should be used
   arg1: ...  
````
3. **Train your model**
   ````shell
    python training.py model=custom_model     # to select config/model/custom_model.yaml
    ````

</p>
</details>

