import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCNHead


def get_seg_model(
    num_classes: int, pretrained: bool, aux_loss: bool = None, backbone: str = "resnet101", **kwargs
) -> DeepLabV3:
    """
    Initialize a DeepLab Model with a resnet backbone
    First init the basic torchvision model
    to enable the use of pretrained weights with different number of classes, the last layer of the
    classifier + aux_classifier is adapted after the basic initialization

    Parameters
    ----------
    num_classes: int
    pretrained: bool
    aux_loss: aux_loss
    backbone: str
        resnet50 or resnet101
    kwargs

    Returns
    -------
    DeepLabV3 :
    """

    # load the deeplab model with the corresponding backbone
    if backbone == "resnet101":
        model = deeplabv3_resnet101(pretrained=pretrained, aux_loss=aux_loss, **kwargs)
    elif backbone == "resnet50":
        model = deeplabv3_resnet101(pretrained=pretrained, aux_loss=aux_loss, **kwargs)

    # to enable pretrained weights the last layer in the classifier head is adopted to match to the
    # number of classes after initializing of the model with pretrained weights
    in_channels = model.classifier[4].in_channels
    kernel_size = model.classifier[4].kernel_size
    stride = model.classifier[4].stride
    model.classifier[4] = torch.nn.Conv2d(in_channels, num_classes, kernel_size, stride)

    # the same is done for the aux_classifier if exists
    if hasattr(model, "aux_classifier"):
        in_channels = model.aux_classifier[4].in_channels
        kernel_size = model.aux_classifier[4].kernel_size
        stride = model.aux_classifier[4].stride
        model.aux_classifier[4] = torch.nn.Conv2d(in_channels, num_classes, kernel_size, stride)

    # For exchanging the complete Head
    # in_features = model.classifier[0].convs[0][0].in_channels
    # model.classifier = DeepLabHead(in_features, num_classes)
    # in_features_aux = model.aux_classifier[0].in_channels
    # model.aux_classifier = FCNHead(in_features_aux, num_classes)

    return model
