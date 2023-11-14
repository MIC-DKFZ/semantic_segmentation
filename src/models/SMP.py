import segmentation_models_pytorch as smp


def get_smp_model_class(arch):
    # Check if the provided architecture is a valid model class in the library
    if hasattr(smp, arch):
        # Get the model class dynamically using getattr
        model_class = getattr(smp, arch)
        return model_class
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


def get_seg_model(
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    arch: str = "Unet",
    backbone: str = "resnet34",
    **kwargs,
):
    """
    Initialize a SMP Model.

    Parameters
    ----------
    num_classes: int
    pretrained: bool
    aux_loss: aux_loss
    arch: str
        Unet, Unet++, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3+
    backbone: str
        ResNet, ResNeXt, ResNeSt, Res2Ne(X)t, RegNet(x/y), GERNet, SE-Net, SK-ResNe(X)t, DenseNet, Inception, EfficientNet, MobileNet, DPN, VGG, Mix Vision Transformer, MobileOne
    kwargs

    Returns
    -------
    A SMP model :
    """
    encoder_weights = "imagenet" if pretrained else None
    model = get_smp_model_class(arch)(
        encoder_name=backbone,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )

    return model
