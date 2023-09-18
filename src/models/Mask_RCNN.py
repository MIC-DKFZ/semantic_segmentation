from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform


# Disable the resize and normalize transform since we take care of this in the data augmentation pipeline
class GeneralizedRCNNTransform_no_transform(GeneralizedRCNNTransform):
    def resize(self, image, target=None):
        return image, target

    def normalize(self, image):
        return image


def get_model_50(
    num_classes, pretrained=True, version="v1", disable_transforms=False, *args, **kwargs
):
    # load an instance segmentation model pre-trained on COCO
    if pretrained:
        if version == "v1":
            model = models.detection.maskrcnn_resnet50_fpn(
                weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, box_detections_per_img=1000
            )
        elif version == "v2":
            model = models.detection.maskrcnn_resnet50_fpn_v2(
                weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_detections_per_img=1000
            )
    else:
        if version == "v1":
            model = models.detection.maskrcnn_resnet50_fpn(box_detections_per_img=1000)
        elif version == "v2":
            model = models.detection.maskrcnn_resnet50_fpn_v2(box_detections_per_img=1000)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    if disable_transforms:
        model.transform = GeneralizedRCNNTransform_no_transform(
            min_size=None, max_size=None, image_mean=None, image_std=None
        )

    return model
