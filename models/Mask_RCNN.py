from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


def get_model_50(num_classes, pretrained=True):
    # load an instance segmentation model pre-trained on COCO
    # model = models.detection.maskrcnn_resnet50_fpn_v2(pretrained=False)
    if pretrained:
        # model = models.detection.maskrcnn_resnet50_fpn(weights="pretrained")
        model = models.detection.maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        )
    else:
        model = models.detection.maskrcnn_resnet50_fpn()

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    # model.transform = GeneralizedRCNNTransform(min_size=800,max_size=1500,image_mean=[0,0,0],image_std=[1,1,1])
    return model
