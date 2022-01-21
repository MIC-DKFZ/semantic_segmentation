import os
import glob
from collections import namedtuple
import logging
log = logging.getLogger(__name__)

import torch
import torchvision.utils

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

#https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])

classes_34 = [
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

classes_19 = [
    CityscapesClass('road', 0, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 1, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('building', 2, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 3, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 4, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('pole', 5, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('traffic light', 6, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 7, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 8, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 9, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 10, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 11, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 12, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 13, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 14, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 15, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('train', 16, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 17, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 18, 18, 'vehicle', 7, True, False, (119, 11, 32)),
]

ignore_label=255

label_mapping = {-1: ignore_label, 0: ignore_label,
                      1: ignore_label, 2: ignore_label,
                      3: ignore_label, 4: ignore_label,
                      5: ignore_label, 6: ignore_label,
                      7: 0, 8: 1, 9: ignore_label,
                      10: ignore_label, 11: 2, 12: 3,
                      13: 4, 14: ignore_label, 15: ignore_label,
                      16: ignore_label, 17: 5, 18: ignore_label,
                      19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                      25: 12, 26: 13, 27: 14, 28: 15,
                      29: ignore_label, 30: ignore_label,
                      31: 16, 32: 17, 33: 18}



class Cityscapes_dataset(torch.utils.data.Dataset):
    def __init__(self,root,split="train",transforms=None):
        if isinstance(root, str):
            root_imgs=root
            root_labels=root
        else:
            root_imgs = root.IMAGES
            root_labels = root.LABELS

        if split=="test":
            split="val"
        imgs_path=os.path.join( root_imgs ,"leftImg8bit_trainvaltest", "leftImg8bit" , split , "*" , "*_leftImg8bit.png" )

        masks_path = os.path.join(root_labels, "gtFine_trainvaltest", "gtFine", split, "*", "*_gt*_labelIds_19classes.png")
        #elif num_classes==34:
        #    masks_path=os.path.join( root_labels,"gtFine_trainvaltest" , "gtFine" , split , "*" , "*_gt*_labelIds.png" )

        self.imgs = list(sorted(glob.glob( imgs_path)))
        self.masks = list(sorted(glob.glob( masks_path)))

        self.transforms=transforms
        log.info("Dataset: Cityscape %s - %s images - %s masks",split,  len(self.imgs),len(self.masks))



    def __getitem__(self, idx):

        img =cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask=cv2.imread(self.masks[idx],-1)

        transformed = self.transforms(image=img, mask=mask)
        img= transformed['image']
        mask = transformed['mask']

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)


def show_cityscape(img=None, mask=None, alpha=.5, classes=19, mean=[0, 0, 0], std=[1, 1, 1]):
    # (input[channel] - mean[channel]) / std[channel]
    def show_img(img_tens):
        img_tens = (img_tens.permute(1, 2, 0) * torch.Tensor(std)) + torch.Tensor(mean)
        img_tens = img_tens.permute(2, 0, 1)
        img_pil = torchvision.transforms.ToPILImage(mode='RGB')(img_tens)
        return img_pil

    def show_mask(mask_tens, mappig):
        #mask is a prediction --> bring into GT format
        if mask_tens.dim() != 2:
            mask_tens = torch.argmax(mask_tens.squeeze(), dim=0).detach().cpu()
        w, h = mask_tens.shape
        mask_np = np.zeros((w, h, 3))
        for class_id in torch.unique(mask_tens):
            x, y = torch.where(mask_tens == class_id)
            if class_id == 255:
                mask_np[x, y] = [0, 0, 0]
            else:
                mask_np[x, y] = mappig[class_id].color
            # color=classes[class_id].color
        # print(img.shape)
        mask_pil = Image.fromarray(np.uint8(mask_np))
        return mask_pil

    if classes == 19:
        mappig = classes_19
    elif classes == 34:
        mappig = classes_34
    i = img is not None
    m = mask is not None
    if i and not m:
        return show_img(img)
    elif not i and m:
        return show_mask(mask, mappig)
    elif i and m:
        img_pil = show_img(img)
        mask_pil = show_mask(mask, mappig)
        return Image.blend(img_pil, mask_pil, alpha=alpha)
    return

def viz_color_encoding():
    width = 700
    height = 60

    def sub_viz(classes):
        num = len(classes)
        img = np.zeros((num * height, width, 3), np.uint8)

        for index, c in enumerate(classes):
            img[index * height:(height + 1) * height, :] = c.color
            cv2.putText(img, str(index) + ".", (10, (index) * height + int(height * 0.75)), cv2.FONT_HERSHEY_COMPLEX,
                        1.5,
                        (255, 255, 255), 2)
            cv2.putText(img, c.name, (150, (index) * height + int(height * 0.75)), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                        (255, 255, 255), 2)
            print(c.name, c.color)
            # break
        for index in range(1, num):
            cv2.line(img, (0, index * height), (width, index * height), (255, 255, 255))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    img_full=sub_viz(classes_34)
    classes_eval = [c for c in classes_34 if c.ignore_in_eval == False]
    img_eval= sub_viz(classes_eval)

    cv2.imwrite('Cityscape_color_encoding_full.png', img_full)
    cv2.imwrite('Cityscape_color_encoding.png', img_eval)



if __name__ == "__main__":
    #transforms = A.Compose([
    #    A.RandomCrop(width=768, height=768),
    #    A.HorizontalFlip(p=0.5),
    #    A.RandomBrightnessContrast(p=0.2),
    #    A.ColorJitter(p=0.2),
    #    A.OneOf([
    #        A.GaussNoise(p=1),
    #        A.ISONoise(p=1),
    #        A.MultiplicativeNoise(p=1)],
    #        p=0.2),
    #    A.OneOf([
    #        A.GaussianBlur(p=1),
    #        A.MedianBlur(p=1),
    #        A.MotionBlur(p=1),
    #        A.GlassBlur(p=1)],
    #        p=0.2),
    #    A.Normalize(
    #        mean=[0.485, 0.456, 0.406],
    #        std=[0.229, 0.224, 0.225],
    #    ),
    #    ToTensorV2()])
    #A.save(transforms,"config/transform_selected.json")
    #print(transform )
    #root="~/home/l727r/Desktop/Cityscape"
    ##splits=["train","val","test"]
    #masks_path = os.path.join(root, "gtFine_trainvaltest", "gtFine", split, "*", "*_gt*_labelIds.png")
    #for c in classes_19:
    #    print(c)
    #for split in splits:
    #    masks_path = os.path.join(root, "gtFine_trainvaltest", "gtFine", split, "*", "*_gt*_labelIds.png")
    #    masks = list(sorted(glob.glob(masks_path)))
    #    print(masks)
    #    break

    #transforms = A.load("config/transform_auto.json")
    #print(transforms)
    transforms = A.Compose([
        #A.RandomCrop(width=768, height=768),
        A.RandomScale(scale_limit=(-0.5,1),always_apply=True,p=1.0),
        A.PadIfNeeded(min_height=768,min_width=768),
        #A.Resize(p=1.0,width=1024, height=512),
        A.RandomCrop(width=768, height=768,always_apply=True,p=1.0),
        #A.ColorJitter(brightness=9,contrast=0,saturation=0,hue=0),
        A.RGBShift(p=1,r_shift_limit=10,g_shift_limit=10,b_shift_limit=10),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],always_apply=True
        ),
        ToTensorV2()])
    print(transforms)
    #print(transforms)
    #A.save(transforms,"config/transform_test.yaml",data_format='yaml')
    #trans={"__version__": "1.1.0", "transform": {"__class_fullname__": "Compose", "p": 1.0, "transforms": [
    #    {"__class_fullname__": "RandomScale", "always_apply": true, "p": 1.0, "interpolation": 1,
    #     "scale_limit": [-0.5, 1.0]},
    #    {"__class_fullname__": "RandomCrop", "always_apply": true, "p": 1.0, "height": 512, "width": 1024},
    #    {"__class_fullname__": "HorizontalFlip", "always_apply": false, "p": 0.5},
    #    {"__class_fullname__": "Normalize", "always_apply": true, "p": 1.0, "mean": [0.485, 0.456, 0.406],
    #     "std": [0.229, 0.224, 0.225], "max_pixel_value": 255.0},
    #    {"__class_fullname__": "ToTensorV2", "always_apply": true, "p": 1.0, "transpose_mask": false}],
    #                                       "bbox_params": null, "keypoint_params": null, "additional_targets": {}}}

    #for i in range(0,100):
    #    print(A.RandomScale(scale_limit=(1,1),always_apply=True).get_params())
    cityscapesPath = "/home/l727r/Desktop/Datasets/cityscapes"
    Cityscape_train = Cityscapes_dataset(cityscapesPath, "train", transforms=transforms)
    #for i in range(0,50):
    img, mask = Cityscape_train[100]
    print(img.shape)
    print(torch.unique(mask))
    out = show_cityscape(img=img, mask=mask, alpha=0., mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #out.show()
    out.save("out.png")

    #transform = Compose([RandomCrop(769), RandomHorizontalFlip(), ToTensor(),
    #                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # transform=Compose([RandomCrop(769),RandomHorizontalFlip(),ToTensor()])
    #Cityscape_train = Cityscape_dataset(cityscapesPath, "train", transforms=transforms)
    # print(Cityscape_train[0][1].shape)
    # print(cityscape_info())

    # for i in range(len(Cityscape_train)):
    #img, mask = Cityscape_train[100]
    #print(torch.unique(mask))
    # print(mask)
    #    print(torch.unique(mask))
    # show_cityscape(mask=mask)
    #out = show_cityscape(img=img, mask=mask,alpha=0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #out.show()
    #out_i = show_cityscape(img=img, alpha=0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #out_m = show_cityscape( mask=mask, alpha=0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #out_m.show()
    #out_i.show()
    #show_cityscape_interactive(img=img, mask=mask, alpha=0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # out=show_cityscape( img=img).show()

    # out.save("Test.png")



