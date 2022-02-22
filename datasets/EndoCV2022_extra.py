import os
import glob
from collections import namedtuple
import pandas as pd

import torch
import torchvision.utils

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils.visualization import show_data
from utils.utils import get_logger
log = get_logger(__name__)

ignore_label=255

#Data shape [3, 1064, 1440],[3, 1072, 1728],[3, 1024, 1280],[3, 1080, 1920],[3, 720, 1280],[3, 1072, 1704]
#

PALETTE = [[0, 0, 0], [255,0,0]]

class EndoCV2022_dataset(torch.utils.data.Dataset):
    def __init__(self,root,fold,split="train",transforms=None):
        if split=="test":
            split="val"

        #data = pd.read_csv(os.path.join(root, "PoleGen2_4CV_2.csv"))
        data = pd.read_csv(os.path.join(root, "polypgen_cleaned_4CV.csv"))
        if fold == "all":
            data=data
        elif fold=="extra":
            if split=="train":
                data=pd.DataFrame()#None
            if split=="val":
                data=data
        elif split=="val":
            data = data[data.fold == fold]
        elif split=="train":
            data=data[data.fold!=fold]
        #print(len(data))
        self.imgs = []
        self.masks = []
        #folder=os.path.join( "EndoCV2022_ChallengeDataset","PolypGen2.0")
        folder="clean_PolypGen2.0"
        #
        #if data != None
        for i, d in data.iterrows():
            self.imgs.append(os.path.join(root,folder, d.vid_folder, "images",
                                       d.image_id))
            self.masks.append(
                os.path.join(root, folder, d.vid_folder, "masks", d.Mask_id))

        if split=="train":
            data_extra = pd.read_csv(os.path.join(root, "external_endocv2.csv"))

            for i, d in data_extra.iterrows():
                self.imgs.append(os.path.join(root,d.im_path))
                self.masks.append(os.path.join(root, d.mask_path))

        #imgs_extra=os.path.join(root,"renamed_pretrain_all_sequence_public_polyp","renamed_pretrain_all_sequence_segmentation","*","image","*")
        #masks_extra=os.path.join(root,"renamed_pretrain_all_sequence_public_polyp","renamed_pretrain_all_sequence_segmentation","*","label","*")
        #print(data_extra)
        #imgs_extra = list(sorted(glob.glob(imgs_extra)))
        #masks_extra = list(sorted(glob.glob(masks_extra)))
        #print(imgs_extra)
        #print("EX",len(imgs_extra),len(masks_extra))
        #self.masks = list(sorted(glob.glob(masks_path)))

        self.transforms = transforms
        log.info("Dataset: EncoCV2022 %s - Fold %s - %s images - %s masks",split, fold,  len(self.imgs),len(self.masks))


    def __getitem__(self, idx):
        #print(self.imgs[idx])
        #print(self.masks[idx])
        img =cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask=(cv2.imread(self.masks[idx]) > 20).astype(np.uint8)[:,:,0]


        transformed = self.transforms(image=img, mask=mask)
        img= transformed['image']
        mask = transformed['mask']

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)



if __name__ == "__main__":

    transforms = A.Compose([
        #A.RandomScale(scale_limit=(-0.5, 0), always_apply=True, p=1.0),
        A.RGBShift( p=1,r_shift_limit= 10,g_shift_limit= 10,b_shift_limit= 10),
        A.PadIfNeeded(min_height=512, min_width=1024),
        A.RandomCrop(height=512,width=1024),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2()])

    root_path="/home/l727r/Desktop/endocv2022/official EndoCV2022 dataset"
    EndoCV=EndoCV2022_dataset(root=root_path,fold="all",split="train",transforms=transforms)
    print(len(EndoCV))
    EndoCV_val=EndoCV2022_dataset(root=root_path,fold="all",split="val",transforms=transforms)
    print(len(EndoCV_val))

    #img,mask=\
    img,mask=EndoCV[3100]
    #print(img.shape,mask.shape)
    out = show_data(img,mask,alpha=0.5,color_mapping=[[0,0,0],[255,0,0]], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    out.show()
    #shapes=[]
    #for i in tqdm(range(0,len(EndoCV))):
    #    img,mask=EndoCV[i]
    #    if EndoCV.masks[i].split(".")[-1]=="jpg":
    #        cv2.imshow("window",mask)
    #        cv2.waitKey(100)
    #    #if img.shape not in shapes: shapes.append(img.shape)
    #    #print(img.shape,mask.shape)
    #print(shapes)
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
    #transforms = A.Compose([
    #    #A.RandomCrop(width=768, height=768),
    #    A.RandomScale(scale_limit=(-0.5,1),always_apply=True,p=1.0),
    #    A.PadIfNeeded(min_height=768,min_width=768),
    #    #A.Resize(p=1.0,width=1024, height=512),
    #    A.RandomCrop(width=768, height=768,always_apply=True,p=1.0),
    #    #A.ColorJitter(brightness=9,contrast=0,saturation=0,hue=0),
    #    A.RGBShift(p=1,r_shift_limit=10,g_shift_limit=10,b_shift_limit=10),
    #    A.HorizontalFlip(p=0.5),
    #    A.Normalize(
    #        mean=[0.485, 0.456, 0.406],
    #        std=[0.229, 0.224, 0.225],always_apply=True
    #    ),
    #    ToTensorV2()])
    #print(transforms)
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
    #cityscapesPath = "/home/l727r/Desktop/Datasets/cityscapes"
    #Cityscape_train = Cityscapes_dataset(cityscapesPath, "train", transforms=transforms)
    #for i in range(0,50):
    #img, mask = Cityscape_train[100]
    #print(img.shape)
    #print(torch.unique(mask))
    #out = show_cityscape(img=img, mask=mask, alpha=0., mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #out.show()
    #out.save("out.png")

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




