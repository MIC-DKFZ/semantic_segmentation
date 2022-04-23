import glob
import os

import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import pickle
from utils.visualization import show_data
from utils.utils import get_logger
log = get_logger(__name__)

ignore_label=255

#Data shape [3, 1064, 1440],[3, 1072, 1728],[3, 1024, 1280],[3, 1080, 1920],[3, 720, 1280],[3, 1072, 1704]
#

PALETTE = [[0, 0, 0], [255,0,0]]

class HidaHackathon2022_dataset(torch.utils.data.Dataset):
    def __init__(self,root,fold,split="train",transforms=None):
        self.root=root
        with open(os.path.join(root, "splits_final.pkl"), "rb") as file:
            splits = pickle.load(file)[fold]

        #if split="train"
        self.imgs='Orthophoto/ortho_'
        self.modalities=['Slope/slope_', 'Prof_curv/pcurv_', 'Tang_curv/tcurv_', 'Flow_Direction/flowdir_', 'Aspect/aspect_', 'DTM/dtm_', 'Flow_Accum/flowacc_', 'Topo_Wetness/twi_']
        self.masks="Ground_truth/mask_"

        if split=="val" or split=="test":
            self.data = [int(s.replace("case","")) for s in splits["val"]]
        elif split=="train":
            self.data = [int(s.replace("case","")) for s in splits["train"]]

        self.transforms=transforms



    def __getitem__(self, idx):

        img=cv2.imread(os.path.join(self.root,self.imgs+str(idx+1)+".png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for mod in self.modalities:
            img_mod = cv2.imread(os.path.join(self.root,mod+str(idx+1)+".png"),-1)
            img=np.concatenate((img, np.expand_dims(img_mod,axis=2)), axis=2)

        mask=cv2.imread(os.path.join(self.root,self.masks+str(idx+1)+".png"),-1)
        mask=np.where(mask==128,1,mask)
        mask=np.where(mask==255,2,mask)

        #mask=np.where(mask==255,mask,2)
        #mask=np.where(mask,255,2)

        transformed = self.transforms(image=img, mask=mask)
        img= transformed['image']
        mask = transformed['mask']

        return img,mask.long()

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":

    transforms = A.Compose([
        A.RandomScale(scale_limit=(0, 1), always_apply=True, p=1.0),
        #A.RGBShift( p=1,r_shift_limit= 10,g_shift_limit= 10,b_shift_limit= 10),
        #A.PadIfNeeded(min_height=512, min_width=1024),
        #A.RandomCrop(height=512,width=1024),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.Normalize(
            mean=[0.485, 0.456, 0.406,0,0,0,0,0,0,0,0],#[0.485, 0.456, 0.406,0.5, 0.5, 0.5,0.5, 0.5, 0.5,0.5,0.5],
            std=[0.229, 0.224, 0.225,1,1,1,1,1,1,1,1]),#[0.229, 0.224, 0.225,0.229, 0.224, 0.225,0.229, 0.224, 0.225,0.229, 0.224]),
        ToTensorV2()])

    root_path="/home/l727r/Desktop/Datasets/Hackathon_500/Hackathon_data"
    #types=glob.glob(root_path+"/*")
    #print([t.rsplit("/",1)[1] for t in types])
    #with open(os.path.join(root_path,"splits_final.pkl"),"rb") as file:
    #    splits=pickle.load(file)
    #print(len(splits[0]["train"]))
    Arch=HidaHackathon2022_dataset(root=root_path,fold=1,split="val",transforms=transforms)
    img,mask=Arch[0]
    print(img.shape,mask.shape)
    print(torch.min(img),torch.max(img))
    for i in range(len(Arch)):
        img,mask=Arch[i]
        print(torch.unique(mask))
    #print(len(EndoCV))
    #EndoCV=EndoCV2022_dataset(root=root_path,fold=1,split="val",transforms=transforms)
    #print(len(EndoCV))

    #img,mask=EndoCV[150]
    #out = show_data(img,mask,alpha=0.5,color_mapping=[[0,0,0],[255,0,0]], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #out.show()




