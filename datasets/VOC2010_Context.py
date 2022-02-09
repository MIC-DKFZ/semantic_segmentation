import os
import glob

import torch
import torchvision.utils

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

from utils.utils import get_logger
log = get_logger(__name__)

CLASSES = ('background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
           'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
           'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
           'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
           'floor', 'flower', 'food', 'grass', 'ground', 'horse',
           'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
           'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
           'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
           'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
           'window', 'wood')

PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
           [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
           [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
           [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
           [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
           [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
           [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
           [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
           [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
           [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
           [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
           [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
           [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
           [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
           [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]



class VOC2010_Context_dataset(torch.utils.data.Dataset):
    def __init__(self,root,split="train",num_classes=60,ignore_index=255,transforms=None):
        if isinstance(root, str):
            root_imgs=root
            root_labels=root
        else:
            root_imgs = root.IMAGES
            root_labels = root.LABELS

        self.split=split
        self.ignore_index=ignore_index
        self.num_classes=num_classes
        if split=="test": split="val"
        imgs_path=os.path.join( root_imgs ,"Images" , split  , "*.jpg" )

        masks_path = os.path.join(root_labels, "Annotations", split, "*.png")

        self.imgs = list(sorted(glob.glob( imgs_path)))
        self.masks = list(sorted(glob.glob( masks_path)))

        self.transforms=transforms
        #log.info("Dataset: VOC2010_Context %s - %s images - %s masks",split,  len(self.imgs),len(self.masks))
        print("Dataset: VOC2010_Context",split,  len(self.imgs),len(self.masks))


    def reduce_num_classes(self,mask):
        mask=mask-1
        mask[mask==-1]=self.ignore_index
        return mask

    def __getitem__(self, idx):
        img =cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask=cv2.imread(self.masks[idx],-1)

        if self.num_classes==59:
            mask=self.reduce_num_classes(mask)
        #if self.split=="val":
        #    mask_o = torch.from_numpy(mask)
        #transforms2 = A.Compose([
        #    ToTensorV2()])

        #if self.split in ["val3","test3"]:
        #    transformed = self.transforms(image=img)
        #    img = transformed['image']
        #    mask = torch.from_numpy(mask)  # .unsqueeze(0)
        #else:

        transformed = self.transforms(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']
        #transformed = transforms2(image=None,mask=mask)
        #mask = transformed['mask']
        #if self.split=="val":
        #    return img, mask.long(),mask_o
        #    return img, mask.long(), idx
        #sample={"img":img,"mask":mask.long()}
        #return sample

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)


def show_voc(img=None, mask=None, alpha=.5, classes=19, mean=[0, 0, 0], std=[1, 1, 1]):
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
                mask_np[x, y] = mappig[class_id]
            # color=classes[class_id].color
        # print(img.shape)
        mask_pil = Image.fromarray(np.uint8(mask_np))
        return mask_pil

    mapping=PALETTE
    i = img is not None
    m = mask is not None
    if i and not m:
        return show_img(img)
    elif not i and m:
        return show_mask(mask, mapping)
    elif i and m:
        img_pil = show_img(img)
        mask_pil = show_mask(mask, mapping)
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

    transforms = A.Compose([
        #A.RandomCrop(width=768, height=768),
        #A.SmallestMaxSize( max_size= 520),
        #A.RandomScale(scale_limit=(-0.5,1),always_apply=True,p=1.0),
        A.PadIfNeeded(min_height=520,min_width=520,border_mode=0, value=0,mask_value=255),
        #A.Resize(p=1.0,width=480, height=480),
        A.RandomCrop(width=520, height=520,always_apply=True,p=1.0),
        #A.ColorJitter(brightness=9,contrast=0,saturation=0,hue=0),
        #A.RGBShift(p=1,r_shift_limit=10,g_shift_limit=10,b_shift_limit=10),
        #A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],always_apply=True
        ),
        ToTensorV2()])
    print(transforms)
    Path = "/home/l727r/Desktop/Datasets/VOC2010_Context"
    Cityscape_train = VOC2010_Context_dataset(Path, "train", transforms=transforms)





    #for i in range(0,50):
    img, mask = Cityscape_train[465]
    print(mask[0,0])
    #print(img.shape)
    #print(torch.unique(mask))
    out = show_voc(img=img, alpha=1., mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    out1 = show_voc(mask=mask, alpha=1., mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #out.show()

    out.save("out.png")
    out1.save("out1.png")

    # def collate_fn(batch):
    #    return tuple(zip(*batch))



