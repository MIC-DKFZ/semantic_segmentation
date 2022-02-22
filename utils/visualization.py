import torch
import torchvision
from PIL import Image
import numpy as np


def show_data(img=None, mask=None, color_mapping=None,alpha=.5,black=[], mean=[0, 0, 0], std=[1, 1, 1]):
    #img: img tensor
    #mask: mask tensor or prediction of the model(also tensor)
    #optional colormapping: list with colors as list (list_id corresponds to class_id)
    #alpha, if both img and mask are provided the alpha during blending
    #black: list of values which should be black, eg ignore index or background
    #mean and std: if data gets normalized this has to be inverted for nice visualization
    def show_img(img_tens):
        # (input[channel] - mean[channel]) / std[channel]
        img_tens = (img_tens.permute(1, 2, 0) * torch.Tensor(std)) + torch.Tensor(mean)
        img_tens = img_tens.permute(2, 0, 1)
        img_pil = torchvision.transforms.ToPILImage(mode='RGB')(img_tens)
        return img_pil

    def show_mask(mask_tens, mappig,blacking):
        if mask_tens.dim() != 2:    #mask is a prediction --> bring into GT format
            mask_tens = torch.argmax(mask_tens.squeeze(), dim=0).detach().cpu()
        w, h = mask_tens.shape
        mask_np = np.zeros((w, h, 3))
        for class_id in torch.unique(mask_tens):
            x, y = torch.where(mask_tens == class_id)

            if class_id in black:
                color=[0,0,0]
            elif mappig!=None:
                color=mappig[class_id]
            else:
                color = list(np.random.choice(range(256), size=3))
            mask_np[x, y] = color

        mask_pil = Image.fromarray(np.uint8(mask_np))
        return mask_pil

    i = img is not None
    m = mask is not None
    if i and not m:
        return show_img(img)
    elif not i and m:
        return show_mask(mask, color_mapping,black)
    elif i and m:
        img_pil = show_img(img)
        mask_pil = show_mask(mask, color_mapping,black)
        return Image.blend(img_pil, mask_pil, alpha=alpha)
    return