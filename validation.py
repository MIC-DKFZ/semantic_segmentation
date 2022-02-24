import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import argparse
import os
import glob

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from Segmentation_Model import SegModel
from utils.utils import hasTrueAttr, hasNotEmptyAttr

from utils.utils import get_logger
log = get_logger(__name__)

from utils.visualization import show_data

# Inference Time
# https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f

def parse_key_into_cfg(cfg,key,value):
    if "." in key:  #Need for recursion
        key_a,key_b=key.split('.', 1)
        parse_key_into_cfg(cfg[key_a],key_b,value)
    else:
        try:
            cfg[key] = value
        except:
            print("Validation Override: key ", key, " not found")

def get_test_config(cfg):
    cfg.ORG_CWD = ""
    if hasNotEmptyAttr(cfg,"TESTING"):
        if hasNotEmptyAttr(cfg.TESTING,"OVERRIDES"):
            keys=cfg.TESTING.OVERRIDES.keys()
            for key in keys:
                #needed recursice Funciton to catch definiition like cfg.xx.xx ...
                parse_key_into_cfg(cfg,key,cfg.TESTING.OVERRIDES[key])

    return cfg

def validation(ckpt_dir,hydra_args,init=True):

    if init:
        hydra.initialize(config_path="config")

        os.chdir(ckpt_dir)

        ###  load parameters from the checkpoint directory which are overritten ###
        overrides = OmegaConf.load(os.path.join("hydra","overrides.yaml"))
        train_overrides=["MODEL.PRETRAINED=False"]#,"pl_trainer.gpus=-1"]

        ### load local config and first override by the the parameters frm the checkpoint dir
        ### afterward override the parameters from the commandline ###
        cfg = hydra.compose(config_name="baseline", overrides=overrides+hydra_args+train_overrides)

        ### change some testing specific parameters ###
    cfg=get_test_config(cfg)
    #print(cfg)
    ### load checkpoint and load model ###
    ckpt_file = glob.glob(os.path.join("checkpoints", "best_*"))[0]
    log.info("Checkpoint Directory: %s",ckpt_file)
    model = SegModel.load_from_checkpoint(ckpt_file, config=cfg)

    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    callbacks = []
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="",
                                             sub_dir="validation")  # ,default_hp_metric=False)

    trainer_args = getattr(cfg, "pl_trainer") if hasNotEmptyAttr(cfg, "pl_trainer") else {}
    trainer = Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        **trainer_args
    )

    trainer.test(model, dataModule)

def visualisation(ckpt_dir,hydra_args,init=True):
    dir=os.getcwd()
    if init:
        hydra.initialize(config_path="config")

        os.chdir(ckpt_dir)

        ###  load parameters from the checkpoint directory which are overritten ###
        overrides = OmegaConf.load(os.path.join("hydra","overrides.yaml"))
        train_overrides=["MODEL.PRETRAINED=False"]#,"pl_trainer.gpus=-1"]

        ### load local config and first override by the the parameters frm the checkpoint dir
        ### afterward override the parameters from the commandline ###
        cfg = hydra.compose(config_name="baseline", overrides=overrides+hydra_args+train_overrides)

        ### change some testing specific parameters ###
    cfg=get_test_config(cfg)
    #print(cfg)
    ### load checkpoint and load model ###
    ckpt_file = glob.glob(os.path.join("checkpoints", "best_*"))[0]
    #log.info("Checkpoint Directory: %s",ckpt_file)
    model = SegModel.load_from_checkpoint(ckpt_file, config=cfg)

    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
    #print(cfg.datamodule.dataset._target_ )
    #module=__import__(os.path.join(dir,cfg.datamodule.dataset._target_.replace(".","/")))
    #show= getattr(module, "show")

    dataModule.setup()
    dataset=dataModule.DS_val
    print(len(dataset))

    img,mask=dataset[671]

    model.cuda()
    model.eval()
    with torch.no_grad():
        #print(img.shape)
        #print(img.unsqueeze(0).shape)
        pred=model(img.unsqueeze(0).cuda())
        pred=pred["out"]
    #print(mask.shape)
    #print(pred.shape)
    #print(pred.shape)
    def dice(gt, pred):
        num_classes=2
        gt = gt.flatten()  # .detach().cpu()
        pred = pred.argmax(1).flatten()  # .detach().cpu()

        with torch.no_grad():
            k = (gt >= 0) & (gt < num_classes)
            gt = gt[k]
            pred = pred[k]
            # print(pred.shape,gt.shape)
            #gt = torch.nn.functional.one_hot(gt, num_classes)
            #pred = torch.nn.functional.one_hot(pred, num_classes)
            # print(pred.shape, gt.shape)
            #gt = gt[:, 1:]
            #pred = pred[:, 1:]
            # print(pred.shape, gt.shape)
            # dice2 = tmF.dice_score(gt, pred)
            dice = (2 * (gt * pred).sum() + 1e-15) / (gt.sum() + pred.sum() + 1e-15)

            return dice

    print("DICE", dice(mask.cuda(), pred))

    pred=pred.squeeze(0)
    pred=torch.argmax(pred.squeeze(), dim=0).detach().cpu()
    s=torch.zeros(pred.shape,dtype=int)

    pred_1=pred==1
    mask_1=mask==1
    pred_0=pred==0
    mask_0=mask==0

    #print(s.dtype,pred.dtype,pred_0.dtype,pred_1.dtype,mask.dtype,mask_0.dtype,mask_1.dtype)
    s[torch.logical_and(pred_1,mask_1)]=2
    s[torch.logical_and(pred_1,mask_0)]=1
    s[torch.logical_and(pred_0,mask_1)]=3

    tp=torch.logical_and(pred_1,mask_1).sum()
    fp=torch.logical_and(pred_1,mask_0).sum()
    fn=torch.logical_and(pred_0,mask_1).sum()
    tn=torch.logical_and(pred_0,mask_0).sum()
    print("D1",2*tp/(2*tp+fp+fn))
    print("D2",2*tn/(2*tn+fp+fn))
    #s[pred!=mask]=2
    #print(s.shape)
    #print(s.dtype,mask.dtype)


    #print(img.shape)
    #i=show(img=img,mask=s,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #b=show(img=img,mask=mask,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #m=show(mask=mask)
    #p=show(mask=pred.detach().cpu())
    #i.show()
    #b.show()
    #m.show()
    #p.show()





    #callbacks = []
    #for _, cb_conf in cfg.CALLBACKS.items():
    #    if cb_conf is not None:
    #        cb = hydra.utils.instantiate(cb_conf)
    #        callbacks.append(cb)

    #tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="",
    #                                         sub_dir="validation")  # ,default_hp_metric=False)

    #trainer_args = getattr(cfg, "pl_trainer") if hasNotEmptyAttr(cfg, "pl_trainer") else {}
    #trainer = Trainer(
    #    logger=tb_logger,
    #    callbacks=callbacks,
    #    **trainer_args
    #)

    #trainer.test(model, dataModule)

def endocv2022():
    import cv2
    from datasets.EndoCV2022_extra import EndoCV2022_dataset_Test
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import torchvision
    from PIL import Image

    model_ckpt="/home/l727r/Desktop/Target_Folder/EndoCV2022/hrnet_ocr_ms/fold_final_waug__DATASET.FOLD_all__epochs_15__lossfunction_RMI_CE_CE_CE__lr_0.001__metric_Dice_class/2022-02-24_10-40-49/checkpoints/best_epoch_4__imageDice_0.9285.ckpt"
    hydra.initialize(config_path="config")
    overrides = ["model=hrnet_ocr_ms","dataset=EndoCV2022","MODEL.PRETRAINED=False","MODEL.MSCALE_INFERENCE=False","MODEL.PRETRAINED_WEIGHTS=''"]
    cfg = hydra.compose(config_name="baseline", overrides=overrides)
    model = SegModel.load_from_checkpoint(model_ckpt, config=cfg)

    test_path="/home/l727r/Desktop/endocv2022/TestSet1"
    preds=glob.glob(os.path.join(test_path,"EndoCV22_round1_test_seq","*.tif"))

    transforms = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2()])

    test_data="/home/l727r/Desktop/endocv2022/TestSet1/testDataRound-I_DONOTDISTRIBUTE/testDataRound-I/testImagesRound-I"
    outdir = "/home/l727r/Desktop/endocv2022/TestSet1/lars_prediction_testDataRound-I"
    outdir = "/home/l727r/Desktop/endocv2022/TestSet1/overlay"
    test_dataset= EndoCV2022_dataset_Test(test_data,transforms)
    #img=test_dataset[0]
    #out = show_data(img, alpha=0.5, color_mapping=[[0, 0, 0], [255, 0, 0]], mean=[0.485, 0.456, 0.406],
    #                std=[0.229, 0.224, 0.225])
    model.cuda()
    model.eval()
    with torch.no_grad():
        for img,name in test_dataset:
            pred = model(img.unsqueeze(0).cuda())
            pred = pred["out"]
            pred=pred.detach().cpu()
            #pred=torch.argmax(pred.squeeze(), dim=0).detach().cpu()*255
            #print(pred.shape,torch.unique(pred))
            out=show_data(img=img,mask=pred,color_mapping=[[0, 0, 0], [255, 0, 0]], mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            #pred=pred.to(torch.qint8)
            #pred_np=np.uint8(pred)
            #print(pred_np.shape, np.unique(pred_np))
            #img_pil=Image.fromarray(np.uint8(pred))

            #img_pil=torchvision.transforms.ToPILImage(mode="int")(pred)
            name=name.split("/")[-1].split(".")[0]
            #naming=os.path.join(outdir,name+".tif")
            naming=os.path.join(outdir,name+".png")
            #print(naming)
            #img_pil.save(naming)
            out.save(naming)
            #img_pil.show()

            #break





    #with torch.no_grad():
        #print(img.shape)
        #print(img.unsqueeze(0).shape)
        #pred=model(img.unsqueeze(0).cuda())
        #pred=pred["out"]
    #out.show()
    #print(preds)
    #mask = cv2.imread(preds[0], -1)
    #print(mask.shape)
    #print(np.unique(mask))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--valdir', type=str, default="")
    args, hydra_args = parser.parse_known_args()

    endocv2022()
    #validation(args.valdir,hydra_args)
    #visualisation(args.valdir,hydra_args)

