import hydra
from pytorch_lightning import Trainer
import torch
import os
import glob
from main import SegModel
from omegaconf import OmegaConf
from os.path import relpath
from pytorch_lightning import loggers as pl_loggers



def validation(ckpt_path,path):
    #path=path+"validation"

    hydra.initialize(config_path="config")
    #hydra.run.dir = path
    #hydra.initialize(config_dir="config")
    # cfg = hydra.compose(config_name="config",overrides=["MODEL.ADAPTED_PRETRAINED_WEIGHTS=""", "MODEL.PRETRAINED=False","MODEL.MSCALE_TRAINING=true","DATASET.ROOT=/home/l727r/Desktop/Cityscape"])
    # cfg = hydra.compose(config_name="config",overrides=["MODEL.ADAPTED_PRETRAINED_WEIGHTS=""", "MODEL.PRETRAINED=False","DATASET.ROOT=/home/l727r/Desktop/Cityscape"])
    cfg = hydra.compose(config_name="baseline", overrides=["model=hrnet", "dataset=VOC2010_Context",
                                                           "MODEL.ADAPTED_PRETRAINED_WEIGHTS=""",
                                                           "MODEL.PRETRAINED=False",
                                                           "+multiscale=True",
                                                           "+multiscales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]",

                                                           ])
    cfg.val_batch_size =1
    cfg.AUGMENTATIONS.TEST=[{'Compose': {'transforms': [{'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}, {'ToTensorV2': None}]}}]
    print(cfg)

    #x=str("hydra.run.dir="+path)
    #print(x)
    model = SegModel.load_from_checkpoint(ckpt_path, config=cfg)
    # print(cfg.dataset)
    # dataModule = getattr(DataModules, cfg.DATASET.NAME)(config=cfg)
    # try:
    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
    # except:
    # dataModule = getattr(DataModules, "BaseDataModule")(config=cfg)
    #tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="", default_hp_metric=False)
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        precision=16,
        benchmark=True,
        #logger = tb_logger,
    )

    trainer.test(model, dataModule)

def validation_old(ckpt_path,hp_path):

    hydra.initialize(config_path=hp_path)
    #cfg = hydra.compose(config_name="config",overrides=["MODEL.ADAPTED_PRETRAINED_WEIGHTS=""", "MODEL.PRETRAINED=False","MODEL.MSCALE_TRAINING=true","DATASET.ROOT=/home/l727r/Desktop/Cityscape"])
    cfg = hydra.compose(config_name="config",overrides=["MODEL.ADAPTED_PRETRAINED_WEIGHTS=""",
                                                        "MODEL.PRETRAINED=False",
                                                        #"DATASET.ROOT=/home/l727r/Desktop/Cityscape",
                                                        "DATASET.ROOT=/home/l727r/Desktop/Datasets/VOC2010_Context",
                                                        "dataset._target_= datasets.VOC2010_Context.VOC2010_Context_dataset"
    ])
    cfg.val_batch_size=1
    #cfg.AUGMENTATIONS.TEST={'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
    cfg.AUGMENTATIONS.TEST.PadIfNeeded.mask_value=255#{'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
    print(OmegaConf.to_yaml(cfg))
    model = SegModel.load_from_checkpoint(ckpt_path, config=cfg)
    #print(cfg.dataset)
    #dataModule = getattr(DataModules, cfg.DATASET.NAME)(config=cfg)
    #try:
    dataModule = hydra.utils.instantiate(cfg.datamodule,_recursive_=False)
    #except:
    #dataModule = getattr(DataModules, "BaseDataModule")(config=cfg)



    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        precision= 16,
        benchmark= True
    )

    #trainer.test(model,dataModule)


import sys
if __name__ == "__main__":
    path= "/home/l727r/Desktop/Target_Folder/PASCALContext/hrnet/data_augmentations=PASCALContext_epochs=200/2022-01-18_16-05-09/"
    #sys.argv.append('hydra.run.dir="hydra.run.dir=/home/l727r/Desktop/Target_Folder/PASCALContext/hrnet/data_augmentations\=PASCALContext_epochs\=200/2022-01-18_16-05-09"')
    ckpt_path = glob.glob( path + "checkpoints/best_epoch*.ckpt")[0]

    #hp_path=path+"hydra"

    #hp_path=relpath(path+"hydra", '.')
    #print(hp_path)
    validation(ckpt_path,path)

    #print(get_augmentations_from_config(cfg.AUGMENTATIONS.TRAIN))
    #print(hp_path)
    #env = OmegaConf.load("config/environment/local.yaml")
    #hydra.initialize(config_path=hp_path)

    #cfg = hydra.compose(config_name="config")#,overrides=env)#,
    #cfg.roots.Pascal_ctx = "/gpu/data/OE0441/l727r/VOC2010_Context"
    #env=OmegaConf.load("config/environment/local.yaml")
    #env=OmegaConf.structured(env)
    #cfg=OmegaConf.structured(cfg)
    #cfg=OmegaConf.merge(cfg,env)
                        #overrides=["MODEL.ADAPTED_PRETRAINED_WEIGHTS=""", "MODEL.PRETRAINED=False",
                        #           "DATASET.ROOT=/home/l727r/Desktop/Cityscape"])
    #cfg.AUGMENTATIONS.TEST= {'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    #         'ToTensorV2': None}
    #print(os.getcwd())
    #cfg.pop("hydra")
    #print(OmegaConf.to_yaml(cfg))
    #print(cfg.pretty())
    #base = "/home/l727r/Desktop/Target_Folder/Cityscape/hrnet_ocr_ms/"
    #folder = "MODEL.MSCALE_TRAINING=false_epochs=500"
    # folder="MODEL.MSCALE_TRAINING=false_epochs=400_lossfunction=wRMI"
    #x = os.listdir(base + folder)
    # x=x[1:]
    # print(x)
    #for name in x:
        # GlobalHydra.instance().clear()
        # hydra._internal.hydra.GlobalHydra.get_state().clear()
    #    hydra.core.global_hydra.GlobalHydra.instance().clear()
    #    name = os.path.join(folder, name)
    #    print(name)
    #    print("/home/l727r/Desktop/Target_Folder/Cityscape/hrnet_ocr_ms/" + name + "/checkpoints/best_epoch*.ckpt")

     #   ckpt_path = glob.glob(
     #       "/home/l727r/Desktop/Target_Folder/Cityscape/hrnet_ocr_ms/" + name + "/checkpoints/best_epoch*.ckpt")
     #   print(ckpt_path)
     #   hp_path = "../Target_Folder/Cityscape/hrnet_ocr_ms/MODEL.MSCALE_TRAINING=False_epochs=400/2022-01-05_12-06-13/hydra"
        # hp_path="../Target_Folder/Cityscape/hrnet_ocr_ms/"+name+"/hydra"
     #   validation(ckpt_path[-1], hp_path)

    # name="MODEL.MSCALE_TRAINING=False_epochs=400/2022-01-05_12-06-13"
    # "/home/l727r/Desktop/Target_Folder/Cityscape/hrnet_ocr_ms/MODEL.MSCALE_TRAINING=false_epochs=300/2022-01-12_07-50-38"
    # name="MODEL.MSCALE_TRAINING=false_epochs=300/2022-01-13_08-48-59"
    # p="MODEL.MSCALE_TRAINING=false_epochs=250"

    # ckpt_path=glob.glob("/home/l727r/Desktop/Target_Folder/Cityscape/hrnet_ocr_ms/"+name+"/checkpoints/best_epoch*.ckpt")
    # print(ckpt_path)
    # hp_path="../Target_Folder/Cityscape/hrnet_ocr_ms/MODEL.MSCALE_TRAINING=False_epochs=400/2022-01-05_12-06-13/hydra"
    # hp_path="../Target_Folder/Cityscape/hrnet_ocr_ms/"+name+"/hydra"
    # validation(ckpt_path[-1],hp_path)

