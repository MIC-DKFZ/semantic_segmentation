import logging
log = logging.getLogger(__name__)
import time
import hydra
from pytorch_lightning import Trainer
import torch
import os
import glob
from main import SegModel#, validation
from omegaconf import OmegaConf,open_dict, DictConfig
from os.path import relpath
from pytorch_lightning import loggers as pl_loggers
import sys
import numpy as np
from tqdm import tqdm
import argparse
from config.utils import hasTrueAttr, hasNotEmptyAttr


#TESTING
#    TEST_AFTERWARDS
#    SCALES
#    FLIP



def get_test_config(cfg):

    cfg.MODEL.ADAPTED_PRETRAINED_WEIGHTS=""

    if (cfg.DATASET.NAME == "VOC2010_Context" or cfg.DATASET.NAME == "VOC2010_Context_60") and cfg.MODEL.NAME == "hrnet_ocr_ms":
        cfg.MODEL.MSCALE_TRAINING = True
        cfg.MODEL.N_SCALES= [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        cfg.val_batch_size = 1#

        cfg.AUGMENTATIONS.TEST = [{'Compose': {
            'transforms': [{'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
                           {'ToTensorV2': None}]}}]

    if cfg.MODEL.NAME == "hrnet_ocr_ms":
        cfg.MODEL.MSCALE_TRAINING = True
        cfg.val_batch_size = 1

    if cfg.DATASET.NAME == "VOC2010_Context" or cfg.DATASET.NAME == "VOC2010_Context_60":
        cfg.val_batch_size = 1
        #cfg.AUGMENTATIONS.TEST = [{'Compose': {
        #    'transforms': [{'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
        #                   {'ToTensorV2': None}]}}]



        #cfg.TESTING= {"MS_TESTING": True, "SCALES": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], "FLIP": True}
        #x=OmegaConf.create({"a": {"b": 10}})
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.TESTING = {"MS_TESTING": True, "SCALES": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], "FLIP": True}

    print(cfg)
    #print(x)
    return cfg

def extra_testing(path):
        print(path)
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        print(os.getcwd())


        #hydra.initialize(config_path="config")

        #overrides=OmegaConf.load('hydra/overrides.yaml')
        config=OmegaConf.load('hydra/config.yaml')

        if config.MODEL.NAME == "hrnet_ocr_ms":
            config.MODEL.MSCALE_TRAINING = True

        if config.DATASET.NAME == "VOC2010_Context" or config.DATASET.NAME == "VOC2010_Context_60":
            config.val_batch_size = 1
            config.MODEL.PRETRAINED = False
            config.AUGMENTATIONS.TEST = [{'Compose': {
                'transforms': [{'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
                               {'ToTensorV2': None}]}}]
            # OmegaConf.update(cfg, "multiscale", {"zonk": 30})
            config.extra_testing = {"multiscale": True, "scales": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], "flip": True}

        cfg=config

        ckpt_path = glob.glob("checkpoints/best*")
        if ckpt_path != []:
            model = SegModel.load_from_checkpoint(ckpt_path[0], config=cfg)
        else:
            model = SegModel(config=cfg)
        dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

        callbacks = []
        for _, cb_conf in cfg.CALLBACKS.items():
            if cb_conf is not None:
                cb = hydra.utils.instantiate(cb_conf)
                callbacks.append(cb)

        trainer = Trainer(
            gpus=torch.cuda.device_count(),
            callbacks=callbacks,
            logger=None,
            # pl_loggers.TensorBoardLogger(save_dir="./", name="validation", version="", default_hp_metric=False),
            #enable_checkpointing=False,
            enable_progress_bar= False,

            precision=16,
            benchmark=True,

            sync_batchnorm=True,
            strategy='ddp' if torch.cuda.device_count() > 1 else None,
        )

        trainer.test(model, dataModule)

        '''if hasTrueAttr(cfg, "extra_testing"):
        #log.info("#####START EXTRA TESTING EPOCH#####")

        cfg=get_test_config(cfg)

        ckpt_path=glob.glob("checkpoints/best*")
        if ckpt_path!=[]:
            model = SegModel.load_from_checkpoint(ckpt_path[0], config=cfg)
        else:
            model = SegModel(config=cfg)
        dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

        trainer = Trainer(
            gpus=AVAIL_GPUS,

            logger=None,#pl_loggers.TensorBoardLogger(save_dir="./", name="validation", version="", default_hp_metric=False),
            enable_checkpointing = False,

            precision=16,
            benchmark= True,

            sync_batchnorm=True,
            strategy='ddp' if AVAIL_GPUS > 1 else None,
        )

        trainer.test(model, dataModule)'''

def validation2(path):
    #path=path+"validation"
    os.chdir(path)

    ckpt_path = glob.glob(path + "checkpoints/best_epoch*.ckpt")[0]

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

@hydra.main(config_path="config", config_name="baseline")
def inference_time(cfg: DictConfig):
    #https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
    #import time
    device = torch.device("cuda")
    runs=5
    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
    dataModule.setup()
    data_loader=dataModule.test_dataloader()

    if cfg.MODEL.NAME=="hrnet":
        #ckpt_path="~/Desktop/Target_Folder/Cityscape/hrnet/epochs=400/2021-12-25_22-43-52_environment=cluster_epochs=400/checkpoints/best_epoch=394_mIoU=0.8092.ckpt"
        ckpt_path = "~/Desktop/Semantic_Segmentation/time/Cityscapes/hrnet/baseline__batch_size_4__epochs_5__val_batch_size_4/2022-01-24_17-31-33/checkpoints/epoch=4-step=3714.ckpt"
        #Time Mean: 55.390916
        #BS=8:  57.698148

    elif cfg.MODEL.NAME == "hrnet_ocr":
        #ckpt_path = "~/Desktop/Target_Folder/Cityscape/hrnet_ocr/epochs=400/2022-01-08_14-25-53/checkpoints/best_epoch=383_mIoU=0.8123.ckpt"
        #Time Mean: 79.570620
        #ckpt_path = "~/Desktop/Semantic_Segmentation/time/Cityscapes/hrnet_ocr/baseline__batch_size_4__epochs_5__val_batch_size_4/2022-01-24_19-12-20/checkpoints/epoch=4-step=3714.ckpt"
        #Time Mean: 79.858021
        ckpt_path = "~/Desktop/Semantic_Segmentation/time/Cityscapes/hrnet_ocr/baseline__batch_size_4__epochs_5__pl_trainer.precision_32__val_batch_size_4/2022-01-24_18-30-25/checkpoints/epoch=4-step=3714.ckpt"
        #Time Mean: 79.848933
        #BS=6:Time Mean: 82.034597
    elif cfg.MODEL.NAME == "hrnet_ocr_ms":
        ckpt_path = "~/Desktop/Semantic_Segmentation/time/Cityscapes/hrnet_ocr_ms/baseline__MODEL.MSCALE_TRAINING_False__batch_size_4__epochs_5__val_batch_size_4/2022-01-24_23-01-39/checkpoints/epoch=4-step=3714.ckpt"
        #Time Mean: 111.868603  MODEL.MSCALE_TRAINING=False
        #465.615138
        #BS=4: Time Mean: 112.682108
        #BS=1 HR Time Mean: 466.565919

    elif cfg.MODEL.NAME == "hrnet_ocr_aspp":
        ckpt_path = "~/Desktop/Semantic_Segmentation/time/Cityscapes/hrnet_ocr_aspp/baseline__batch_size_4__epochs_5__val_batch_size_4/2022-01-24_20-24-39/checkpoints/epoch=4-step=3714.ckpt"
        #BS=3  147.505566
    model = SegModel.load_from_checkpoint(ckpt_path, config=cfg)
    '''
    Ich bin gerade noch dabei das Feetback zu meiner Cityscapes Präsentation von letzter Woche umzusetzen und habe eine Frage wie ich die time complexity am besten angebe. 
    Ist low priority also schaus die an wenn du Zeit dafür hast.
    
    Ich habe die inference time bestimmt und bin mir jetzt unsicher wie ich die training time definieren soll.
    Bis jetzt hatte ich training und validation time seperat angegeben, also t_train = reine trainings zeit und t_val = zeit fürs validieren
    Denkst du es wäre besser die training time als t_train beizubehalten oder training time als die zeit für einen epoch zu nehmen (also t_train + t_val)?
    
    '''

    model.to(device)
    model.eval()
    t=[]
    print(np.mean([1,2,3]))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i in range(0,runs):
            log.info("Starting Run number: %i", i)
            #start=time.time()
            start.record()
            for i,batch in enumerate(tqdm(data_loader)):

                x, y_gt = batch
                _=model(x.to(device))

            end.record()
            torch.cuda.synchronize()
            t_total = start.elapsed_time(end) / 1000

            #t_total=time.time()-start
            log.info("Time: %f",t_total)
            t.append(t_total)
    log.info("Time Mean: %f",np.mean(t))

def validation(ckpt_dir,hydra_args):

    hydra.initialize(config_path="config")

    os.chdir(ckpt_dir)

    overrides = OmegaConf.load(os.path.join("hydra","overrides.yaml"))

    train_overrides=["MODEL.PRETRAINED=False"]

    cfg = hydra.compose(config_name="baseline", overrides=overrides+hydra_args+train_overrides)

    cfg=get_test_config(cfg)
    #return
    ckpt_file = glob.glob(os.path.join("checkpoints", "best_*"))[0]

    model = SegModel.load_from_checkpoint(ckpt_file, config=cfg)

    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="",
                                             sub_dir="validation")  # ,default_hp_metric=False)

    trainer_args = getattr(cfg, "pl_trainer") if hasNotEmptyAttr(cfg, "pl_trainer") else {}
    trainer = Trainer(
        gpus=torch.cuda.device_count(),

        logger=tb_logger,

        **trainer_args
    )

    trainer.test(model, dataModule)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser = parser.ArgumentParser(...)
    parser.add_argument('--valdir', type=str, default="")
    args, hydra_args = parser.parse_known_args()

    #print(args)
    #print(unknown)
    #args = parser.parse_args()

    #sys.argv.append('hydra.run.dir='+args.dir)

    #chpt_path = "/home/l727r/Desktop/Target_Folder/Cityscapes/hrnet_ocr_ms/rmi_coarse__MODEL.MSCALE_TRAINING_False__epochs_5__lossfunction_wRMI,wCE,wCE,wCE__lr_0.001/2022-01-28_11-59-25/"
    validation(args.valdir,hydra_args)
    #inference_time()



    #path= "/home/l727r/Desktop/Target_Folder/PASCALContext/hrnet/data_augmentations=PASCALContext_epochs=200/2022-01-18_16-05-09/"
    #sys.argv.append('hydra.run.dir="hydra.run.dir=/home/l727r/Desktop/Target_Folder/PASCALContext/hrnet/data_augmentations\=PASCALContext_epochs\=200/2022-01-18_16-05-09"')
    #ckpt_path = glob.glob( path + "checkpoints/best_epoch*.ckpt")[0]

    #hp_path=path+"hydra"

    #hp_path=relpath(path+"hydra", '.')
    #print(hp_path)
    #validation2(path)

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

