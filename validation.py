import logging
log = logging.getLogger(__name__)
import time
import hydra
from pytorch_lightning import Trainer
import torch
import os
import glob
from Segmentation_Model import SegModel#, validation
from omegaconf import OmegaConf,open_dict, DictConfig
from os.path import relpath
from pytorch_lightning import loggers as pl_loggers
import sys
import numpy as np
from tqdm import tqdm
import argparse
from _utils import hasTrueAttr, hasNotEmptyAttr


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
        #OmegaConf.set_struct(cfg, True)
        #with open_dict(cfg):
        #    cfg.TESTING = {"MS_TESTING": True, "SCALES": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], "FLIP": True}

    #print(cfg)
    #print(x)
    return cfg

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

    parser.add_argument('--valdir', type=str, default="")
    args, hydra_args = parser.parse_known_args()

    validation(args.valdir,hydra_args)

