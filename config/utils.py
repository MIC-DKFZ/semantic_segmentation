from omegaconf import DictConfig, OmegaConf

def hasTrueAttr(obj,attr):
    if hasattr(obj,attr):
        if obj[attr]:
            return True
    return False

def hasNotEmptyAttr(obj,attr):
    if hasattr(obj,attr):
        if obj[attr]!=None:
            return True
    return False

def get_config(file_hp="baseline.yaml"):

    cfg = OmegaConf.load(file_hp)
    cfg_model = OmegaConf.load(cfg.MODEL.FILE)
    cfg_data = OmegaConf.load(cfg.DATASET.FILE)
    cfg = OmegaConf.merge(cfg_data, cfg_model, cfg)

    return cfg

def print_config(cfg):
    print(OmegaConf.to_yaml(cfg))

def include_args(cfg,args):
    cfg=OmegaConf.merge(cfg,OmegaConf.from_dotlist(args))
    return cfg


'''
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
from yacs.config import CfgNode as CN

def hasTrueAttr(obj,attr):
    if hasattr(obj,attr):
        if obj[attr]:
            return True
    return False

def hasNotEmptyAttr(obj,attr):
    if hasattr(obj,attr):
        if obj[attr]!=None:
            return True
    return False

def convert_to_dict(cfg_node, key_list):
    if not isinstance(cfg_node, CN):

        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict

def update_config(cfg,file_hp="config_test/hyperparameters.yaml"):
    cfg.defrost()
    #loading hyperparameters for training
    cfg.merge_from_file(file_hp)

    #loading model parameters
    model_file=cfg.MODEL.FILE
    cfg.merge_from_file(model_file)

    # loading dataset parameters
    dataset_file = cfg.DATASET.FILE
    cfg.merge_from_file(dataset_file)

    cfg.freeze()

def get_config(file_hp="config_test/hyperparameters.yaml"):

    cfg=CN(new_allowed=True)
    #cfg.LOGDIR
    cfg.MODEL =CN(new_allowed=True)
    cfg.DATASET = CN(new_allowed=True)
    cfg.TRAIN = CN(new_allowed=True)
    #cfg.TEST = CN(new_allowed=True)

    update_config(cfg,file_hp)

    return cfg

#import sys
#import hydra
#from omegaconf import DictConfig, OmegaConf


#def my_app(cfg : DictConfig) -> None:
#    #print(cfg.MODEL.PRETRAINED)
#    print(OmegaConf.to_yaml(cfg))

#if __name__ == "__main__":
#    #sys.argv.append('hydra.run.dir=baseline.yaml')
#    dir="baseline.yaml"
#    #hydra.main(config_path=".", config_name=dir)#
#
#
#   hydra.initialize(config_path=".", job_name="utils")
#    cfg = hydra.compose(config_name=dir)
#    my_app(cfg)
'''