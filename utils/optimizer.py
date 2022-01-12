import torch

def get_optimizer_from_cfg(parameters,cfg):
    if cfg.optimizer in ["sgd","SGD","Stochastic Gradient Descent"]:
        return torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum,
                        weight_decay=cfg.wd)
    elif cfg.optimizer in  ["adam","Adam","ADAM"]:
        return torch.optim.Adam(parameters, lr=1e-3)
    return