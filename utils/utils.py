import logging
from pytorch_lightning.utilities import rank_zero_only
logging.basicConfig(level=logging.INFO)
def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def num_gpus(avail_GPUS,selected_GPUS):
    #Transfering pytorch lightning gpu argument into the number of gpus
    #Needed since lightning enables to pass gpu as list or string
    if selected_GPUS in [-1,"-1"]:
        num_gpus = avail_GPUS
    elif selected_GPUS in [0,"0",None]:
        num_gpus = 0
    elif isinstance(selected_GPUS,int):
        num_gpus=selected_GPUS
    elif isinstance(selected_GPUS,list):
        num_gpus=len(selected_GPUS)
    elif isinstance(selected_GPUS, str):
        num_gpus=len(selected_GPUS.split((",")))
    return num_gpus


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