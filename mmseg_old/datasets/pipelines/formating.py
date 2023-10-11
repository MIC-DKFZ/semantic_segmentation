# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import warnings

from .formatting import *

warnings.warn(
    "DeprecationWarning: mmseg_old.datasets.pipelines.formating will be "
    "deprecated in 2021, please replace it with "
    "mmseg_old.datasets.pipelines.formatting."
)
