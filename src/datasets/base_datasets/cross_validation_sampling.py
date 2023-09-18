from src.datasets.base_datasets.sampling import SamplingDataset
from src.datasets.base_datasets.cross_validation import CVDataset


class CVSamplingDataset(SamplingDataset, CVDataset):
    pass
