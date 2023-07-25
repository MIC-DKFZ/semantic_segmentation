from datasets.Base_Datasets.sampling import Sampling_Dataset
from datasets.Base_Datasets.cross_validation import CV_Dataset


class CV_Sampling_Dataset(Sampling_Dataset, CV_Dataset):
    pass
