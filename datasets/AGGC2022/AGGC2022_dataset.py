import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import zarr
import json
import pickle as pkl
from utils.utils import get_logger
from numcodecs import blosc
from threadpoolctl import threadpool_limits
import cv2

cv2.setNumThreads(0)
blosc.use_threads = False
log = get_logger(__name__)


def get_dataset(
    data_dir,
    split="train",
    fold=0,
    sampling="box",
    NumSamplesPerSubject=1000,
    RandomProb=0.2,
    # Class_Prob={"1": 0.125, "2": 0.125, "3": 0.25, "4": 0.25, "5": 0.25},
    # Class_Prob={"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.8},
    # Class_Prob={"1": 1, "2": 1, "3": 1, "4": 1, "5": 4},
    Class_Prob={"1": 0.75, "2": 0.75, "3": 1, "4": 1, "5": 1.5},
    PatchSize=512,
    Subset=None,
    transforms=None,
    sampling_strategy=0,
):
    # train_folds = ["Split1.txt", "Split2.txt", "Split3.txt", "Split4.txt", "Split5.txt"]
    train_folds = ["Split1.json", "Split2.json", "Split3.json", "Split4.json", "Split5.json"]
    validation_fold = [train_folds.pop(4 - fold)]
    if split == "train":
        if sampling_strategy == 0:
            Class_Prob = {"1": 0.75, "2": 0.75, "3": 1, "4": 1, "5": 8}
            Cases = GetTrainFold(os.path.join(data_dir, "Splits"), train_folds)
            if Subset is not None:
                Cases = [case for case in Cases if Subset in case]
            Cases = [case[1:] if case.startswith("/") else case for case in Cases]
            dataset = ChallengeDataset_point_sampling(
                data_dir,
                Cases,
                NumSamplesPerSubject,
                RandomProb,
                Class_Prob,
                PatchSize,
                transforms,
            )
        elif sampling_strategy == 1:
            Class_Prob = {"1": 0.75, "2": 0.75, "3": 1, "4": 1, "5": 1.5}
            Cases = GetTrainFold_2(os.path.join(data_dir, "New_Splits"), train_folds)
            if Subset is not None:
                Cases = {case: [e for e in elem if Subset in e[0]] for case, elem in Cases.items()}
            Cases = {
                case: [[e[0][1:], e[1]] for e in elem if e[0].startswith("/")]
                for case, elem in Cases.items()
            }
            dataset = ChallengeDataset_point_sampling_new(
                data_dir,
                Cases,
                NumSamplesPerSubject,
                RandomProb,
                Class_Prob,
                PatchSize,
                transforms,
            )
        log.info(
            "AGGC2022 Train Datasets for Fold {} with lenght {} and {} sampling".format(
                fold, len(dataset), sampling_strategy
            )
        )
        return dataset
        """# Cases = GetTrainFold(os.path.join(data_dir, "Splits"), train_folds)
        Cases = GetTrainFold_2(os.path.join(data_dir, "New_Splits"), train_folds)
        # print(Cases)
        # Cases = ["/Subset1/imgs/Subset1_Train_1.zarr"]
        # print(Cases)
        if Subset is not None:
            # Cases = [case for case in Cases if Subset in case]
            Cases = {case: [e for e in elem if Subset in e[0]] for case, elem in Cases.items()}
        # Cases = [case[1:] if case.startswith("/") else case for case in Cases]
        Cases = {
            case: [[e[0][1:], e[1]] for e in elem if e[0].startswith("/")]
            for case, elem in Cases.items()
        }
        # print(Cases)
        # quit()
        if sampling == "box":
            dataset = ChallengeDataset(
                data_dir, Cases, NumSamplesPerSubject, RandomProb, Class_Prob, PatchSize, transforms
            )
        elif sampling == "point":
            if sampling_strategy == 1:
                dataset = ChallengeDataset_point_sampling_new(
                    data_dir,
                    Cases,
                    NumSamplesPerSubject,
                    RandomProb,
                    Class_Prob,
                    PatchSize,
                    transforms,
                )
            else:
                if sampling_strategy == 0:
                    dataset = ChallengeDataset_point_sampling_new(
                        data_dir,
                        Cases,
                        NumSamplesPerSubject,
                        RandomProb,
                        Class_Prob,
                        PatchSize,
                        transforms,
                    )
        # elif sampling == "both":
        #    dataset = ChallengeDataset_box_point_sampling(
        #        data_dir, Cases, NumSamplesPerSubject, RandomProb, Class_Prob, PatchSize, transforms
        #    )
        # log.info(
        #    "AGGC2022 Train Datasets for Fold {} with lenght {} and {} sampling".format(
        #        fold, len(dataset), sampling
        #    )
        # )
        return dataset
        """
    elif split == "val" or split == "test":

        if sampling_strategy == 0:
            Cases = GetTrainFold(os.path.join(data_dir, "Splits"), validation_fold)
        elif sampling_strategy == 1:
            Cases = GetTrainFold_2(os.path.join(data_dir, "New_Splits"), validation_fold)
            Cases = [[e[0] for e in elem] for val, elem in Cases.items()]
            Cases = set(
                [element.replace(".png", ".zarr") for sublist in Cases for element in sublist]
            )
        if Subset is not None:
            Cases = [case for case in Cases if Subset in case]
        Cases = [case[1:] if case.startswith("/") else case for case in Cases]
        dataset = ChallengeDataset_Validation(data_dir, Cases, PatchSize, transforms)
        log.info(
            "AGGC2022 Validation Datasets for Fold {} with lenght {}".format(fold, len(dataset))
        )
        return dataset
        """# Cases = GetTrainFold(os.path.join(data_dir, "Splits"), validation_fold)
        Cases = GetTrainFold_2(os.path.join(data_dir, "New_Splits"), validation_fold)
        # print(Cases)
        Cases = [[e[0] for e in elem] for val, elem in Cases.items()]
        Cases = set([element.replace(".png", ".zarr") for sublist in Cases for element in sublist])
        # print(Cases)
        # print(len(Cases))
        # quit()
        # Cases = ["/Subset1/imgs/Subset1_Train_1.zarr"]
        if Subset is not None:
            Cases = [case for case in Cases if Subset in case]
        Cases = [case[1:] if case.startswith("/") else case for case in Cases]
        dataset = ChallengeDataset_Validation(data_dir, Cases, PatchSize, transforms)
        log.info(
            "AGGC2022 Validation Datasets for Fold {} with lenght {}".format(fold, len(dataset))
        )
        return dataset
        """


def GetTrainFold(BaseFoldPath, CaseList):
    Cases = []
    for fold in CaseList:
        FoldPath = os.path.join(BaseFoldPath, fold.replace(".json", ".txt"))
        SplitFile = open(FoldPath, "r")
        SplitContent = SplitFile.read()
        SplitFold = SplitContent.split("\n")
        SplitFold = SplitFold[:-1]
        Cases += SplitFold
    return Cases


def GetTrainFold_2(BaseFoldPath, CaseList, val=False):

    stroma_cases = []
    normal_cases = []
    g3_cases = []
    g4_cases = []
    g5_cases = []
    for fold in CaseList:
        FoldPath = os.path.join(BaseFoldPath, fold)
        with open(FoldPath) as case_file:
            Split = json.load(case_file)
        stroma_cases = stroma_cases + Split["Stroma"]
        normal_cases = normal_cases + Split["Normal"]
        g3_cases = g3_cases + Split["G3"]
        g4_cases = g4_cases + Split["G4"]
        g5_cases = g5_cases + Split["G5"]

    if val == True:
        from operator import itemgetter

        val_cases = stroma_cases + normal_cases + g3_cases + g4_cases + g5_cases
        val_cases = list(map(itemgetter(0), val_cases))
        Cases = list(np.unique(val_cases))
    else:
        Cases = {}
        Cases["Stroma"] = stroma_cases
        Cases["Normal"] = normal_cases
        Cases["G3"] = g3_cases
        Cases["G4"] = g4_cases
        Cases["G5"] = g5_cases

    return Cases


class ChallengeDataset(Dataset):
    def __init__(
        self,
        data_dir,
        Cases,
        NumSamplesPerSubject,
        RandomProb,
        Class_Prob,
        PatchSize,
        transform=None,
    ):
        self.data_dir = data_dir
        self.Cases = Cases
        SubjectCount = len(self.Cases)
        self.NumPatches = list(range(1, SubjectCount * NumSamplesPerSubject + 1))

        self.RandomProb = RandomProb
        self.NumSamplesPerSubject = NumSamplesPerSubject
        self.PatchSize = PatchSize
        self.ClassProb = Class_Prob
        self.transform = transform

    def __len__(self):
        return len(self.NumPatches)

    def __getitem__(self, item):
        item += 1
        WSIPath = self.Cases[
            int(np.ceil(item / self.NumSamplesPerSubject)) - 1
        ]  # assign WSI subject (1-100) to samples between 0-500000

        # ImgPath = self.data_dir + WSIPath
        ImgPath = os.path.join(self.data_dir, WSIPath)
        Img = zarr.open(ImgPath, mode="r")
        Mask = zarr.open(ImgPath.replace("imgs", "masks"), mode="r")
        Prob = np.random.random(1)  # Determine if we sample random patches or from bounding box

        if Prob.item() < self.RandomProb:
            x_pos = np.random.randint(0, Img.shape[0] - self.PatchSize)
            y_pos = np.random.randint(0, Img.shape[1] - self.PatchSize)
            Extracted_Patch = Img[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize, :]
            Extracted_Mask = Mask[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize]

        else:
            BoxFile = open(ImgPath.replace("imgs", "boxes").replace(".zarr", ".json"))  # +'.json')
            BoundingBoxes = json.load(BoxFile)
            PresentClasses = set(BoundingBoxes.keys())
            Probs = {key: self.ClassProb[key] for key in self.ClassProb.keys() & PresentClasses}
            classes = list(Probs.keys())
            probs = list(Probs.values())
            probs = probs / np.sum(probs)
            choice = np.random.choice(classes, p=probs)
            Boxes = BoundingBoxes[choice]

            areaList = []

            for box in Boxes:
                area = (box[2] - box[0]) * (box[3] - box[1])
                areaList.append(area)

            areaList = areaList / np.sum(areaList)
            BoxCords = Boxes[np.random.choice(list(range(0, len(Boxes))), p=areaList)]

            if (BoxCords[2] - BoxCords[0]) > self.PatchSize:
                x_pos = np.random.randint(
                    BoxCords[0] - int(self.PatchSize / 2),
                    BoxCords[2] - int(self.PatchSize / 2),
                )
            else:
                if BoxCords[2] == BoxCords[0]:
                    x_pos = BoxCords[0] - int(self.PatchSize / 2)
                else:
                    x_pos = np.random.randint(BoxCords[0] - int(self.PatchSize / 2), BoxCords[2])

            if (BoxCords[3] - BoxCords[1]) > self.PatchSize:
                y_pos = np.random.randint(
                    BoxCords[1] - int(self.PatchSize / 2),
                    BoxCords[3] - int(self.PatchSize / 2),
                )
            else:
                if BoxCords[3] == BoxCords[1]:
                    y_pos = BoxCords[3] - int(self.PatchSize / 2)
                else:
                    y_pos = np.random.randint(BoxCords[1] - int(self.PatchSize / 2), BoxCords[3])

            # Check if patch is still in WSI
            if not 0 <= x_pos < (Img.shape[0] - self.PatchSize):
                if not 0 <= x_pos:
                    x_pos = 0
                else:
                    x_pos = Img.shape[0] - self.PatchSize

            if not 0 <= y_pos < (Img.shape[1] - self.PatchSize):
                if not 0 <= y_pos:
                    y_pos = 0
                else:
                    y_pos = Img.shape[1] - self.PatchSize

            Extracted_Patch = Img[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize, :]
            Extracted_Mask = Mask[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize]

        if self.transform:
            augmented = self.transform(image=Extracted_Patch, mask=Extracted_Mask)

            Extracted_Patch = augmented["image"].type(torch.float)
            Extracted_Mask = augmented["mask"]
            Extracted_Mask = Extracted_Mask.type(torch.long)
        subset = WSIPath.rsplit("/", 1)[1].split("_", 1)[0]
        # print(subset)
        return Extracted_Patch, Extracted_Mask, subset


class ChallengeDataset_Validation(Dataset):
    def __init__(
        self,
        data_dir,
        Cases,
        PatchSize,
        transform=None,
    ):
        self.cases = Cases
        self.PatchSize = PatchSize
        self.transform = transform
        self.data_dir = data_dir
        self.NumPatches_ProWSI = []
        self.x_loc = np.array(0)
        self.y_loc = np.array(0)
        self.case_select = np.array(0)
        for subject in self.cases:
            idx = self.cases.index(subject)
            # Loading selected zarr WSI

            ImgPath = os.path.join(self.data_dir, subject)
            # ImgPath = self.data_dir+ subject
            self.Img = zarr.open(ImgPath, mode="r")
            # self.Mask = zarr.open(ImgPath.replace("imgs", "masks"), mode="r")
            self.Mask = zarr.open(ImgPath.replace("imgs", "masks_2"), mode="r")
            # Create grid over WSI
            Shape = self.Img.shape
            x_steps = np.array(range(np.uint8(np.floor(Shape[0] / PatchSize))))
            y_steps = np.array(range(np.uint8(np.floor(Shape[1] / PatchSize))))
            xx, yy = np.meshgrid(x_steps, y_steps)
            num_patches = np.reshape(xx, -1).shape[0]

            # Get all grid positions in x and y vector
            self.NumPatches_ProWSI.append(num_patches)
            self.x_loc = np.concatenate([self.x_loc, np.reshape(xx, -1)], axis=None)
            self.y_loc = np.concatenate([self.y_loc, np.reshape(yy, -1)], axis=None)
            self.case_select = np.concatenate(
                [self.case_select, np.repeat(idx, num_patches)], axis=None
            )

        self.x_loc = self.x_loc[1:]
        self.y_loc = self.y_loc[1:]
        self.case_select = self.case_select[1:]

    def __len__(self):
        return np.sum(self.NumPatches_ProWSI)

    def __getitem__(self, item):
        with threadpool_limits(limits=1, user_api=None):
            # Calculate patch position
            item_pos = [
                ((self.x_loc[item]) * self.PatchSize),
                ((self.x_loc[item] + 1) * self.PatchSize),
                ((self.y_loc[item]) * self.PatchSize),
                ((self.y_loc[item] + 1) * self.PatchSize),
            ]

            # Select correct WSI
            subject = self.cases[self.case_select[item]]
            self.Img = zarr.open(os.path.join(self.data_dir, subject), mode="r")
            self.Mask = zarr.open(
                # os.path.join(self.data_dir, subject.replace("imgs", "masks")),
                # mode="r"
                os.path.join(self.data_dir, subject.replace("imgs", "masks_2")),
                mode="r",
            )

            # Get patch from WSI and mask
            Extracted_Patch = self.Img[
                item_pos[0] : item_pos[1],
                item_pos[2] : item_pos[3],
                :,
            ]

            Extracted_Mask = self.Mask[item_pos[0] : item_pos[1], item_pos[2] : item_pos[3]]

            # # If patch can not be fully cut out, pad to the right and lower side
            # if (ep_shape[0] < self.PatchSize) or (ep_shape[1] < self.PatchSize):
            #     pad = np.full((self.PatchSize, self.PatchSize, 3), 0)
            #     pad[0 : ep_shape[0], 0 : ep_shape[1], :] = Extracted_Patch
            #     Extracted_Patch = pad

            #     pad = np.full((self.PatchSize, self.PatchSize, 3), 255)
            #     pad[0 : ep_shape[0], 0 : ep_shape[1], :] = Extracted_Mask
            #     Extracted_Mask = pad

            # Apply transformations to padded image (Only to tensor for valdiation!)
            if self.transform:
                augmented = self.transform(image=Extracted_Patch, mask=Extracted_Mask)

                Extracted_Patch = augmented["image"].type(torch.float)
                Extracted_Mask = augmented["mask"]
                Extracted_Mask = Extracted_Mask.type(torch.long)

            subset = subject.rsplit("/", 1)[1].split("_", 1)[0]
            # print(subset)
            # print(Extracted_Patch.shape, Extracted_Mask.shape, subset)
        return Extracted_Patch, Extracted_Mask, subset


class ChallengeDataset_point_sampling(Dataset):
    def __init__(
        self,
        data_dir,
        Cases,
        NumSamplesPerSubject,
        RandomProb,
        Class_Prob,
        PatchSize,
        transform=None,
    ):
        self.data_dir = data_dir
        self.Cases = Cases
        SubjectCount = len(self.Cases)
        self.NumPatches = list(range(1, SubjectCount * NumSamplesPerSubject + 1))

        self.RandomProb = RandomProb
        self.NumSamplesPerSubject = NumSamplesPerSubject
        self.PatchSize = PatchSize
        self.ClassProb = Class_Prob
        self.transform = transform

    def __len__(self):
        return len(self.NumPatches)

    def __getitem__(self, item):
        item += 1
        WSIPath = self.Cases[
            int(np.ceil(item / self.NumSamplesPerSubject)) - 1
        ]  # assign WSI subject (1-100) to samples between 0-500000
        # ImgPath = self.data_dir + WSIPath
        ImgPath = os.path.join(self.data_dir, WSIPath)
        Img = zarr.open(ImgPath, mode="r")
        # Mask = zarr.open(ImgPath.replace("imgs", "masks"), mode="r")
        Mask = zarr.open(ImgPath.replace("imgs", "masks_2"), mode="r")
        Prob = np.random.random(1)  # Determine if we sample random patches or from bounding box

        if Prob.item() < self.RandomProb:
            x_pos = np.random.randint(0, Img.shape[0] - self.PatchSize)
            y_pos = np.random.randint(0, Img.shape[1] - self.PatchSize)
            Extracted_Patch = Img[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize, :]
            Extracted_Mask = Mask[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize]

        else:
            PointFile = open(
                # ImgPath.replace("imgs", "sample_points").replace(".zarr", ".pkl"), "rb"
                ImgPath.replace("imgs", "sample_points_2").replace(".zarr", ".pkl"),
                "rb",
            )  # +'.json')
            SamplePoints = pkl.load(PointFile)
            PresentClasses = set(SamplePoints.keys())
            Probs = {key: self.ClassProb[key] for key in self.ClassProb.keys() & PresentClasses}
            classes = list(Probs.keys())
            probs = list(Probs.values())
            probs = probs / np.sum(probs)
            choice = np.random.choice(classes, p=probs)
            Points = SamplePoints[choice]
            point_id = np.random.randint(0, len(Points[0]))
            x_pos = Points[0][point_id]
            y_pos = Points[1][point_id]

            # I want the selected point to be in the center this means the distance to the patch border
            # shouldn't be larger than 0.25 or the patch size
            x_pos = np.random.randint(x_pos - self.PatchSize * 0.9, x_pos - self.PatchSize * 0.1)
            y_pos = np.random.randint(y_pos - self.PatchSize * 0.9, y_pos - self.PatchSize * 0.1)

            # Check if patch is still in WSI
            if not 0 <= x_pos < (Img.shape[0] - self.PatchSize):
                if not 0 <= x_pos:
                    x_pos = 0
                else:
                    x_pos = Img.shape[0] - self.PatchSize

            if not 0 <= y_pos < (Img.shape[1] - self.PatchSize):
                if not 0 <= y_pos:
                    y_pos = 0
                else:
                    y_pos = Img.shape[1] - self.PatchSize

            Extracted_Patch = Img[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize, :]
            Extracted_Mask = Mask[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize]

        if self.transform:
            augmented = self.transform(image=Extracted_Patch, mask=Extracted_Mask)

            Extracted_Patch = augmented["image"].type(torch.float)
            Extracted_Mask = augmented["mask"]
            Extracted_Mask = Extracted_Mask.type(torch.long)

        subset = WSIPath.rsplit("/", 1)[1].split("_", 1)[0]
        return Extracted_Patch, Extracted_Mask, subset


class ChallengeDataset_point_sampling_new(Dataset):
    def __init__(
        self,
        data_dir,
        Cases,
        NumSamplesPerSubject,
        RandomProb,
        Class_Prob,
        PatchSize,
        transform=None,
    ):
        self.data_dir = data_dir
        self.Cases = Cases
        # print(Cases)
        x = [[e[0] for e in elem] for val, elem in Cases.items()]
        x = set([element for sublist in x for element in sublist])
        # print(x)
        # print(len(x))
        # quit()
        # SubjectCount = len(self.Cases)
        self.NumPatches = len(x) * NumSamplesPerSubject

        self.RandomProb = RandomProb
        # self.NumSamplesPerSubject = NumSamplesPerSubject
        self.PatchSize = PatchSize
        self.ClassProb = Class_Prob
        self.transform = transform
        self.LookUpDict = {"Stroma": "1", "Normal": "2", "G3": "3", "G4": "4", "G5": "5"}

    def __len__(self):
        return self.NumPatches

    def __getitem__(self, item):
        item += 1

        classes = list(self.Cases.keys())
        probs = list(self.ClassProb.values())
        probs = probs / np.sum(probs)
        choice = np.random.choice(classes, p=probs)  # sth. between Normal, Stroma,G3,G4,G5
        cases_containing_class = self.Cases[choice]
        # print("A", classes, probs, choice)
        # print("B", cases_containing_class)
        # print("C", self.Cases)

        probs_for_case = list(pixel_area[1] for pixel_area in cases_containing_class)
        probs_for_case = probs_for_case / np.sum(probs_for_case)
        case_number = np.random.choice(
            list(range(0, len(cases_containing_class))), p=probs_for_case
        )
        case = cases_containing_class[case_number]
        case_path = case[0].replace(".png", ".zarr")

        # WSIPath = self.Cases[
        #    int(np.ceil(item / self.NumSamplesPerSubject)) - 1
        # ]  # assign WSI subject (1-100) to samples between 0-500000
        # ImgPath = self.data_dir + WSIPath
        ImgPath = os.path.join(self.data_dir, case_path)
        Img = zarr.open(ImgPath, mode="r")
        # Mask = zarr.open(ImgPath.replace("imgs", "masks"), mode="r")
        Mask = zarr.open(ImgPath.replace("imgs", "masks_2"), mode="r")
        Prob = np.random.random(1)  # Determine if we sample random patches or from bounding box

        if Prob.item() < self.RandomProb:
            x_pos = np.random.randint(0, Img.shape[0] - self.PatchSize)
            y_pos = np.random.randint(0, Img.shape[1] - self.PatchSize)
            Extracted_Patch = Img[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize, :]
            Extracted_Mask = Mask[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize]

        else:
            PointFile = open(
                # ImgPath.replace("imgs", "sample_points").replace(".zarr", ".pkl"), "rb"
                ImgPath.replace("imgs", "sample_points_2").replace(".zarr", ".pkl"),
                "rb",
            )  # +'.json')
            SamplePoints = pkl.load(PointFile)
            PresentClasses = set(SamplePoints.keys())
            Probs = {key: self.ClassProb[key] for key in self.ClassProb.keys() & PresentClasses}
            classes = list(Probs.keys())
            probs = list(Probs.values())
            probs = probs / np.sum(probs)
            choice = np.random.choice(classes, p=probs)

            Points = SamplePoints[choice]
            point_id = np.random.randint(0, len(Points[0]))
            x_pos = Points[0][point_id]
            y_pos = Points[1][point_id]

            # I want the selected point to be in the center this means the distance to the patch border
            # shouldn't be larger than 0.25 or the patch size
            x_pos = np.random.randint(x_pos - self.PatchSize * 0.9, x_pos - self.PatchSize * 0.1)
            y_pos = np.random.randint(y_pos - self.PatchSize * 0.9, y_pos - self.PatchSize * 0.1)

            # Check if patch is still in WSI
            if not 0 <= x_pos < (Img.shape[0] - self.PatchSize):
                if not 0 <= x_pos:
                    x_pos = 0
                else:
                    x_pos = Img.shape[0] - self.PatchSize

            if not 0 <= y_pos < (Img.shape[1] - self.PatchSize):
                if not 0 <= y_pos:
                    y_pos = 0
                else:
                    y_pos = Img.shape[1] - self.PatchSize

            Extracted_Patch = Img[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize, :]
            Extracted_Mask = Mask[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize]

        if self.transform:
            augmented = self.transform(image=Extracted_Patch, mask=Extracted_Mask)

            Extracted_Patch = augmented["image"].type(torch.float)
            Extracted_Mask = augmented["mask"]
            Extracted_Mask = Extracted_Mask.type(torch.long)

        subset = ImgPath.rsplit("/", 1)[1].split("_", 1)[0]
        # if Extracted_Patch.shape!=[512,512,3
        return Extracted_Patch, Extracted_Mask, subset


class ChallengeDataset_box_point_sampling(Dataset):
    def __init__(
        self,
        data_dir,
        Cases,
        NumSamplesPerSubject,
        RandomProb,
        Class_Prob,
        PatchSize,
        transform=None,
    ):
        self.data_dir = data_dir
        self.Cases = Cases
        SubjectCount = len(self.Cases)
        self.NumPatches = list(range(1, SubjectCount * NumSamplesPerSubject + 1))

        self.RandomProb = RandomProb
        self.NumSamplesPerSubject = NumSamplesPerSubject
        self.PatchSize = PatchSize
        self.ClassProb = Class_Prob
        self.transform = transform

    def __len__(self):
        return len(self.NumPatches)

    def __getitem__(self, item):
        item += 1
        WSIPath = self.Cases[
            int(np.ceil(item / self.NumSamplesPerSubject)) - 1
        ]  # assign WSI subject (1-100) to samples between 0-500000
        # ImgPath = self.data_dir + WSIPath
        ImgPath = os.path.join(self.data_dir, WSIPath)
        Img = zarr.open(ImgPath, mode="r")
        Mask = zarr.open(ImgPath.replace("imgs", "masks"), mode="r")
        Prob = np.random.random(1)  # Determine if we sample random patches or from bounding box

        if Prob.item() < self.RandomProb:
            x_pos = np.random.randint(0, Img.shape[0] - self.PatchSize)
            y_pos = np.random.randint(0, Img.shape[1] - self.PatchSize)
            Extracted_Patch = Img[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize, :]
            Extracted_Mask = Mask[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize]

        elif Prob.item() < (1 - self.RandomProb) / 2 + self.RandomProb:
            PointFile = open(
                ImgPath.replace("imgs", "sample_points").replace(".zarr", ".pkl"), "rb"
            )  # +'.json')
            SamplePoints = pkl.load(PointFile)
            PresentClasses = set(SamplePoints.keys())
            Probs = {key: self.ClassProb[key] for key in self.ClassProb.keys() & PresentClasses}
            classes = list(Probs.keys())
            probs = list(Probs.values())
            probs = probs / np.sum(probs)
            choice = np.random.choice(classes, p=probs)
            Points = SamplePoints[choice]
            point_id = np.random.randint(0, len(Points[0]))
            x_pos = Points[0][point_id]
            y_pos = Points[1][point_id]

            # I want the selected point to be in the center this means the distance to the patch border
            # shouldn't be larger than 0.25 or the patch size
            x_pos = np.random.randint(x_pos - self.PatchSize * 0.75, x_pos - self.PatchSize * 0.25)
            y_pos = np.random.randint(y_pos - self.PatchSize * 0.75, y_pos - self.PatchSize * 0.25)

            # Check if patch is still in WSI
            if not 0 <= x_pos < (Img.shape[0] - self.PatchSize):
                if not 0 <= x_pos:
                    x_pos = 0
                else:
                    x_pos = Img.shape[0] - self.PatchSize

            if not 0 <= y_pos < (Img.shape[1] - self.PatchSize):
                if not 0 <= y_pos:
                    y_pos = 0
                else:
                    y_pos = Img.shape[1] - self.PatchSize

            Extracted_Patch = Img[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize, :]
            Extracted_Mask = Mask[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize]
        else:
            BoxFile = open(ImgPath.replace("imgs", "boxes").replace(".zarr", ".json"))  # +'.json')
            BoundingBoxes = json.load(BoxFile)
            PresentClasses = set(BoundingBoxes.keys())
            Probs = {key: self.ClassProb[key] for key in self.ClassProb.keys() & PresentClasses}
            classes = list(Probs.keys())
            probs = list(Probs.values())
            probs = probs / np.sum(probs)
            choice = np.random.choice(classes, p=probs)
            Boxes = BoundingBoxes[choice]

            areaList = []

            for box in Boxes:
                area = (box[2] - box[0]) * (box[3] - box[1])
                areaList.append(area)

            areaList = areaList / np.sum(areaList)
            BoxCords = Boxes[np.random.choice(list(range(0, len(Boxes))), p=areaList)]

            if (BoxCords[2] - BoxCords[0]) > self.PatchSize:
                x_pos = np.random.randint(
                    BoxCords[0] - int(self.PatchSize / 2),
                    BoxCords[2] - int(self.PatchSize / 2),
                )
            else:
                if BoxCords[2] == BoxCords[0]:
                    x_pos = BoxCords[0] - int(self.PatchSize / 2)
                else:
                    x_pos = np.random.randint(BoxCords[0] - int(self.PatchSize / 2), BoxCords[2])

            if (BoxCords[3] - BoxCords[1]) > self.PatchSize:
                y_pos = np.random.randint(
                    BoxCords[1] - int(self.PatchSize / 2),
                    BoxCords[3] - int(self.PatchSize / 2),
                )
            else:
                if BoxCords[3] == BoxCords[1]:
                    y_pos = BoxCords[3] - int(self.PatchSize / 2)
                else:
                    y_pos = np.random.randint(BoxCords[1] - int(self.PatchSize / 2), BoxCords[3])

            # Check if patch is still in WSI
            if not 0 <= x_pos < (Img.shape[0] - self.PatchSize):
                if not 0 <= x_pos:
                    x_pos = 0
                else:
                    x_pos = Img.shape[0] - self.PatchSize

            if not 0 <= y_pos < (Img.shape[1] - self.PatchSize):
                if not 0 <= y_pos:
                    y_pos = 0
                else:
                    y_pos = Img.shape[1] - self.PatchSize

            Extracted_Patch = Img[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize, :]
            Extracted_Mask = Mask[x_pos : x_pos + self.PatchSize, y_pos : y_pos + self.PatchSize]
        if self.transform:
            augmented = self.transform(image=Extracted_Patch, mask=Extracted_Mask)

            Extracted_Patch = augmented["image"].type(torch.float)
            Extracted_Mask = augmented["mask"]
            Extracted_Mask = Extracted_Mask.type(torch.long)

        return Extracted_Patch, Extracted_Mask
