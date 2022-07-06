import numpy as np
import copy
import random

from staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools.tissue_masks.luminosity_threshold_tissue_locator import (
    LuminosityThresholdTissueLocator,
)
from staintools.miscellaneous.get_concentrations import get_concentrations

import spams


def convert_RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).
    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    mask = I == 0
    I[mask] = 1
    return np.maximum(-1 * np.log(I / 255), 1e-6)


AugmentationStainMatrices = [
    np.array([[0.43302991, 0.85491449, 0.28566819], [0.03297139, 0.94233339, 0.33304754]]),
    np.array([[0.52606201, 0.8061208, 0.27097603], [0.07723074, 0.94943948, 0.30430264]]),
    np.array([[0.49148437, 0.83158453, 0.25867023], [0.00701892, 0.96764777, 0.25220733]]),
]


class StainAugmentor(object):
    def __init__(self, method, sigma1=0.2, sigma2=0.2, augment_background=True):
        if method.lower() == "macenko":
            self.extractor = MacenkoStainExtractor
        elif method.lower() == "vahadane":
            self.extractor = VahadaneStainExtractor
        else:
            raise Exception("Method not recognized.")
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.augment_background = augment_background

    def fit(self, I):
        """
        Fit to an image I.
        :param I:
        :return:
        """
        self.image_shape = I.shape
        # self.stain_matrix = self.extractor.get_stain_matrix(I)
        self.stain_matrix = AugmentationStainMatrices[random.randint(0, 2)]

        # self.source_concentrations = get_concentrations(I, self.stain_matrix)
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        self.source_concentrations = (
            spams.lasso(X=OD.T, D=self.stain_matrix.T, mode=2, lambda1=0.01, pos=True, numThreads=1)
            .toarray()
            .T
        )

        self.n_stains = self.source_concentrations.shape[1]
        try:
            self.tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(I).ravel()
        except:
            pass

    def pop(self):
        """
        Get an augmented version of the fitted image.
        :return:
        """
        augmented_concentrations = copy.deepcopy(self.source_concentrations)

        for i in range(self.n_stains):
            alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)

            beta = np.random.uniform(-self.sigma2, self.sigma2)
            if self.augment_background:
                augmented_concentrations[:, i] *= alpha
                augmented_concentrations[:, i] += beta
            else:
                augmented_concentrations[self.tissue_mask, i] *= alpha
                augmented_concentrations[self.tissue_mask, i] += beta

        I_augmented = 255 * np.exp(-1 * np.dot(augmented_concentrations, self.stain_matrix))
        I_augmented = I_augmented.reshape(self.image_shape)
        I_augmented = np.clip(I_augmented, 0, 255)

        return np.array(I_augmented, dtype=np.uint8)
