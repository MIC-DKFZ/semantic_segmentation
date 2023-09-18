import torch
from src.metric.confmat import ConfusionMatrix


class Dice(ConfusionMatrix):
    def __init__(
        self,
        per_class: bool = False,
        name: str = "Dice",
        replace_nan: bool = True,
        ignore_bg: bool = False,
        **kwargs,
    ):
        """
        Init the Dice Class as subclass of ConfusionMatrix
        Behaviour similar to torchmetric.F1Score(num_classes=6,average="macro",mdmc_average="global",multiclass=True)

        Parameters
        ----------
        per_class : bool, optional
            If False the mean Dice over the classes is returned
            If True additionally the Dice for each class is returned
        name : str, optional
            Name of the metric, used as prefix for logging the class scores
        replace_nan : bool, optional
            replace NaN by 0.0
        kwargs :
            arguments passed to the super class (ConfusionMatrix)
        """

        super().__init__(**kwargs)

        self.per_class = per_class
        self.name = name
        self.replace_nan = replace_nan
        self.ignore_bg = ignore_bg

    def get_dice_from_mat(self, confmat: torch.Tensor) -> torch.Tensor:
        """
        Computing the Dice from a confusion matrix (class wise)

        Parameters
        ----------
        confmat : torch.Tensor

        Returns
        -------
        torch.Tensor :
            Tensor contains the IoU for each class
        """
        intersection = confmat.diag()
        Dice = 2 * intersection / (confmat.sum(1) + confmat.sum(0))

        # for using a ignore class
        # if self.ignore_class is not None and 0 <= self.ignore_class < self.num_classes:
        #    Dice = torch.cat((Dice[: self.ignore_class], Dice[self.ignore_class + 1 :]))

        return Dice

    def compute(self) -> dict or torch.Tensor:
        """
        Compute the Dice from the confusion matrix.
        Depended on initialization return mean Dice with or without Dice per class
        For Computing the mean set all NaN to 0

        Returns
        -------
        dict or torch.Tensor :
            a single Tensor if only mean IoU is returned, a dict if additionally the class wise
            Dice is returned
        """
        Dice = self.get_dice_from_mat(self.mat.clone())

        # if self.replace_nan:
        #     Dice[Dice.isnan()] = 0.0

        if self.ignore_bg:
            mDice = torch.nanmean(Dice[1:])  # .mean()
        else:
            mDice = torch.nanmean(Dice)  # mean()

        if self.per_class:
            if self.ignore_bg:
                Dice = {self.name + "_" + self.labels[i]: Dice[i] for i in range(1, len(Dice))}
            else:
                Dice = {self.name + "_" + self.labels[i]: Dice[i] for i in range(len(Dice))}
            Dice["mean" + self.name] = mDice
            return Dice
        else:
            return mDice
