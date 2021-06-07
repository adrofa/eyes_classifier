from train.versions.model import get_model
from train.versions.augmentation import get_augmentation

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np
import torch

CONFIG = {
    "augmentation_version": 1,
    "models": {
        # (model_version, weights_path)
        "single": [(4, "../output/models/hypothesis-4/fold-1/model.pt")],

        "ensemble": [(4, "../output/models/hypothesis-4/fold-1/model.pt"),
                     (4, "../output/models/hypothesis-4/fold-2/model.pt"),
                     (4, "../output/models/hypothesis-4/fold-3/model.pt"),
                     (4, "../output/models/hypothesis-4/fold-4/model.pt"),
                     (4, "../output/models/hypothesis-4/fold-5/model.pt")],
    }
}


class OpenEyesClassificator:
    """Classifier for opened/closed eyes dataset.

    Args:
        cls_type (str): if "single" - classifier uses a single model;
            if "ensemble" - classifier uses an ensemble of models.
        device (str): "cpu" or "gpu" - desired device for inference.
    """
    def __init__(self, cls_type="ensemble", device="cpu"):
        self.device = device
        self.augmentation = get_augmentation(CONFIG["augmentation_version"])["valid"]
        self.models = [get_model(version, weights).eval()
                       for (version, weights) in CONFIG["models"][cls_type]]

    def predict(self, inpIm):
        """Inference for the provided image.

        Args:
            inpIm (str): path to the image.
        Returns:
            is_open_score (float): class score for the image between 0 and 1;
                higher score means opened eyes, lower score - closed.
        """
        img = self.prepare_img(inpIm)
        is_open_score = torch.tensor(0).type(torch.FloatTensor)
        with torch.no_grad():
            for m in self.models:
                is_open_score += (m.forward(img) / len(self.models))
        is_open_score = torch.sigmoid(is_open_score).item()
        return is_open_score

    def prepare_img(self, inpIm):
        """Open an image and prepare it for passing through the CNN.

        Args:
            inpIm (str): path to the image.
        Returns:
            img (torch.tensor): augumented image of shape BxHxWxC;
                B - batch dimension (always = 1),
                H, W - height and width (always = 24 - dataset's specificity),
                C - channels (always = 1 - grayscale images).
        """
        augmentation = A.Compose([self.augmentation, ToTensorV2(p=1)])
        img = cv2.imread(str(inpIm), cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 2)
        img = augmentation(image=img)["image"]
        img = img.unsqueeze(0)
        return img
