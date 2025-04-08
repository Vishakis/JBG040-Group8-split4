import numpy as np
import torch
import torchvision.transforms as T
from skimage import exposure
from skimage.filters import gaussian
from pathlib import Path
from typing import Tuple


class ImageDataset:
    def __init__(self, x: Path, y: Path, preprocess: str = "none", augment: bool = False) -> None:
        """
        :param x: Path to .npy file for images (assumed shape [N, H, W] for grayscale).
        :param y: Path to .npy file for labels (assumed shape [N]).
        :param preprocess: One of ["none", "hist_eq", "clahe", "hist_eq_gaussian"].
        :param augment: Whether to apply data augmentation (flip/rotate).
        """
        # Load images and labels from disk.
        self.imgs = self.load_numpy_arr_from_npy(x)  # e.g. shape (N, 128, 128)
        self.targets = self.load_numpy_arr_from_npy(y)  # e.g. shape (N,)
        self.preprocess = preprocess
        self.augment = augment


        # Define transforms: convert tensor to PIL, resize to 224x224, apply augmentation if desired, then convert back to tensor.
        if self.augment:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.3),          #Adjust data augmentation flipping here
                T.RandomRotation(degrees=10),           #Adjust data augmentation rotation here
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
            ])


    def __len__(self) -> int:
        return len(self.targets)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.imgs[idx] / 255.0  # Normalize
        if self.preprocess == "hist_eq":
            image = exposure.equalize_hist(image)
        elif self.preprocess == "clahe":
            image = exposure.equalize_adapthist(image)
        elif self.preprocess == "hist_eq_gaussian":
            image = gaussian(image, sigma=1)
            image = exposure.equalize_hist(image)


        image = torch.from_numpy(image).float()


        # If image is 2D, add a channel dimension:
        if image.ndim == 2:
            image = image.unsqueeze(0)
        # If image doesn't have 3 channels, replicate:
        if image.shape[0] != 3:
            image = image.repeat(3, 1, 1)


        image = self.transform(image)




        label = int(self.targets[idx])
        return image, label


    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        return np.load(path)
