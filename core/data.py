import torch.utils.data as data
import os.path
import random
import torch
from PIL import Image

class ImageAttributeDataset(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, filename, transform):
        """Initialize and preprocess the CelebA dataset."""
        self.lines = [line.rstrip().split() for line in open(filename, 'r')]
        self.transform = transform
        self.length = len(self.lines)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        line = self.lines[index]
        image = Image.open(line[0])
        conditions = [int(condition) for condition in line[1:]]
        return self.transform(image), torch.Tensor(conditions)

    def __len__(self):
        """Return the number of images."""
        return self.length