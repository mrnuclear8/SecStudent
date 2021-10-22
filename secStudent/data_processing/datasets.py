import glob
import random
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train", key_path=r'C:\Users\XXX\PycharmProjects\secStudent\saved_key\keytest.pth'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.key = torch.load(key_path)
        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
        self.crop = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((256, 256))]
        )
    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        # concat = torch.cat([item_A, item_B], dim=0)
        # croped = self.crop(concat)
        # item_A = croped[:3]
        # item_B = croped[3:]

        item_A = torch.cat([item_A, self.key], dim=0)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
