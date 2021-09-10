import PIL.Image
from torch.utils.data import Dataset
from PIL import Image
import os


class ImageFolder(Dataset):
    def __init__(self, root, file_list="train.txt", label_file="label.txt", image_size=None, transform=None):
        self.root = root
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transform = transform

        lines = open(os.path.join(root, file_list)).readlines()
        self.files = [line.strip().split(",") for line in lines if line.strip()]

        self.label_names = open(os.path.join(root, label_file)).read().strip().split(",")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_file, mask_file = self.files[i]

        img = Image.open(os.path.join(self.root, img_file)).convert("RGB")
        if self.image_size is not None and img.size != self.image_size:
            img = img.resize(self.image_size)

        mask = Image.open(os.path.join(self.root, mask_file)).convert("I")
        if self.image_size is not None and mask.size != self.image_size:
            mask = mask.resize(self.image_size, resample=PIL.Image.NEAREST)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask

    @property
    def num_classes(self):
        return len(self.label_names)
