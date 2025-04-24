import os

from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image


class PairedDataset(Dataset):
    def __init__(self, low_light_root, normal_light_root, image_size=[200, 200]):
        super().__init__()
        # 只保留有效图像扩展名的文件
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        self.low_light_dataset = [
            os.path.join(low_light_root, image) 
            for image in os.listdir(low_light_root)
            if os.path.splitext(image.lower())[1] in valid_extensions
        ]
        self.normal_light_dataset = [
            os.path.join(normal_light_root, image) 
            for image in os.listdir(normal_light_root)
            if os.path.splitext(image.lower())[1] in valid_extensions
        ]
        self.transforms = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        low_light = Image.open(self.low_light_dataset[idx]).convert('RGB')
        normal_light = Image.open(
            self.normal_light_dataset[idx]).convert('RGB')
        low_light = self.transforms(low_light)
        normal_light = self.transforms(normal_light)

        return low_light, normal_light

    def __len__(self):
        return len(self.low_light_dataset)


class UnpairedDataset(Dataset):
    def __init__(self, low_light_root, low_light_transforms, image_size=[200, 200]):
        super().__init__()
        self.low_light_dataset = [os.path.join(
            low_light_root, image) for image in os.listdir(low_light_root)]
        self.low_light_transform = low_light_transforms
        self.transforms = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        low_light = Image.open(self.low_light_dataset[idx]).convert('RGB')
        low_light = self.low_light_transform(low_light)

        return low_light

    def __len__(self):
        return len(self.low_light_dataset)