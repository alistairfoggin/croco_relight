import os
import random

import numpy as np
from PIL import Image
import torch
from equilib import equi2pers
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

class PreloadedBigTimeDataset(Dataset):
    def __init__(self, root_dir, internal_folder="00", transform=None):
        self.root_dir = root_dir
        self.subdirectories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.images = []
        self.transform = transform
        self.internal_folder = internal_folder

        self.to_img = transforms.ToImage()
        self.base_transform = transforms.Compose([
            transforms.Resize((2160, 3840)),
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])

        for subdir_name in self.subdirectories:
            if self.internal_folder is not None:
                subdir_path = os.path.join(self.root_dir, subdir_name, self.internal_folder)
            else:
                subdir_path = os.path.join(self.root_dir, subdir_name)
            image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

            if len(image_files) < 2:
                raise ValueError(f"Subdirectory {subdir_name} does not contain at least two images.")

            subdir_images = []
            np.random.shuffle(image_files)
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(subdir_path, img_file)
                image = self.to_img(Image.open(img_path).convert("RGB"))

                subdir_images.append(image)
                if i > 8:
                    break

            self.images.append(subdir_images)

    def __len__(self):
        return len(self.subdirectories)

    def __getitem__(self, index):
        image_files = self.images[index]
        imgs = torch.stack(random.sample(image_files, 2))
        imgs = self.base_transform(imgs)
        return imgs


class BigTimeDataset(Dataset):
    def __init__(self, root_dir, internal_folder="00", transform=None, device=None):
        """
        Args:
            root_dir (string): Directory with all the subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.subdirectories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.transform = transform
        self.internal_folder = internal_folder
        self.device = device

    def get_idx(self, dir_name):
        return self.subdirectories.index(dir_name)

    def __len__(self):
        return len(self.subdirectories)

    def __getitem__(self, idx):
        subdir_name = self.subdirectories[idx]

        if self.internal_folder is not None:
            subdir_path = os.path.join(self.root_dir, subdir_name, self.internal_folder)
        else:
            subdir_path = os.path.join(self.root_dir, subdir_name)
        image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

        if len(image_files) < 2:
            raise ValueError(f"Subdirectory {subdir_name} does not contain at least two images.")

        selected_images = random.sample(image_files, 2)
        pair_images = []
        for img_file in selected_images:
            img_path = os.path.join(subdir_path, img_file)
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)
            pair_images.append(image)
        return torch.stack(pair_images)  # stack the pair of images


def project_imgs(batch, resolution=(224, 224), fov=80.0):
    # batch Bx2x3xHxW
    rand_pitch = np.random.randn(batch.shape[0]) * np.pi / 10
    rand_yaw = np.random.rand(batch.shape[0]) * 2 * np.pi
    rots = [{
        'roll': 0.,
        'pitch': rand_pitch[i],  # rotate vertical
        'yaw': rand_yaw[i],  # rotate horizontal
    } for i in range(batch.shape[0])]

    pers_imgs = torch.zeros(batch.shape[0], 2, 3, resolution[0], resolution[1], device=batch.device)

    # obtain perspective image
    pers_imgs[:, 0] = equi2pers(
        equi=batch[:, 0],
        rots=rots,
        height=resolution[0],
        width=resolution[1],
        fov_x=fov,
        mode="bilinear",
    )

    pers_imgs[:, 1] = equi2pers(
        equi=batch[:, 1],
        rots=rots,
        height=resolution[0],
        width=resolution[1],
        fov_x=fov,
        mode="bilinear",
    )

    return pers_imgs


if __name__ == "__main__":
    # Example usage:
    # root_dir = "../../bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/"  # replace with your directory path
    root_dir = "../../time360/result/"  # replace with your directory path
    num_subdirectories_per_batch = 8

    transform = transforms.Compose([
        transforms.Resize((2160, 3840)),
        # transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    res = 224
    dataset = BigTimeDataset(root_dir, internal_folder=None, transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_subdirectories_per_batch, shuffle=True)

    for batch in dataloader:
        batch = batch.to(torch.device("cuda"))
        batch = project_imgs(batch, resolution=(res, res), fov=80.0).cpu()
        print("Batch shape:", batch.shape)  # should be [num_subdirectories_per_batch, 2, 3, 224, 224]
        dir1 = batch[0]
        img1 = dir1[0]
        img2 = dir1[1]
        combined_img = np.zeros((res, res * 2, 3))
        combined_img[:, :res, :] = img1.permute(1, 2, 0).numpy()
        combined_img[:, res:, :] = img2.permute(1, 2, 0).numpy()
        plt.imshow(combined_img)
        plt.show()
        # Do something with the batch
