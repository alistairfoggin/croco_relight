import os
import random

import numpy as np
from PIL import Image
import torch
from equilib import Equi2Pers
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms


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
        self.device=device

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
        return torch.stack(pair_images).to(self.device)  # stack the pair of images

class BigTime360Dataset(BigTimeDataset):
    def __init__(self, root_dir, internal_folder=None, resolution=(224, 224), fov=80.0, base_transform=None, final_transform=None, device=None):
        super().__init__(root_dir, internal_folder=internal_folder, transform=base_transform, device=device)
        self.final_transform = final_transform

        # Intialize equi2pers
        self.equi2pers = Equi2Pers(
            height=resolution[1],
            width=resolution[0],
            fov_x=fov,
            mode="bilinear",
        )

    def __getitem__(self, idx):
        equirectangular_imgs = super().__getitem__(idx)
        # TODO: speed up projection process with batching
        # rotations
        rots = {
            'roll': 0.,
            'pitch': np.random.randn() * np.pi/10,  # rotate vertical
            'yaw': np.random.rand() * 2 * np.pi,  # rotate horizontal
        }

        # obtain perspective image
        pers_imgs = self.equi2pers(
            equi=equirectangular_imgs,
            rots=[rots, rots],
        )
        return pers_imgs


if __name__ == "__main__":
    # Example usage:
    # root_dir = "../../bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/"  # replace with your directory path
    root_dir = "../../time360/result/"  # replace with your directory path
    num_subdirectories_per_batch = 8

    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    res = 512
    dataset = BigTime360Dataset(root_dir, resolution=(res, res), fov=80.0, base_transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_subdirectories_per_batch, shuffle=True)

    for batch in dataloader:
        print("Batch shape:", batch.shape)  # should be [num_subdirectories_per_batch, 2, 3, 224, 224]
        dir1 = batch[0]
        img1 = dir1[0]
        img2 = dir1[1]
        combined_img = np.zeros((res, res*2, 3))
        combined_img[:, :res, :] = img1.permute(1, 2, 0).numpy()
        combined_img[:, res:, :] = img2.permute(1, 2, 0).numpy()
        plt.imshow(combined_img)
        plt.show()
        # Do something with the batch
