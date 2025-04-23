import os
import random

import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms


class BigTimeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.subdirectories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.transform = transform

    def get_idx(self, dir_name):
        return self.subdirectories.index(dir_name)

    def __len__(self):
        return len(self.subdirectories)

    def __getitem__(self, idx):
        subdir_name = self.subdirectories[idx]

        subdir_path = os.path.join(self.root_dir, subdir_name, "00")
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


if __name__ == "__main__":
    # Example usage:
    root_dir = "../../bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/"  # replace with your directory path
    num_subdirectories_per_batch = 8

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = BigTimeDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_subdirectories_per_batch, shuffle=True)

    for batch in dataloader:
        print("Batch shape:", batch.shape)  # should be [num_subdirectories_per_batch, 2, 3, 224, 224]
        dir1 = batch[0]
        img1 = dir1[0]
        img2 = dir1[1]
        combined_img = np.zeros((224, 224*2, 3))
        combined_img[:, :224, :] = img1.permute(1, 2, 0).numpy()
        combined_img[:, 224:, :] = img2.permute(1, 2, 0).numpy()
        plt.imshow(combined_img)
        plt.show()
        # Do something with the batch
