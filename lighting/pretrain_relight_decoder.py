import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import ImageNet
from torchvision.utils import make_grid

from lighting.relight import img_mean, img_std, ssim_l1_loss_fn
from lighting.relight_model import CroCoDecode

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

if __name__ == "__main__":
    epochs = 10
    lr = 1e-4
    batch_size = 128

    run = wandb.init(
        entity="alistair-foggin-university-of-york",
        project="croco-relighting-pretrain",
        config={"epochs": epochs, "learning_rate": lr, "batch_size": batch_size},
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    ckpt = torch.load('pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth', 'cpu')
    croco = CroCoDecode(**ckpt.get('croco_kwargs', {}))
    croco.load_state_dict(ckpt['model'])
    croco.setup()
    croco = croco.to(device)
    croco_optim = torch.optim.Adam(croco.parameters(), lr=lr)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    img_loss_fn = ssim_l1_loss_fn(0.2, False)

    train_dataset = ImageNet("datasets/imagenet", split="train", transform=transform)
    val_dataset = ImageNet("datasets/imagenet", split="train", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for epoch in range(epochs):
        for i, (img_batch, _) in enumerate(train_dataloader):
            img_gt = img_batch.to(device)

            croco_optim.zero_grad()
            img_pred = croco.encode_decode(img_gt)
            loss = img_loss_fn(img_pred, img_gt)
            loss.backward()
            croco_optim.step()

            if i % 10 == 0:
                run.log(data={"loss": loss.item()}, step=i + epoch * len(train_dataloader) // batch_size)
                print(f"Epoch {epoch} iter {i} loss: {loss.item():.4f}")
            if i % 100 == 0:
                all_imgs = torch.cat((img_gt[:4].detach(), img_pred[:4].detach()), dim=0)
                img_mean_tensor = torch.tensor(img_mean).reshape(1, -1, 1, 1).to(device)
                img_std_tensor = torch.tensor(img_std).reshape(1, -1, 1, 1).to(device)
                all_imgs = torch.clamp(all_imgs * img_std_tensor + img_mean_tensor, 0, 1)
                img_array = make_grid(all_imgs, nrow=4).cpu()
                # show(img_array.cpu())

                wandb_imgs = wandb.Image(img_array, caption="Top: Ground Truth, Bottom: Reconstruction")
                run.log({"train_images": wandb_imgs})
            if i % 10000 == 0:
                torch.save(croco.state_dict(), "lighting/models/croco_relight_pretrained.pth")
                run.log_artifact("lighting/models/croco_relight_pretrained.pth", name="croco-decoder-model",
                                 type="model")

        val_loss = []
        for i, (img_batch, _) in enumerate(val_dataloader):
            img_gt = img_batch.to(device)
            with torch.no_grad():
                img_pred = croco.encode_decode(img_gt)
                loss = img_loss_fn(img_pred, img_gt)
                val_loss.append(loss.item())

            if i == 0:
                all_imgs = torch.cat((img_gt[:4].detach(), img_pred[:4].detach()), dim=0)
                img_mean_tensor = torch.tensor(img_mean).reshape(1, -1, 1, 1).to(device)
                img_std_tensor = torch.tensor(img_std).reshape(1, -1, 1, 1).to(device)
                all_imgs = torch.clamp(all_imgs * img_std_tensor + img_mean_tensor, 0, 1)
                img_array = make_grid(all_imgs, nrow=4).cpu()
                wandb_imgs = wandb.Image(img_array, caption="Top: Ground Truth, Bottom: Reconstruction")
                run.log({"val_images": wandb_imgs})

        val_loss = np.mean(val_loss)
        run.log(data={"val_loss": loss.item()}, step=epoch * len(train_dataloader) // batch_size)
        print(f"Epoch {epoch} validation loss: {loss.item():.4f}")

    run.log({"epoch": epoch, "batch": i, "loss": loss.item()})
    torch.save(croco.state_dict(), "lighting/models/croco_relight_pretrained.pth")
    run.log_artifact("lighting/models/croco_relight_pretrained.pth", name="croco-decoder-model", type="model")
    run.finish()
