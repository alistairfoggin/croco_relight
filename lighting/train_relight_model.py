import numpy as np
import torch
from equilib import Equi2Pers
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

import torchvision.transforms.v2 as transforms
from lighting.dataloader import project_imgs, PreloadedBigTimeDataset, BigTimeDataset
from lighting.relight import ssim_l1_loss_fn, img_mean, img_std
from lighting.relight_model import CroCoDecode, RelightModule

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    ckpt = torch.load('pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth', 'cpu')
    croco_decode = CroCoDecode(**ckpt.get('croco_kwargs', {}), pretrained_model=ckpt['model'])
    decode_ckpt = torch.load('lighting/models/croco_relight_pretrained.pth', 'cpu')
    croco_decode.load_state_dict(decode_ckpt)
    croco_relight = RelightModule(croco_decode).to(device)
    croco_optim = torch.optim.Adam(croco_relight.parameters(), lr=0.0001)

    root_dir1 = "../bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/"
    root_dir2 = "../time360/result/"

    # bigtime_transform = transforms.Compose([
    #     transforms.Resize(512),
    #     transforms.CenterCrop(448),
    #     transforms.ToImage(),
    #     transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    #     transforms.ToDtype(torch.float32, scale=True),
    #     transforms.Normalize(mean=img_mean, std=img_std),
    # ])
    equirect_transform = transforms.Compose([
        # transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.Resize((2160, 3840)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    # dataset_bigtime = BigTimeDataset(root_dir1, transform=bigtime_transform, device=device)
    dataset_360 = PreloadedBigTimeDataset(root_dir2, internal_folder=None, transform=None)
    # dataset_360 = BigTimeDataset(root_dir2, internal_folder=None, transform=equirect_transform)
    # dataset = ConcatDataset([dataset1, dataset2])
    batch_size = 12
    dataloader = DataLoader(dataset_360, batch_size=batch_size, shuffle=True, num_workers=0)

    latent_loss_fn = nn.MSELoss()
    img_loss_fn = ssim_l1_loss_fn(0.2, True)

    for epoch in range(300):
        for i, batch in enumerate(dataloader):
            batch = project_imgs(batch.to(device), resolution=(224, 224))

            # obtain perspective image

            img1 = batch[:, 0]
            img2 = batch[:, 1]

            # img1 = equi2pers(equi=img1,rots=rots)
            # img2 = equi2pers(equi=img2,rots=rots)

            croco_optim.zero_grad()

            # Accidentally decoded using feat1 and feat2 instead of static1 and static2. I'm an idiot lol
            img1_relit, img2_relit, static1, static2 = croco_relight(img1, img2)
            # _, _, delit_static1, delit_static2, _, _ = croco_relight(img1_delit, img2_delit)

            loss_relight = img_loss_fn(img1_relit, img2) + img_loss_fn(img2_relit, img1) #+ 0.5 * (img_loss_fn(img1_recon, img1) + img_loss_fn(img2_recon, img2))
            # loss_delight = img_loss_fn(img1_delit, img2_delit)
            # loss_reconstruction = img_loss_fn(img1_recon, img1) + img_loss_fn(img2_recon, img2)
            loss_static_latents = latent_loss_fn(static1, static2) #+ latent_loss_fn(pre_latent, post_latent) #+ latent_loss_fn(delit_static1, static1.detach()) + latent_loss_fn(delit_static2, static2.detach())

            loss = loss_relight + 0.1 * loss_static_latents #+ 0.1 * latent_loss_fn(pre_latent.detach(), post_latent) #+ 0.2 * loss_delight
            # loss = loss_static_latents #+ 0.2 * loss_delight
            # loss = latent_loss_fn(pre_latent.detach(), post_latent)
            loss.backward()
            croco_optim.step()

            if i == 0:
                print(
                    f"Epoch {epoch}, iteration {i}, Relighting loss: {loss_relight.item()}")# +
                    # f" Reconstruction loss: {loss_reconstruction.item()}," +
                    # f" Static Latent Loss: {loss_static_latents.item()}")

                res = 224
                # with torch.no_grad():
                #     img1 = transforms.Resize(res)(img1)
                #     img2 = transforms.Resize(res)(img2)
                #     img1_relit, img2_relit, _, _ = croco(img1, img2)
                out_img = np.zeros((res * 2, res * 2, 3))
                # 0 0: img2 relit to match img1
                out_img[:res, :res, :] = img1[0].permute(1, 2, 0).detach().cpu().numpy()
                # 0 1: img1 reconstruction
                # out_img[:res, res:2 * res, :] = img1_delit[0].permute(1, 2, 0).detach().cpu().numpy()
                # 0 2: img1 gt
                out_img[:res, res:, :] = img2_relit[0].permute(1, 2, 0).detach().cpu().numpy()
                # 1 0: img1 relit to match img2
                out_img[res:, :res, :] = img2[0].permute(1, 2, 0).detach().cpu().numpy()
                # 1 1: img2 reconstruction
                # out_img[res:, res:2 * res, :] = img2_delit[0].permute(1, 2, 0).detach().cpu().numpy()
                # 1 2: img2 gt
                out_img[res:, res:, :] = img1_relit[0].permute(1, 2, 0).detach().cpu().numpy()
                out_img = out_img * np.array(img_std).reshape((1, 1, -1)) + np.array(img_mean).reshape((1, 1, -1))
                plt.imshow(out_img)
                plt.show()

    torch.save(croco_relight.state_dict(), "lighting/models/croco_relight_delight_from_pretrain2.pth")
    # torch.save(lighting_decoder.state_dict(), "models/decoder_dpt.pth")
