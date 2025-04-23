import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from lighting.dataloader import BigTimeDataset
from lighting.relight import LightingExtractor, LightingDecoder
from models.croco import CroCoNet

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

    ckpt = torch.load('../pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth', 'cpu')
    croco = CroCoNet(**ckpt.get('croco_kwargs', {}), mask_ratio=0.0).to(device)
    croco.load_state_dict(ckpt['model'], strict=True)

    lighting_extractor = LightingExtractor(rope=croco.rope).to(device)
    lighting_extractor.load_state_dict(torch.load("models/extractor.pth"))
    lighting_decoder = LightingDecoder(rope=croco.rope).to(device)
    lighting_decoder.load_state_dict(torch.load("models/decoder.pth"))

    root_dir = "../../bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/"  # replace with your directory path
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    dataset = BigTimeDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # batch = next(iter(dataloader)).to(device)
    batch = dataset[dataset.get_idx("0350")].unsqueeze(0).to(device)

    img1 = batch[:, 0]
    img2 = batch[:, 1]
    with torch.no_grad():
        feat1, pos1, mask1 = croco._encode_image(img1, do_mask=False)
        feat2, pos2, mask2 = croco._encode_image(img2, do_mask=False)

        static1, dyn1 = lighting_extractor(feat1, pos1)
        static2, dyn2 = lighting_extractor(feat2, pos2)

        # Interpolate between dyn1 and dyn2
        latents = []
        for i in range(6):
            t = i / 5.
            latents.append(torch.lerp(dyn1, dyn2, t))
        latents = torch.cat(latents, dim=0)

        img1_relit = croco.unpatchify(lighting_decoder(feat1.expand(6, -1, -1), pos1.repeat(6, 1, 1), latents))
        img2_relit = croco.unpatchify(lighting_decoder(feat2.expand(6, -1, -1), pos2.repeat(6, 1, 1), latents))

        out_img = np.zeros((224 * 2, 224 * 6, 3))
        out_img[:224, :, :] = torch.cat(img1_relit.permute(0, 2, 3, 1).unbind(), dim=1).cpu().numpy()
        out_img[224:, :, :] = torch.cat(img2_relit.permute(0, 2, 3, 1).unbind(), dim=1).cpu().numpy()
        out_img = out_img * np.array(img_std).reshape((1, 1, -1)) + np.array(img_mean).reshape((1, 1, -1))
        plt.imshow(out_img)
        plt.show()

        original_imgs = np.zeros((224, 224 * 2, 3))
        original_imgs[:, :224] = img1[0].permute(1, 2, 0).cpu().numpy()
        original_imgs[:, 224:] = img2[0].permute(1, 2, 0).cpu().numpy()
        original_imgs = original_imgs * np.array(img_std).reshape((1, 1, -1)) + np.array(img_mean).reshape((1, 1, -1))
        plt.imshow(original_imgs)
        plt.show()
