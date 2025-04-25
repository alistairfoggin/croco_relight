from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

import torchvision.transforms.v2 as transforms
from lighting.dataloader import BigTimeDataset
from lighting.relight import LightingExtractor, ssim_l1_loss_fn, img_mean, img_std
from models.croco import CroCoNet
from models.head_downstream import PixelwiseTaskWithDPT


class CroCoRelighting(CroCoNet):

    def __init__(self, pretrained_model=None, **kwargs):
        """ Build network for binocular downstream task
        It takes an extra argument head, that is called with the features
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        """
        super(CroCoRelighting, self).__init__(**kwargs)
        if pretrained_model is not None:
            self.load_state_dict(pretrained_model, strict=True)
            # Reset decoder and freeze encoder
            self.enc_blocks.requires_grad_(False)
            self.mask_token = None
            self.prediction_head = None
            self._set_decoder(self.enc_embed_dim, self.dec_embed_dim, 16, 4, 2, partial(nn.LayerNorm, eps=1e-6), False)
            # self._set_decoder(1024, 1024, 16, 8, 2, partial(nn.LayerNorm, 1e-6), False)
        self.lighting_extractor = LightingExtractor(patch_size=self.enc_embed_dim, rope=self.rope)
        # step = self.dec_depth//4
        hooks_idx = [3, 2, 1, 0]
        head = PixelwiseTaskWithDPT(hooks_idx=hooks_idx, num_channels=3)
        head.setup(self, dim_tokens=self.dec_embed_dim)
        self.head = head

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    # def _set_mask_token(self, *args, **kwargs):
    #     """ No mask token """
    #     self.mask_token = None
    #     return
    #
    # def _set_prediction_head(self, *args, **kwargs):
    #     """ No prediction head for downstream tasks, define your own head """
    #     return

    def encode_image_pairs(self, img1, img2, return_all_blocks=False):
        """ run encoder for a pair of images
            it is actually ~5% faster to concatenate the images along the batch dimension
             than to encode them separately
        """
        ## the two commented lines below is the naive version with separate encoding
        # out, pos, _ = self._encode_image(img1, do_mask=False, return_all_blocks=return_all_blocks)
        # out2, pos2, _ = self._encode_image(img2, do_mask=False, return_all_blocks=False)
        ## and now the faster version
        out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0), do_mask=False,
                                         return_all_blocks=return_all_blocks)
        if return_all_blocks:
            out, out2 = list(map(list, zip(*[o.chunk(2, dim=0) for o in out])))
            out2 = out2[-1]
        else:
            out, out2 = out.chunk(2, dim=0)
        pos, pos2 = pos.chunk(2, dim=0)
        return out, out2, pos, pos2

    def forward(self, img1, img2):
        B, C, H1, W1 = img1.size()
        _, _, H2, W2 = img2.size()
        img1_info = {'height': H1, 'width': W1}
        img2_info = {'height': H2, 'width': W2}
        return_all_blocks = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks
        feat1, feat2, pos1, pos2 = self.encode_image_pairs(img1, img2, return_all_blocks=False)

        static1, dyn1, dyn_pos1 = self.lighting_extractor(feat1, pos1)
        static2, dyn2, dyn_pos2 = self.lighting_extractor(feat2, pos2)

        # Relight img 1 to be like img 2
        decout1 = self._decoder(static1, pos1, None, dyn2, dyn_pos2, return_all_blocks=return_all_blocks)
        decout2 = self._decoder(static2, pos2, None, dyn1, dyn_pos1, return_all_blocks=return_all_blocks)
        return self.head(decout1, img1_info), self.head(decout2, img2_info), static1, static2

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    ckpt = torch.load('../pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth', 'cpu')
    croco = CroCoRelighting(**ckpt.get('croco_kwargs', {}), pretrained_model=ckpt['model']).to(device)
    croco_optim = torch.optim.Adam(croco.parameters(), lr=0.0001)

    root_dir = "../../bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/"  # replace with your directory path

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(512),
        transforms.CenterCrop(224*2),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    dataset = BigTimeDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    latent_loss_fn = nn.MSELoss()
    img_loss_fn = ssim_l1_loss_fn(0.2, True)

    for epoch in range(10_000):
        batch = next(iter(dataloader)).to(device)

        img1 = batch[:, 0]
        img2 = batch[:, 1]

        croco_optim.zero_grad()

        # Accidentally decoded using feat1 and feat2 instead of static1 and static2. I'm an idiot lol
        img1_relit, img2_relit, static1, static2 = croco(img1, img2)

        loss_relight = img_loss_fn(img1_relit, img2) + img_loss_fn(img2_relit, img1)
        # loss_reconstruction = img_loss_fn(img1_recon, img1) + img_loss_fn(img2_recon, img2)
        loss_static_latents = latent_loss_fn(static1, static2)

        loss = loss_relight + 0.1 * loss_static_latents  # + 0.2 * loss_reconstruction + 0.1 * loss_static_latents
        loss.backward()

        croco_optim.step()

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch}, Relighting loss: {loss_relight.item()}," +
                # f" Reconstruction loss: {loss_reconstruction.item()}," +
                f" Static Latent Loss: {loss_static_latents.item()}")

        res = 448
        if epoch % 30 == 0:
            # with torch.no_grad():
            #     img1 = transforms.Resize(res)(img1)
            #     img2 = transforms.Resize(res)(img2)
            #     img1_relit, img2_relit, _, _ = croco(img1, img2)
            out_img = np.zeros((res * 2, res * 3, 3))
            # 0 0: img2 relit to match img1
            out_img[:res, :res, :] = img2_relit[0].permute(1, 2, 0).detach().cpu().numpy()
            # 0 1: img1 reconstruction
            # out_img[:res, res:2 * res, :] = img1_recon[0].permute(1, 2, 0).detach().cpu().numpy()
            # 0 2: img1 gt
            out_img[:res, 2 * res:, :] = img1[0].permute(1, 2, 0).detach().cpu().numpy()
            # 1 0: img1 relit to match img2
            out_img[res:, :res, :] = img1_relit[0].permute(1, 2, 0).detach().cpu().numpy()
            # 1 1: img2 reconstruction
            # out_img[res:, res:2 * res, :] = img2_recon[0].permute(1, 2, 0).detach().cpu().numpy()
            # 1 2: img2 gt
            out_img[res:, 2 * res:, :] = img2[0].permute(1, 2, 0).detach().cpu().numpy()
            out_img = out_img * np.array(img_std).reshape((1, 1, -1)) + np.array(img_mean).reshape((1, 1, -1))
            plt.imshow(out_img)
            plt.show()

    torch.save(croco.state_dict(), "models/croco_relight2.pth")
    # torch.save(lighting_decoder.state_dict(), "models/decoder_dpt.pth")
