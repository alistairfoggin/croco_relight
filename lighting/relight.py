from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from pytorch_msssim import SSIM

from lighting.dataloader import BigTimeDataset
from models.blocks import Block, Mlp, Attention, DropPath, DecoderBlock
from models.croco import CroCoNet
from models.head_downstream import PixelwiseTaskWithDPT
from models.pos_embed import get_2d_sincos_pos_embed


class LightingExtractor(nn.Module):
    def __init__(self, patch_size=1024, num_heads=16, mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 lighting_feature_ratio=1, rope=None):
        super(LightingExtractor, self).__init__()
        # B x num_tokens x patch_size
        self.dynamic_token = nn.Parameter(torch.rand((1, lighting_feature_ratio, patch_size)), requires_grad=True)
        self.base_blocks = nn.ModuleList()
        for _ in range(3):
            self.base_blocks.append(
                Block(patch_size, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=rope))

        # self.static_blk = Block(patch_size, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=rope)
        # self.dynamic_blk = Block(patch_size, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=rope)
        # self.lighting_feature_ratio = lighting_feature_ratio

    def forward(self, x, xpos):
        x = torch.cat((x, self.dynamic_token.expand(x.shape[0], 1, self.dynamic_token.shape[2])), dim=1)
        dyn_pos = torch.ones(xpos.shape[0], 1, xpos.shape[2], dtype=xpos.dtype, device=xpos.device) * -1
        xpos_extra = torch.cat((xpos, dyn_pos), dim=1)
        # x: [B, 196+1, 1024]
        for blk in self.base_blocks:
            x = blk(x, xpos_extra)
        # static: [B, 196, 1024]
        static = x[:, :-1, :]
        # dynamic: [B, 1, 1024]
        dynamic = x[:, -1:, :]
        # static = self.static_blk(x, xpos)
        # dynamic = self.dynamic_blk(x, xpos)[:, 0:0 + self.lighting_feature_ratio, :]
        # combined = self.activation(self.combined_layer(x))
        return static, dynamic, dyn_pos


class LightingBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim * 2, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)

    def forward(self, x, xpos, dynamic_feature):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(torch.cat((self.norm2(x), dynamic_feature.expand(-1, x.shape[1], -1)), dim=-1)))
        return x


class LightingDecoder(nn.Module):
    def __init__(self, embedding_dim=1024, num_heads=16, mlp_ratio=4, patch_size=16, num_blocks=4, rope=None):
        super(LightingDecoder, self).__init__()
        self.dec_blocks = nn.ModuleList()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        for _ in range(num_blocks):
            self.dec_blocks.append(
                LightingBlock(embedding_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=rope))
        self.dec_norm = norm_layer(embedding_dim)
        self.prediction_head = nn.Linear(embedding_dim, patch_size ** 2 * 3)

    def forward(self, x, xpos, dynamic_feature):
        # static_features: [B, 196, 1024]
        # dynamic_feature: [B,   1, 1024]
        for blk in self.dec_blocks:
            x = blk(x, xpos, dynamic_feature)
        x = self.dec_norm(x)
        out = self.prediction_head(x)
        return out


class AltLightingDecoder(nn.Module):
    def __init__(self, enc_embed_dim=1024, dec_embed_dim=1024, dec_num_heads=16, dec_depth=4, mlp_ratio=2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_im2_in_dec=False, rope=None):
        super(AltLightingDecoder, self).__init__()
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # self.dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                         norm_mem=norm_im2_in_dec, rope=rope)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

        self.prediction_head = PixelwiseTaskWithDPT(layer_dims=[128, 256, 512, 1024])
        self.prediction_head.num_channels = 1
        # TODO: call prediction_head.setup() and incorporate correct params for dpt

    def forward(self, static_feat, pos, dyn_feat, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block
                           instead of just the features from the last block (eg for some prediction heads)

        masks1 can be None => assume image1 fully visible
        """
        # encoder to decoder layer
        static_f = self.decoder_embed(static_feat)
        dyn_f = self.decoder_embed(dyn_feat)
        # append masked tokens to the sequence
        f1 = static_f
        f2 = dyn_f.expand(-1, f1.shape[1], -1)
        # add positional embedding
        # if self.dec_pos_embed is not None:
        #     f1 = f1 + self.dec_pos_embed
        #     f2 = f2 + self.dec_pos_embed
        # apply Transformer blocks
        out = f1
        out2 = f2
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out, out2 = blk(_out, out2, pos, pos)
                out.append(_out)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2 = blk(out, out2, pos, pos)
            out = self.dec_norm(out)

        img_info = {'height': 448, 'width': 448}
        res = self.prediction_head(out, img_info)
        return res


device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
img_mean_tensor = torch.tensor(img_mean).reshape(1, -1, 1, 1).to(device)
img_std_tensor = torch.tensor(img_std).reshape(1, -1, 1, 1).to(device)


def rescale_image(img):
    return torch.clamp(img * img_std_tensor + img_mean_tensor, min=0., max=1.)


def ssim_l1_loss_fn(ssim_ratio=0.2, use_l1=True):
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    main_loss = L1Loss() if use_l1 else MSELoss()

    def loss_fn(pred_img, gt_img):
        Ll1 = main_loss(pred_img, gt_img)
        simloss = 1 - ssim(rescale_image(pred_img), rescale_image(gt_img))
        return (1 - ssim_ratio) * Ll1 + ssim_ratio * simloss

    return loss_fn


if __name__ == "__main__":
    ckpt = torch.load('../pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth', 'cpu')
    croco = CroCoNet(**ckpt.get('croco_kwargs', {}), mask_ratio=0.0).to(device)
    croco.load_state_dict(ckpt['model'], strict=True)

    lighting_extractor = LightingExtractor(rope=croco.rope).to(device)
    # lighting_extractor.load_state_dict(torch.load("models/extractor_feat.pth"))
    lighting_decoder = AltLightingDecoder(rope=croco.rope).to(device)
    # lighting_decoder.load_state_dict(torch.load("models/decoder_feat.pth"))

    extractor_optim = torch.optim.Adam(lighting_extractor.parameters(), lr=0.0001)
    decoder_optim = torch.optim.Adam(lighting_decoder.parameters(), lr=0.0001)

    root_dir = "../../bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/"  # replace with your directory path

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    dataset = BigTimeDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    latent_loss_fn = nn.MSELoss()
    img_loss_fn = ssim_l1_loss_fn(0.2, True)

    for epoch in range(5000):
        batch = next(iter(dataloader)).to(device)

        img1 = batch[:, 0]
        img2 = batch[:, 1]
        with torch.no_grad():
            feat1, pos1, mask1 = croco._encode_image(img1, do_mask=False)
            feat2, pos2, mask2 = croco._encode_image(img2, do_mask=False)

        extractor_optim.zero_grad()
        decoder_optim.zero_grad()

        static1, dyn1 = lighting_extractor(feat1, pos1)
        static2, dyn2 = lighting_extractor(feat2, pos2)

        # Accidentally decoded using feat1 and feat2 instead of static1 and static2. I'm an idiot lol
        img1_relit = croco.unpatchify(lighting_decoder(static1, pos1, dyn2))
        img2_relit = croco.unpatchify(lighting_decoder(static2, pos2, dyn1))

        img1_recon = croco.unpatchify(lighting_decoder(static1, pos1, dyn1))
        img2_recon = croco.unpatchify(lighting_decoder(static2, pos2, dyn2))

        loss_relight = img_loss_fn(img1_relit, img2) + img_loss_fn(img2_relit, img1)
        loss_reconstruction = img_loss_fn(img1_recon, img1) + img_loss_fn(img2_recon, img2)
        loss_static_latents = latent_loss_fn(static1, static2)

        loss = loss_relight + 0.2 * loss_static_latents  # + 0.2 * loss_reconstruction + 0.1 * loss_static_latents
        loss.backward()

        extractor_optim.step()
        decoder_optim.step()

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch}, Relighting loss: {loss_relight.item()}," +
                f" Reconstruction loss: {loss_reconstruction.item()}," +
                f" Static Latent Loss: {loss_static_latents.item()}")

        res = 448
        if epoch % 30 == 0:
            out_img = np.zeros((res * 2, res * 3, 3))
            # 0 0: img2 relit to match img1
            out_img[:res, :res, :] = img2_relit[0].permute(1, 2, 0).detach().cpu().numpy()
            # 0 1: img1 reconstruction
            out_img[:res, res:2 * res, :] = img1_recon[0].permute(1, 2, 0).detach().cpu().numpy()
            # 0 2: img1 gt
            out_img[:res, 2 * res:, :] = img1[0].permute(1, 2, 0).detach().cpu().numpy()
            # 1 0: img1 relit to match img2
            out_img[res:, :res, :] = img1_relit[0].permute(1, 2, 0).detach().cpu().numpy()
            # 1 1: img2 reconstruction
            out_img[res:, res:2 * res, :] = img2_recon[0].permute(1, 2, 0).detach().cpu().numpy()
            # 1 2: img2 gt
            out_img[res:, 2 * res:, :] = img2[0].permute(1, 2, 0).detach().cpu().numpy()
            out_img = out_img * np.array(img_std).reshape((1, 1, -1)) + np.array(img_mean).reshape((1, 1, -1))
            plt.imshow(out_img)
            plt.show()

    torch.save(lighting_extractor.state_dict(), "models/extractor_dpt.pth")
    torch.save(lighting_decoder.state_dict(), "models/decoder_dpt.pth")
