from functools import partial

import torch
from torch import nn

from lighting.relight import LightingExtractor, LightingEntangler
from models.blocks import Block
from models.croco import CroCoNet
from models.head_downstream import PixelwiseTaskWithDPT


class CroCoDecode(CroCoNet):

    def __init__(self, pretrained_model=None, **kwargs):
        """ Build network for binocular downstream task
        It takes an extra argument head, that is called with the features
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        """
        super(CroCoDecode, self).__init__(**kwargs)
        if pretrained_model is not None:
            self.load_state_dict(pretrained_model, strict=True)
            # Reset decoder and freeze encoder
            self.freeze_encoder()
            self.mask_token = None
            self.prediction_head = None
        self._set_mono_decoder(self.enc_embed_dim, self.enc_embed_dim, 16, 4, 2, partial(nn.LayerNorm, eps=1e-6))
        # regular decoder is now lighting mixer
        self._set_decoder(self.enc_embed_dim, self.enc_embed_dim, 16, 2, 2, partial(nn.LayerNorm, eps=1e-6), False)

        self.lighting_extractor = LightingExtractor(patch_size=self.enc_embed_dim, rope=self.rope)
        hooks_idx = [3, 2, 1, 0]
        head = PixelwiseTaskWithDPT(hooks_idx=hooks_idx, num_channels=3)
        head.setup(self, dim_tokens=self.enc_embed_dim)
        self.head = head

    def freeze_encoder(self):
        self.enc_blocks.requires_grad_(False)
        self.patch_embed.requires_grad_(False)
        self.enc_norm.requires_grad_(False)

    def _set_mono_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer):
        self.out_dec_depth = dec_depth
        self.out_dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        self.out_decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.out_dec_blocks = nn.ModuleList([
            Block(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.out_dec_norm = norm_layer(dec_embed_dim)

    def _mono_decoder(self, feat, pos, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # encoder to decoder layer
        f1_ = self.out_decoder_embed(feat)
        # apply Transformer blocks
        out = f1_
        if return_all_blocks:
            _out, out = out, []
            for blk in self.out_dec_blocks:
                _out = blk(_out, pos)
                out.append(_out)
            out[-1] = self.out_dec_norm(out[-1])
        else:
            for blk in self.out_dec_blocks:
                out = blk(out, pos)
            out = self.out_dec_norm(out)
        return out

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def encode_image_pairs(self, img1, img2, return_all_blocks=False):
        """ run encoder for a pair of images
            it is actually ~5% faster to concatenate the images along the batch dimension
             than to encode them separately
        """
        out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0), do_mask=False,
                                         return_all_blocks=return_all_blocks)
        return out, pos

    def encode_decode(self, img):
        _, _, H, W = img.size()
        img_info = {'height': H, 'width': W}
        latents, pos, _ = self._encode_image(img, do_mask=False, return_all_blocks=False)
        decout = self._mono_decoder(latents, pos, return_all_blocks=True)
        return self.head(decout, img_info)

    def decode(self, latents, pos, img_info):
        decout = self._mono_decoder(latents, pos, return_all_blocks=True)
        return self.head(decout, img_info)


class RelightModule(nn.Module):
    def __init__(self, croco: CroCoNet):
        super(RelightModule, self).__init__()
        self.croco = croco
        self.croco.requires_grad_(False)
        self.lighting_extractor = LightingExtractor(patch_size=croco.enc_embed_dim, extractor_depth=3, rope=croco.rope)
        self.lighting_entangler = LightingEntangler(patch_size=croco.enc_embed_dim, extractor_depth=3, rope=croco.rope)
        

    def forward(self, img1, img2):
        B, C, H1, W1 = img1.size()
        _, _, H2, W2 = img2.size()
        img1_info = {'height': H1, 'width': W1}
        img2_info = {'height': H2, 'width': W2}
        feat, pos = self.croco.encode_image_pairs(img1, img2, return_all_blocks=False)

        static, dyn, dyn_pos = self.lighting_extractor(feat, pos)

        # Swap dyn1 and dyn2
        dyn1, dyn2 = dyn.chunk(2, dim=0)
        swapped_dyn = torch.cat((dyn2, dyn1), dim=0)

        # Relight img 1 to be like img 2
        relit_feat, _ = self.lighting_entangler(static, pos, swapped_dyn, dyn_pos)
        # recon_feat = self.lighting_entangler(static, pos, dyn, dyn_pos)

        # recon_img = self.croco.decode(recon_feat, pos, img1_info)
        relit_img = self.croco.decode(relit_feat, pos, img2_info)

        # recon_img1, recon_img2 = recon_img.chunk(2, dim=0)
        relit_img1, relit_img2 = relit_img.chunk(2, dim=0)

        static1, static2 = static.chunk(2, dim=0)

        # relit_feat1, relit_feat2 = relit_feat.chunk(2, dim=0)
        # new_feat = torch.cat((relit_feat2, relit_feat1), dim=0)

        return (relit_img1, relit_img2,
                static1, static2)#, feat, new_feat,
                # recon_img1, recon_img2)
