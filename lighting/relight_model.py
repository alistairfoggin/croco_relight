from functools import partial

import torch
from torch import nn

from lighting.relight import LightingExtractor
from models.blocks import Block
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
        self._set_lighting_decoder(self.enc_embed_dim, self.dec_embed_dim, 16, 4, 2, partial(nn.LayerNorm, eps=1e-6), False)
            # self._set_decoder(1024, 1024, 16, 8, 2, partial(nn.LayerNorm, 1e-6), False)
        self.lighting_extractor = LightingExtractor(patch_size=self.enc_embed_dim, rope=self.rope)
        # step = self.dec_depth//4
        hooks_idx = [3, 2, 1, 0]
        head = PixelwiseTaskWithDPT(hooks_idx=hooks_idx, num_channels=3)
        head.setup(self, dim_tokens=self.dec_embed_dim)
        self.head = head

    def _set_lighting_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        # must overwrite decoder blocks
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            Block(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def _lighting_decoder(self, feat, pos, dyn_feat, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # encoder to decoder layer
        f1_ = self.decoder_embed(feat + dyn_feat)
        # append masked tokens to the sequence
        B, Nenc, C = f1_.size()
        # add positional embedding
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
        # apply Transformer blocks
        out = f1_
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out = blk(_out, pos)
                out.append(_out)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out = blk(out, pos)
            out = self.dec_norm(out)
        return out

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
        out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0), do_mask=False,
                                         return_all_blocks=return_all_blocks)
        return out, pos

    def forward(self, img1, img2):
        B, C, H1, W1 = img1.size()
        _, _, H2, W2 = img2.size()
        img1_info = {'height': H1, 'width': W1}
        img2_info = {'height': H2, 'width': W2}
        return_all_blocks = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks
        # feat1, feat2, pos1, pos2 = self.encode_image_pairs(img1, img2, return_all_blocks=False)
        feat, pos = self.encode_image_pairs(img1, img2, return_all_blocks=False)

        static, dyn, dyn_pos = self.lighting_extractor(feat, pos)
        # static1, dyn1, dyn_pos1 = self.lighting_extractor(feat1, pos1)
        # static2, dyn2, dyn_pos2 = self.lighting_extractor(feat2, pos2)

        # Swap dyn1 and dyn2
        dyn1, dyn2 = dyn.chunk(2, dim=0)
        # dyn_pos1, dyn_pos2 = dyn_pos.chunk(2, dim=0)
        swapped_dyn = torch.cat((dyn2, dyn1), dim=0)
        # swapped_dyn_pos = torch.cat((dyn_pos2, dyn_pos1), dim=0)
        zero_dyn = torch.zeros_like(swapped_dyn, device=swapped_dyn.device)

        # Relight img 1 to be like img 2
        decout = self._lighting_decoder(static, pos, swapped_dyn, return_all_blocks=return_all_blocks)
        zero_decout = self._lighting_decoder(static, pos, zero_dyn, return_all_blocks=return_all_blocks)
        # decout1, decout2 = decout.chunk(2, dim=0)
        if return_all_blocks:
            decout1, decout2 = list(map(list, zip(*[o.chunk(2, dim=0) for o in decout])))
            zero_decout1, zero_decout2 = list(map(list, zip(*[o.chunk(2, dim=0) for o in zero_decout])))
            # decout2 = decout2[-1]
        else:
            decout1, decout2 = decout.chunk(2, dim=0)
            zero_decout1, zero_decout2 = zero_decout.chunk(2, dim=0)
        static1, static2 = static.chunk(2, dim=0)
        # decout1 = self._decoder(static1, pos1, None, dyn2, dyn_pos2, return_all_blocks=return_all_blocks)
        # decout2 = self._decoder(static2, pos2, None, dyn1, dyn_pos1, return_all_blocks=return_all_blocks)
        return (self.head(decout1, img1_info), self.head(decout2, img2_info),
                static1, static2,
                self.head(zero_decout1, img1_info), self.head(zero_decout2, img2_info))
