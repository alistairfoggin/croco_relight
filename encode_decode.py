import torch
import torchvision
from PIL import Image
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, CenterCrop

from models.croco import CroCoNet

device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1, 3, 1, 1).to(device, non_blocking=True)
imagenet_std = [0.229, 0.224, 0.225]
imagenet_std_tensor = torch.tensor(imagenet_std).view(1, 3, 1, 1).to(device, non_blocking=True)
trfs = Compose([ToTensor(), CenterCrop(224), Normalize(mean=imagenet_mean, std=imagenet_std)])

def vis_results(ref_img, out_img):
    # patchified = model.patchify(ref_img)
    # mean = patchified.mean(dim=-1, keepdim=True)
    # var = patchified.var(dim=-1, keepdim=True)
    # decoded_image = model.unpatchify(model.patchify(out_img) * (var + 1.e-6)**.5 + mean)
    # undo imagenet normalization, prepare masked image
    decoded_image = out_img * imagenet_std_tensor + imagenet_mean_tensor
    input_image = ref_img * imagenet_std_tensor + imagenet_mean_tensor

    # make visualization
    visualization = torch.cat((input_image, decoded_image), dim=3) # 2*(B, 3, H, W) -> B, 3, H, W*2
    # visualization = torch.cat((ref_img, out_img), dim=3) # 2*(B, 3, H, W) -> B, 3, H, W*2
    B, C, H, W = visualization.shape
    visualization = visualization.permute(1, 0, 2, 3).reshape(C, B*H, W)
    visualization = torchvision.transforms.v2.functional.to_pil_image(torch.clamp(visualization, 0, 1))
    fname = "encode_output_average.png"
    visualization.save(fname)
    print('Visualization save in '+fname)

# load 224x224 images and transform them to tensor
image1 = trfs(Image.open('data/0076.png').convert('RGB')).to(device, non_blocking=True).unsqueeze(0)
image2 = trfs(Image.open('data/0196.png').convert('RGB')).to(device, non_blocking=True).unsqueeze(0)

ckpt = torch.load('pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth', 'cpu')
model = CroCoNet(**ckpt.get('croco_kwargs', {}), mask_ratio=0.5).to(device)
model.eval()
msg = model.load_state_dict(ckpt['model'], strict=True)

# with torch.inference_mode():
with torch.no_grad():
    feat1, pos1, mask1 = model._encode_image(image1, do_mask=False)
    feat2, pos2, mask2 = model._encode_image(image2, do_mask=False)
    print(feat1.shape, feat2.shape)
    print(pos1.shape, pos2.shape)
    feat_gt = (feat1 + feat2) / 2.

# out, mask, target = model(image1, image1, do_mask=False)
output_img = torch.nn.Parameter(torch.randn_like(image1), requires_grad=True)
out_optim = torch.optim.Adam([output_img], lr=1e-1)
loss_fn = torch.nn.MSELoss()

# for epoch in range(500):
#     out_optim.zero_grad()
#     feat_pred, pos_pred, mask_pred = model._encode_image(output_img, do_mask=False)
#     loss = loss_fn(feat_pred, feat_gt.detach())
#     loss.backward()
#     out_optim.step()
#     print(f'Epoch {epoch+1}, Loss {loss.item()}')
#
#
# vis_results(image1, output_img)
