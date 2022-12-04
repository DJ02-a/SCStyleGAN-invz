import sys
import os
sys.path.append("./")
import torch
import torchvision.transforms as transforms
from packages.stylegan2.model_ori import Generator
from SC_StyleGAN.nets import Spatial_E
from packages.psp_official.main import PSPEncoderOfficial
import cv2
from PIL import Image
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm

transforms2 = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5),(0.5))
        ])

def to_one_hot(semantic_map):
    semantic_map_ = torch.tensor(np.array(semantic_map),dtype=torch.float32)
    H, W = semantic_map_.shape
    semantic_map_ = F.interpolate(torch.reshape(semantic_map_,(1,1,H,W)),(512,512)).squeeze()
    semantic_map_ = torch.tensor(semantic_map_, dtype=torch.int64)


    one_hot = torch.zeros((19, 512, 512),dtype=torch.float32)
    one_hot_ = one_hot.scatter_(0, semantic_map_.unsqueeze(0), 1.0)
    return one_hot_

# image 2 vector

G = Generator(512, 512, 8, channel_multiplier=2).cuda().eval()
# G.load_state_dict(torch.load('./packages/stylegan2/ckpt/ffhq.pt', map_location=torch.device('cuda'))['g_ema'], strict=False)
G.load_state_dict(torch.load('./packages/stylegan2/ckpt/kface_512_40000.pt', map_location=torch.device('cuda'))['g_ema'], strict=False)
for param in G.parameters():
    param.requires_grad = False

E = Spatial_E().cuda().eval()
# E.load_state_dict(torch.load('./train_result/6.celeba_l11_feat10_gl10_ll10_lre-5/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)
# E.load_state_dict(torch.load('./train_result/5.kface_l11_feat10_gl10_ll10_lre-5/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)
E.load_state_dict(torch.load('/home/jjy/workspace/SC-StyleGAN/train_result/10.kface_l11_feat10_gl10_ll10_lre-5_onlymask/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)

for param in E.parameters():
    param.requires_grad = False

semantic_path = './assets/recon2/semantic2.png'
sketch_path = './assets/recon2/sketch.png'
vis_path = './assets/recon2/vis2.png'
style_img_path = '/home/jjy/workspace/SC-StyleGAN/assets/000025.png'
style_path = '/home/jjy/workspace/SC-StyleGAN/assets/000025.npy'

semantic = Image.open(semantic_path)
sketch = Image.open(sketch_path)
style_img = Image.open(style_img_path)
style_vector = torch.tensor(np.load(style_path)).cuda()

semantic_ = to_one_hot(semantic)
sketch_ = transforms2(sketch)


# gen fake image
recon_img, _ = G(style_vector, start_layer=0, end_layer=8)
recon_img_ = (recon_img.clone().detach().squeeze().permute([1,2,0]).cpu().numpy()*.5+.5)*255

features = E(sketch_.unsqueeze(0).cuda(), semantic_.unsqueeze(0).cuda())
fake_img, fake_feature_map = G(style_vector, layer_in=features, start_layer=4, end_layer=8)
fake_img_ = (fake_img.clone().detach().squeeze().permute([1,2,0]).cpu().numpy()*.5+.5)*255

# vis
vis_np = np.array(Image.open(vis_path))
sketch_np = np.array(sketch)
sketch_np_ = cv2.resize(sketch_np,(512,512))
sketch_np_ = np.expand_dims(sketch_np_,axis=-1)
mix = cv2.resize(vis_np*.5 + sketch_np_*.5,(512,512))
recon_img_ = cv2.resize(np.array(recon_img_),(512,512))
style_img_ = cv2.resize(np.array(style_img),(512,512))
fake_img_ = cv2.resize(np.array(fake_img_),(512,512))

vis = np.concatenate((recon_img_[:,:,::-1],mix[:,:,::-1],fake_img_[:,:,::-1]),axis=1)
cv2.imwrite('./testing3.png',vis)