from pydoc import describe
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

def to_one_hot(sementic_map):
    sementic_map_ = torch.tensor(np.array(sementic_map),dtype=torch.float32)
    H, W = sementic_map_.shape
    sementic_map_ = F.interpolate(torch.reshape(sementic_map_,(1,1,H,W)),(512,512)).squeeze()
    sementic_map_ = torch.tensor(sementic_map_, dtype=torch.int64)


    one_hot = torch.zeros((19, 512, 512),dtype=torch.float32)
    one_hot = one_hot.scatter_(0, sementic_map_.unsqueeze(0), 1.0)
    return one_hot

# image 2 vector

G = Generator(512, 512, 8, channel_multiplier=2).cuda().eval()
# G.load_state_dict(torch.load('./packages/stylegan2/ckpt/ffhq.pt', map_location=torch.device('cuda'))['g_ema'], strict=False)
G.load_state_dict(torch.load('./packages/stylegan2/ckpt/kface_512_40000.pt', map_location=torch.device('cuda'))['g_ema'], strict=False)
for param in G.parameters():
    param.requires_grad = False

E = Spatial_E().cuda().eval()
# E.load_state_dict(torch.load('./train_result/6.celeba_l11_feat10_gl10_ll10_lre-5/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)
E.load_state_dict(torch.load('./train_result/5.kface_l11_feat10_gl10_ll10_lre-5/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)

for param in E.parameters():
    param.requires_grad = False

style_img_path = './assets/000025.png'
style_path = './assets/000025.npy'

count = 0
import glob
for label in ['hair','ear','front_hair','nose']:
    semantic_paths = sorted(glob.glob(f'./assets/{label}/semantic/*.*'))
    sketch_paths = sorted(glob.glob(f'./assets/{label}/sketch/*.*'))
    vis_paths = sorted(glob.glob(f'./assets/{label}/vis/*.*'))
    
    pbar = tqdm(range(len(semantic_paths)),desc=f'{label}_semantic')
    for i in pbar:
        semantic = Image.open(semantic_paths[i])
        sketch = Image.open(sketch_paths[0])
        style_img = Image.open(style_img_path)
        style_vector = torch.tensor(np.load(style_path)).cuda()

        semantic_ = to_one_hot(semantic)
        sketch_ = transforms2(sketch)

        # gen fake image
        features = E(sketch_.unsqueeze(0).cuda(), semantic_.unsqueeze(0).cuda())
        fake_img, fake_feature_map = G(style_vector, layer_in=features, start_layer=4, end_layer=8)
        fake_img_ = (fake_img.clone().detach().squeeze().permute([1,2,0]).cpu().numpy()*.5+.5)*255

        # vis
        vis_np = np.array(Image.open(vis_paths[i]))
        sketch_np = np.array(sketch)
        sketch_np_ = cv2.resize(sketch_np,(512,512))
        sketch_np_ = np.expand_dims(sketch_np_,axis=-1)
        mix = cv2.resize(vis_np*.5 + sketch_np_*.5,(512,512))
        style_img_ = cv2.resize(np.array(style_img),(512,512))
        fake_img_ = cv2.resize(np.array(fake_img_),(512,512))

        vis = np.concatenate((style_img_[:,:,::-1],mix[:,:,::-1],fake_img_[:,:,::-1]),axis=1)
        cv2.imwrite(f'./result3/{label}/{str(count).zfill(6)}.png',vis)
        count += 1
    pbar = tqdm(range(len(sketch_paths)),desc=f'{label}_sketch')

    for i in pbar:
        semantic = Image.open(semantic_paths[-1])
        sketch = Image.open(sketch_paths[i])
        style_img = Image.open(style_img_path)
        style_vector = torch.tensor(np.load(style_path)).cuda()

        semantic_ = to_one_hot(semantic)
        sketch_ = transforms2(sketch)


        # gen fake image
        features = E(sketch_.unsqueeze(0).cuda(), semantic_.unsqueeze(0).cuda())
        fake_img, fake_feature_map = G(style_vector, layer_in=features, start_layer=4, end_layer=8)
        fake_img_ = (fake_img.clone().detach().squeeze().permute([1,2,0]).cpu().numpy()*.5+.5)*255

        # vis
        vis_np = np.array(Image.open(vis_paths[-1]))
        sketch_np = np.array(sketch)
        sketch_np_ = cv2.resize(sketch_np,(512,512))
        sketch_np_ = np.expand_dims(sketch_np_,axis=-1)
        mix = cv2.resize(vis_np*.5 + sketch_np_*.5,(512,512))
        style_img_ = cv2.resize(np.array(style_img),(512,512))
        fake_img_ = cv2.resize(np.array(fake_img_),(512,512))

        vis = np.concatenate((style_img_[:,:,::-1],mix[:,:,::-1],fake_img_[:,:,::-1]),axis=1)
        cv2.imwrite(f'./result3/{label}/{str(count).zfill(6)}.png',vis)
        count += 1