import sys
import os
sys.path.append("./")
import torch
import torchvision.transforms as transforms
from packages.stylegan2.model_ori import Generator
from SC_StyleGAN.nets import Spatial_E
from packages.psp_official.main import PSPEncoderOfficial
from packages.face_parsing.utils import *
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

def swap_part(semantic_img1, semantic_img2, sketch_img1, sketch_img2, target='hair'):
    label = 17
    
    semantic_img1_, semantic_img2_ = np.array(semantic_img1), np.array(semantic_img2)
    sketch_img1_, sketch_img2_ = np.array(sketch_img1.resize((512,512))), np.array(sketch_img2.resize((512,512)))


    semantic_img1_[label] = semantic_img2_[label]

    GaussianBlur = transforms.GaussianBlur(kernel_size=25, sigma=(0.1, 5))
    mask = np.array(GaussianBlur(Image.fromarray(semantic_img2_[label])))/255
    sketch_img = sketch_img2_ * mask + sketch_img1_ * (1-mask)

    new_semantic = torch.tensor(semantic_img1_)
    new_sketch_img = Image.fromarray(sketch_img).convert("L")

    return new_semantic, new_sketch_img

# image 2 vector

G = Generator(512, 512, 8, channel_multiplier=2).cuda().eval()
# G.load_state_dict(torch.load('./packages/stylegan2/ckpt/ffhq.pt', map_location=torch.device('cuda'))['g_ema'], strict=False)
G.load_state_dict(torch.load('./packages/stylegan2/ckpt/kface_512_40000.pt', map_location=torch.device('cuda'))['g_ema'], strict=False)
for param in G.parameters():
    param.requires_grad = False

E = Spatial_E().cuda().eval()
# E.load_state_dict(torch.load('./train_result/6.celeba_l11_feat10_gl10_ll10_lre-5/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)
# E.load_state_dict(torch.load('./train_result/5.kface_l11_feat10_gl10_ll10_lre-5/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)
E.load_state_dict(torch.load('./train_result/7.kface_l11_feat10_gl10_ll10_lre-5_mask/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)

for param in E.parameters():
    param.requires_grad = False

# Image 1
semantic_path_img1 = './assets/recon1/semantic.png'
sketch_path_img1 = './assets/recon1/sketch.png'
vis_path_img1 = './assets/recon1/vis.png'
style_img_path_img1 = './assets/recon1/image.png'
style_path_img1 = './assets/recon1/vector.npy'

# Image 2
semantic_path_img2 = './assets/recon2/semantic.png'
sketch_path_img2 = './assets/recon2/sketch.png'
vis_path_img2 = './assets/recon2/vis.png'
style_img_path_img2 = './assets/recon2/image.png'
style_path_img2 = './assets/recon2/vector.npy'

# pp img 1
semantic_img1 = Image.open(semantic_path_img1)
sketch_img1 = Image.open(sketch_path_img1)
style_img1 = Image.open(style_img_path_img1)
style_vector_img1 = torch.tensor(np.load(style_path_img1)).cuda()

semantic_img1_ = to_one_hot(semantic_img1)

# pp img 2
semantic_img2 = Image.open(semantic_path_img2)
sketch_img2 = Image.open(sketch_path_img2)
style_img2 = Image.open(style_img_path_img2)
style_vector_img2 = torch.tensor(np.load(style_path_img2)).cuda()

semantic_img2_ = to_one_hot(semantic_img2)

# swap
new_semantic, new_sketch_img = swap_part(semantic_img1_, semantic_img2_, sketch_img1, sketch_img2, 'hair')
new_sketch_img_ = transforms2(new_sketch_img)


# gen fake image
features = E(new_sketch_img_.unsqueeze(0).cuda(), new_semantic.unsqueeze(0).cuda())
fake_img, fake_feature_map = G(style_vector_img2, layer_in=features, start_layer=4, end_layer=8)
fake_img_ = (fake_img.clone().detach().squeeze().permute([1,2,0]).cpu().numpy()*.5+.5)*255

# vis
vis_np = np.array(Image.open(vis_path_img1))
sketch_np = np.array(sketch_img1)
sketch_np_ = cv2.resize(sketch_np,(512,512))
sketch_np_ = np.expand_dims(sketch_np_,axis=-1)

# mix = cv2.resize(vis_np*.5 + sketch_np_*.5,(512,512))
tmp = np.expand_dims(np.array(new_sketch_img),axis=-1)

mix = cv2.resize(tmp*.5 + vis_np*.5,(512,512))
style_img_ = cv2.resize(np.array(style_img2),(512,512))
fake_img_ = cv2.resize(np.array(fake_img_),(512,512))

vis = np.concatenate((style_img_[:,:,::-1],mix[:,:,::-1],fake_img_[:,:,::-1]),axis=1)
cv2.imwrite('./test_swap.png',vis)
cv2.imwrite('./tmp2.png',np.array(new_semantic)[17]*255)