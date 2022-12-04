from re import L
import sys
import os
from syslog import LOG_SYSLOG
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

from packages.lpips.lpips import LPIPS


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

def more_detail(style_np, style_img, iter=10):
    # def cal_p_norm_loss(latent_in):
    #     latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - X_mean).bmm(
    #         X_comp.T.unsqueeze(0)) / X_stdev
    #     p_norm_loss = opts.p_norm_lambda * (latent_p_norm.pow(2).mean())
    #     return p_norm_loss

    def loss_cal(img1, img2, latent_):
        loss = 0.0

        lpips = LPIPS().eval().to("cuda")
        L2 = torch.nn.MSELoss().to("cuda")

        loss += lpips(img1, img2)
        loss += L2(img1, img2)
        # loss += cal_p_norm_loss(latent_)

        return loss

    # setting...
    latent = torch.tensor(style_np + avg, requires_grad=True)
    latent_ = latent.cuda()

    style_img_ = transforms2(style_img).cuda()

    opt = torch.optim.Adam([latent_], lr=0.0005)

    print('optimizing style vector...')
    pbar = tqdm(range(iter))
    for i in pbar:
        opt.zero_grad()

        fake_img, _ = G(latent_, start_layer=0, end_layer=8)
        loss = loss_cal(style_img_.unsqueeze(0), fake_img, latent_) # lpips, l2
        
        loss.backward()
        opt.step()
        pbar.set_description(f'loss : {round(loss.item(), 3)}')
        
        # test
        img = (fake_img.clone().detach().squeeze().permute([1,2,0]).cpu().numpy()*.5+.5)*255
        img_ = np.concatenate((np.array(style_img.resize((512,512)))[:,:,::-1],img[:,:,::-1]),axis=1)
        cv2.imwrite(f'./test_{i}.png',img_)

    return latent

# image 2 vector

G = Generator(512, 512, 8, channel_multiplier=2).cuda().eval()
# G.load_state_dict(torch.load('./packages/stylegan2/ckpt/ffhq.pt', map_location=torch.device('cuda'))['g_ema'], strict=False)
G.load_state_dict(torch.load('./packages/stylegan2/ckpt/kface_512_40000.pt', map_location=torch.device('cuda'))['g_ema'], strict=False)
for param in G.parameters():
    param.requires_grad = False
avg = G.mean_latent(1000)
E = Spatial_E().cuda().eval()
# E.load_state_dict(torch.load('./train_result/6.celeba_l11_feat10_gl10_ll10_lre-5/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)
# E.load_state_dict(torch.load('./train_result/5.kface_l11_feat10_gl10_ll10_lre-5/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)
E.load_state_dict(torch.load('./train_result/7.kface_l11_feat10_gl10_ll10_lre-5_mask/ckpt/E_latest.pt', map_location=torch.device('cuda'))['model'], strict=False)

for param in E.parameters():
    param.requires_grad = False

sketch_path = '/home/jjy/workspace/SC-StyleGAN/assets/k-celeb-front/sketch/004.png'
# sketch_path = '/home/jjy/workspace/SC-StyleGAN/assets/k-celeb-front/sketch/004.png'
semantic_path = '/home/jjy/workspace/SC-StyleGAN/assets/k-celeb-front/semantic/004.png'
# semantic_path = '/home/jjy/workspace/SC-StyleGAN/assets/k-celeb-front/semantic/004.png'
vis_path = '/home/jjy/workspace/SC-StyleGAN/assets/k-celeb-front/vis/004.png'
style_img_path = '/home/jjy/workspace/SC-StyleGAN/assets/k-celeb-front/image/003.png'
style_path = '/home/jjy/workspace/SC-StyleGAN/assets/k-celeb-front/vector/003.npy'

# semantic_path = './assets/test/test1.png'
# sketch_path = './assets/test/test.png'
# vis_path = './assets/test/test2.png'
# style_img_path = './assets/test/test3.png'
# style_path = './assets/test/test.npy'

# semantic_path = './assets/celeba/semantic/00000.png'
# sketch_path = './assets/celeba/sketch/00000.png'
# style_img_path = './assets/image.png'
# style_path = './assets/vector.npy'
# vis_path = './assets/celeba/vis/00000.png'

semantic = Image.open(semantic_path)
sketch = Image.open(sketch_path)
style_img = Image.open(style_img_path)
style_vector = torch.tensor(np.load(style_path)).cuda()
more_detail(style_vector, style_img,100)
semantic_ = to_one_hot(semantic)
sketch_ = transforms2(sketch)

# gen fake image
features = E(sketch_.unsqueeze(0).cuda(), semantic_.unsqueeze(0).cuda())
fake_img, fake_feature_map = G(avg + style_vector, layer_in=features, start_layer=4, end_layer=8)
fake_img_ = (fake_img.clone().detach().squeeze().permute([1,2,0]).cpu().numpy()*.5+.5)*255

# vis
vis_np = np.array(Image.open(vis_path))
sketch_np = np.array(sketch)
sketch_np_ = cv2.resize(sketch_np,(512,512))
sketch_np_ = np.expand_dims(sketch_np_,axis=-1)
mix = cv2.resize(vis_np*.5 + sketch_np_*.5,(512,512))
style_img_ = cv2.resize(np.array(style_img),(512,512))
fake_img_ = cv2.resize(np.array(fake_img_),(512,512))

vis = np.concatenate((style_img_[:,:,::-1],mix[:,:,::-1],fake_img_[:,:,::-1]),axis=1)
cv2.imwrite('./test_003_004.png',vis)