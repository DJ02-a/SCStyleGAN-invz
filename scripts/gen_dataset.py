import sys
sys.path.append('./')



import torch
from packages.stylegan2.model_ import Generator
from packages.face_parsing.main import FaceParser
from packages.sketch_ae.image2sketch.nets import Image2Sketch_AE
from torchvision import transforms
from tqdm import tqdm
import cv2
import numpy as np

G = Generator(512, 512, 8, channel_multiplier=2).cuda().eval()
G.load_state_dict(torch.load('./packages/stylegan2/ckpt/kface_512_40000.pt', map_location=torch.device('cuda'))['g_ema'], strict=False)
for param in G.parameters():
    param.requires_grad = False

I2S = Image2Sketch_AE(3,1).cuda().eval()
I2S.load_state_dict(torch.load('./packages/sketch_ae/train_result/only_l1/ckpt/E_latest.pt')['model'])
for param in I2S.parameters():
    param.requires_grad = False

faceparser = FaceParser().cuda().eval()

transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

avg_latent = G.mean_latent(10000)
avg_latent_ = avg_latent.cuda().unsqueeze(1).repeat(1,16,1)

save_path = './'
import os
os.makedirs('/home/jjy/dataset/k_face/train/img',exist_ok=True)
os.makedirs('/home/jjy/dataset/k_face/train/sketch',exist_ok=True)
os.makedirs('/home/jjy/dataset/k_face/train/label',exist_ok=True)
os.makedirs('/home/jjy/dataset/k_face/train/vis',exist_ok=True)
os.makedirs('/home/jjy/dataset/k_face/train/vector',exist_ok=True)


def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    return vis_parsing_anno_color

for i in tqdm(range(28000)):
    sample_z = torch.randn(1,512,device=torch.device('cuda')).unsqueeze(1).repeat(1,16,1)
    sample, vector = G(sample_z,truncation=0.7,truncation_latent=avg_latent_,return_latents=True)
    sketch = I2S(sample)
    semantic_map = faceparser.get_face_mask(sample)

    sample_img_ = (sample.clone().detach().squeeze().permute([1,2,0]).cpu().numpy()*.5+.5)*255
    sketch_ = (sketch.clone().detach().squeeze().clamp(0,1).cpu().numpy())*255
    semantic_map_ = (semantic_map.clone().detach().squeeze(0).cpu().numpy())
    vis = vis_parsing_maps(sample_img_,semantic_map_, stride=1)
    cv2.imwrite(f'/home/jjy/dataset/k_face/train/img/{str(i).zfill(6)}.png',sample_img_[:,:,::-1])
    cv2.imwrite(f'/home/jjy/dataset/k_face/train/sketch/{str(i).zfill(6)}.png',sketch_)
    cv2.imwrite(f'/home/jjy/dataset/k_face/train/label/{str(i).zfill(6)}.png',semantic_map_)
    cv2.imwrite(f'/home/jjy/dataset/k_face/train/vis/{str(i).zfill(6)}.png',vis)
    np.save(f'/home/jjy/dataset/k_face/train/vector/{str(i).zfill(6)}.npy',vector.cpu().numpy())
