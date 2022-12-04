import sys
sys.path.append('./')
import random
import numpy as np
import torch
from lib import utils
from lib.model_interface import ModelInterface
from packages import Generator
from SC_StyleGAN.nets import Spatial_E
from SC_StyleGAN.loss import MyModelLoss
from packages.image2sketch.model import Image2Sketch_AE
from packages.face_parsing.main import FaceParser
import torch.nn.functional as F


class SCStyleGAN(ModelInterface):
    def set_networks(self):
        self.G = Generator(self.args.img_size, 512, 8, channel_multiplier=2).cuda(self.gpu).eval()
        self.G.load_state_dict(torch.load(self.args.style_ckpt, map_location=torch.device('cuda'))['g_ema'], strict=False)
        for param in self.G.parameters():
            param.requires_grad = False
            
        self.I2S = Image2Sketch_AE(3,1).cuda(self.gpu).eval()
        self.I2S.load_state_dict(torch.load('./packages/image2sketch/ckpt/E_latest.pt',map_location='cuda')['model'])
        for param in self.I2S.parameters():
            param.requires_grad = False
            
        self.faceparsing = FaceParser()
        
        self.G.avg_latent = self.G.mean_latent(10000) # 
        self.E = Spatial_E().cuda(self.gpu).train()

    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.args)

    def go_step(self, global_step):

        # run G
        self.run_G()

        # update G
        loss_G = self.loss_collector.get_loss_G(self.dict)
        utils.update_net(self.opt_E, loss_G)
        # print images
        self.train_images = [
            F.interpolate(self.dict["gt_img"],(512,512)),
            F.interpolate(self.dict['sketch_vis'],(512,512)),
            F.interpolate(self.dict['label_vis'],(512,512)),
            F.interpolate(self.dict["fake_img"],(512,512)),
        ]

    def run_G(self):
        # gen synthesized GT
        sample_z = torch.randn(4, 512, device='cuda') # b, style dim
        gt_img, gt_latent, gt_feature_map = self.G([sample_z], truncation=.7, truncation_latent = self.G.avg_latent)

        _gt_img = F.interpolate(gt_img, (512,512))
        
        # gen sketch and semantic
        sketch = self.I2S(_gt_img)
        label = self.faceparsing(_gt_img)
        one_hot = self.to_one_hot(label)
        
        self.dict['sketch_vis'] = F.interpolate(sketch.repeat(1,3,1,1),(1024,1024))
        self.dict['label_vis'] = F.interpolate((label.type(torch.float32)/19).unsqueeze(1).repeat(1,3,1,1),(1024,1024))

        # gen fake image
        features = self.E(sketch, one_hot)
        fake_img, _, fake_feature_map = self.G(gt_latent, input_is_latent=True, layer_in=features, start_layer=4, end_layer=8)    
        
        self.dict["fake_img"] = fake_img
        self.dict["gt_img"] = gt_img
        
        self.dict["fake_feature_map"] = fake_feature_map
        self.dict["gt_feature_map"] = gt_feature_map

    def run_D(self):
        pass

    def do_validation(self, step):
        with torch.no_grad():
            result_images = self.G(self.valid_source, self.valid_target)[0]
        self.valid_images = [
            self.valid_source, 
            self.valid_target, 
            result_images
            ]

    @property
    def loss_collector(self):
        return self._loss_collector
        
    def to_one_hot(self, label_map):
        label_map_ = label_map.type(torch.float32)
        B, H, W = label_map_.shape
        label_map_ = F.interpolate(torch.reshape(label_map_,(B,1,H,W)),(512,512)).squeeze()
        label_map_ = label_map_.type(torch.int64)

        one_hot = torch.zeros((B, 19, 512, 512),dtype=torch.float32, device=label_map_.device)
        # cv2.imwrite('./test_label.png',(label_map[0].clone().detach().cpu().numpy()*10))
        # cv2.imwrite('./test_one_hot.png',(one_hot_.clone().detach().unsqueeze(-1).cpu().numpy()[0][0]*255))
        for i in range(B):
            one_hot[i].scatter_(0, label_map_[i].unsqueeze(0), 1.0)
        return one_hot