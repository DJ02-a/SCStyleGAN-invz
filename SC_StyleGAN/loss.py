from lib.loss_interface import Loss, LossInterface
import torch
import torch.nn.functional as F

class MyModelLoss(LossInterface):
    def get_loss_G(self, dict):
        L_G = 0.0
        
        # l1 loss
        if self.args.W_l1:
            L_l1 = Loss.get_L1_loss(dict["fake_img"], dict["gt_img"])
            L_G += self.args.W_l1 * L_l1
            self.loss_dict["L_l1"] = round(L_l1.item(), 4)
        
        # feat loss
        if self.args.W_feat:
            L_feat = .0
            for i in range(len(dict["fake_feature_map"])):
                L_feat += Loss.get_L1_loss(dict["fake_feature_map"][i], dict["gt_feature_map"][i])
            L_G += self.args.W_feat * (L_feat / float(len(dict["fake_feature_map"])))
            self.loss_dict["L_feat"] = round(L_feat.item(), 4)
            
        # global LPIPS loss
        if self.args.W_global_lpips:
            re_fake_img, re_gt_img = F.interpolate(dict["fake_img"],(64,64)), F.interpolate(dict["gt_img"],(64,64))
            L_global_lpips = Loss.get_lpips_loss(re_fake_img, re_gt_img)
            L_G += self.args.W_global_lpips * L_global_lpips
            self.loss_dict["L_global_lpips"] = round(L_global_lpips.item(), 4)

        # # local LPIPS loss
        if self.args.W_local_lpips:
            L_local_lpips = .0
            for i in range(20):
                patch_fake_img, patch_gt_img = self.get_path(dict["fake_img"], dict["gt_img"])
                L_local_lpips += Loss.get_lpips_loss(patch_fake_img, patch_gt_img)
                # random crop
            L_G += self.args.W_local_lpips * (L_local_lpips / 20.0 )
            self.loss_dict["L_local_lpips"] = round(L_local_lpips.item(), 4)

        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, dict):
        L_real = (F.relu(1 - dict["d_real"])).mean()
        L_fake = (F.relu(1 + dict["d_fake"])).mean()
        L_D = L_real + L_fake
        
        self.loss_dict["L_real"] = round(L_real.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D

    def get_path(self, img1, img2, patch_size=256):
        B, C, H, W = img1.size()
        s = patch_size
        y_offset = torch.randint(0,W // s, (1,), device=img1.device)
        x_offset = torch.randint(0,W // s, (1,), device=img2.device)
        patch_img1 = img1[:, :,
                    y_offset*s:(y_offset+1)*s,
                    x_offset*s:(x_offset+1)*s]
        patch_img2 = img2[:, :,
                    y_offset*s:(y_offset+1)*s,
                    x_offset*s:(x_offset+1)*s]
        # patch_img = img.view(B, C, H//s, s, W//s, s)
        return patch_img1, patch_img2

        