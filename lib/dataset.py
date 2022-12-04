import os
import glob
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from lib import utils
import numpy as np
import torch.nn.functional as F
import cv2

class SingleFaceDatasetTrain(Dataset):
    def __init__(self, dataset_root_list, isMaster):
        self.image_path_list = sorted(glob.glob(dataset_root_list+'/img/*.*'))
        self.seg_path_list = sorted(glob.glob(dataset_root_list+'/label/*.*'))
        self.vis_path_list = sorted(glob.glob(dataset_root_list+'/vis/*.*'))
        self.sketch_path_list = sorted(glob.glob(dataset_root_list+'/sketch/*.*'))
        self.vector_path_list = sorted(glob.glob(dataset_root_list+'/vector/*.*'))
        self.image_num_list = len(self.image_path_list)

        self.transforms1 = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transforms2 = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):
        # img = Image.open(self.image_path_list[item])
        # seg = Image.open(self.seg_path_list[item])
        # vis = Image.open(self.vis_path_list[item])
        # sketch = Image.open(self.sketch_path_list[item])
        # high_vector = torch.tensor(np.load(self.vector_path_list[item]))
        # high_vector = high_vector.repeat(8,1)
        # low_vector = torch.randn(18,512)
        random_vector = torch.randn(16,512)

        # seg_ = self.to_one_hot(seg)

        # return self.transforms1(img), seg_, self.transforms1(vis), self.transforms2(sketch), high_vector, low_vector
        return random_vector

    def __len__(self):
        return  self.image_num_list

    def to_one_hot(self, semantic_map):
        semantic_map_ = torch.tensor(np.array(semantic_map),dtype=torch.float32)
        H, W = semantic_map_.shape
        semantic_map_ = F.interpolate(torch.reshape(semantic_map_,(1,1,H,W)),(512,512)).squeeze()
        semantic_map_ = torch.tensor(semantic_map_, dtype=torch.int64)

        one_hot = torch.zeros((19, 512, 512),dtype=torch.float32)
        one_hot_ = one_hot.scatter_(0, semantic_map_.unsqueeze(0), 1.0)
        return one_hot_


class SingleFaceDatasetValid(Dataset):
    def __init__(self, valid_data_dir, isMaster):
        
        self.source_path_list = sorted(glob.glob(f"{valid_data_dir}/source/*.*g"))
        self.target_path_list = sorted(glob.glob(f"{valid_data_dir}/target/*.*g"))
        self.image_path_list = self.source_path_list + self.target_path_list
        self.num = len(self.image_path_list)
        
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the validation.")

    def __getitem__(self, idx):
        
        image = Image.open(self.image_path_list[idx]).convert("RGB")

        return self.transforms(image)

    def __len__(self):
        return self.num



class DoubleFaceDatasetTrain(Dataset):
    def __init__(self, dataset_root_list, isMaster, same_prob=0.2):
        self.same_prob = same_prob
        self.image_path_list, self.image_num_list = utils.get_all_images(dataset_root_list)

        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):
        idx = 0
        while item >= self.image_num_list[idx]:
            item -= self.image_num_list[idx]
            idx += 1
        image_path = self.image_path_list[idx][item]
        
        Xs = Image.open(image_path).convert("RGB")

        if random.random() > self.same_prob:
            image_path = random.choice(self.image_path_list[random.randint(0, len(self.image_path_list)-1)])
            Xt = Image.open(image_path).convert("RGB")
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return sum(self.image_num_list)


class DoubleFaceDatasetValid(Dataset):
    def __init__(self, valid_data_dir, isMaster):
        
        self.source_path_list = sorted(glob.glob(f"{valid_data_dir}/source/*.*g"))
        self.target_path_list = sorted(glob.glob(f"{valid_data_dir}/target/*.*g"))

        # take the smaller number if two dirs have different numbers of images
        self.num = min(len(self.source_path_list), len(self.target_path_list))
        
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the validation.")

    def __getitem__(self, idx):
        
        Xs = Image.open(self.source_path_list[idx]).convert("RGB")
        Xt = Image.open(self.target_path_list[idx]).convert("RGB")

        return self.transforms(Xs), self.transforms(Xt)

    def __len__(self):
        return self.num


