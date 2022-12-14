from __future__ import division
import os
from torchvision.transforms import ToTensor
import torch
import torch.utils.data as data
import pandas
import random
import PIL.Image as Image
from PIL import ImageStat

class dataset(data.Dataset):
    def __init__(self, data_path, anno, swap_size=[7,7], totensor=None, swap=None, common_aug=None,  train=False, val=False):
        self.root_path = data_path
        self.swap_size=swap_size
        self.common_aug=common_aug
        self.train=train
        self.totensor=totensor
        self.val=val
        self.path = anno['image_name']
        self.labels = anno['anno']
        self.swap=swap
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        paths=os.path.join(self.root_path, self.path[item])
        img_unswap=Image.open(paths).convert('RGB')
        if self.train:
            img_unswap = self.common_aug(img_unswap)
        unswap_regions=self.crop_image(img_unswap, self.swap_size)
        ssize=self.swap_size[0]*self.swap_size[1]
        origin_loc=[(i-ssize//2)/ssize for i in range(0, ssize)]
        label=self.labels[item]
        swap_label=label+200
        img_unswap=self.totensor(img_unswap)
        if self.val:
            return img_unswap, label, swap_label, origin_loc
        if self.train:
            swap_image, swap_loc=self.swap(unswap_regions)
            swap_image=self.totensor(swap_image)
            #swap_crop=self.crop_image(swap_image, self.swap_size)
            return img_unswap, swap_image, label, swap_label, origin_loc, swap_loc 

    def crop_image(self, img, size):
        w, h = img.size
        px = [int(w/size[0])*i for i in range(0, size[0]+1)]
        py = [int(h/size[1])*i for i in range(0, size[1]+1)] 
        ret_list=[]
        for j in range(len(py) - 1):
            for i in range(len(px) - 1):
                ret_list.append(img.crop((px[i], py[j], min(px[i + 1], w), min(py[j + 1], h))))
        return ret_list

def collate_train(batch):
    imgs = []
    cls_labels = []
    adv_labels = []
    loc = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        cls_labels.append(sample[2])
        cls_labels.append(sample[2])
        adv_labels.append(sample[2])
        adv_labels.append(sample[3])
        loc.append(sample[4])
        loc.append(sample[5])
        
    cls_labels = torch.Tensor(cls_labels)
    adv_labels = torch.Tensor(adv_labels)
    loc = torch.Tensor(loc)
    return torch.stack(imgs, 0), cls_labels, adv_labels, loc
def collate_val(batch):
    imgs = []
    cls_labels = []
    adv_labels = []
    loc = []
    for sample in batch:
        imgs.append(sample[0])
        cls_labels.append(sample[1])
        adv_labels.append(sample[2])
        loc.append(sample[3])
    cls_labels = torch.Tensor(cls_labels)
    adv_labels = torch.Tensor(adv_labels)
    loc = torch.Tensor(loc)
    return torch.stack(imgs,0), cls_labels, adv_labels, loc
    
