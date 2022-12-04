import argparse
import DCL_transforms
import os
import pandas as pd
from dataset_CUB import dataset
from torchvision import transforms

import DCL_model

dataset_transforms = {}

def Init_transforms(args):
    global dataset_transforms
    dataset_transforms = {
        'swap': transforms.Compose([
            DCL_transforms.Randomswap(args.swap_window)
        ]),
        'img_aug': transforms.Compose([
            transforms.RandomRotation(degrees = 15),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(args.crop_reso),
            transforms.RandomCrop((args.crop_reso, args.crop_reso))
        ]),
        'totensor': transforms.Compose([
            transforms.Resize(args.crop_reso, args.crop_reso),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

def Init_dataset(args):
    global dataset_transforms
    
    ct_train_f = open('./anno/ct_train.txt', 'r')
    ct_val_f = open('./anno/ct_val.txt', 'r')
    train_anno = {'image_name': [], 'anno': []}
    val_anno = {'image_name': [], 'anno': []}
    
    lines = ct_train_f.readlines()
    for line in lines:
        line = line.split(sep = ' ')
        train_anno['image_name'].append(line[0])
        train_anno['anno'].append(int(line[1]))
    
    lines = ct_val_f.readlines()
    for line in lines:
        line = line.split(sep = ' ')
        val_anno['image_name'].append(line[0])
        val_anno['anno'].append(int(line[1]))

    path=os.path.join(args.data_dir,'data')

    tran_set = dataset(path, train_anno, swap=dataset_transforms["swap"], common_aug=dataset_transforms["img_aug"], train=True)
    val_set = dataset(path, val_anno, swap=None, common_aug=None, val=True)
    return tran_set, val_set
    


def Init_model(args):
    model = DCL_model.DCL_Network(args, is_train=True)
    return model

def Initialization():
    parser = argparse.ArgumentParser(description='DCL_arguments')

    parser.add_argument('--save_dir', default = None)
    parser.add_argument('--data_dir', default = None)
    parser.add_argument('--epoch_num', default = 180)
    parser.add_argument('--train_batchsize', default = 16)
    parser.add_argument('--eval_batchsize', default = 16)
    parser.add_argument('--eval_epoch', default = 10)
    parser.add_argument('--base_lr', default = 0.008)
    parser.add_argument('--decay_step', default = 60)
    parser.add_argument('--cls_lr_ratio', default = 10.0)
    parser.add_argument('--train_num_workers', default = 8)
    parser.add_argument('--eval_num_workers', default = 8)
    parser.add_argument('--crop_reso', default = 448)
    parser.add_argument('--swap_window', default = 7)
    parser.add_argument('--resnet50_path', default = None)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')

    args = parser.parse_args()

    args.save_dir = os.path.normpath(args.save_dir)
    args.data_dir = os.path.normpath(args.data_dir)
    args.resnet50_path = os.path.normpath(args.resnet50_path)

    Init_transforms(args)

    train_dataset, eval_dataset = Init_dataset(args)

    model = Init_model(args)

    return args, train_dataset, eval_dataset, model