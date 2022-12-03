import argparse
import transforms
import os
import pandas as pd
from .dataset_CUB import dataset
import model

def Init_transforms(args):
    dataset_transforms = {
        'swap': transforms.Compose([
            transforms.Randomswap(args.swap_window)
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
    return dataset_transforms

def Init_dataset(args):
    train_anno=pd.readcsv(os.path.join((args.data_dir, 'anno/ct_train.txt'),sep=" ",header=None,names=['Imagename', 'label']))
    val_anno=pd.readcsv(os.path.join((args.data_dir, 'anno/ct_val.txt'),sep=" ",header=None,names=['Imagename', 'label']))
    path=os.path.join(args.data_dir,'data')
    transformers = Init_transforms(args)
    tran_set = dataset(path, train_anno, swap=transformers["swap"], common_aug=transformers["img_aug"], train=True)
    val_set = dataset(path, val_anno, swap=transformers["None"], common_aug=transformers["None"], val=True)
    return tran_set, val_set
    


def Init_model(args):


def Initialization():
    parser = argparse.ArgumentParser(description='DCL_arguments')

    parser.add_argument('--save_dir', default = None)
    parser.add_argument('--data_dir', default = None)
    parser.add_argument('--epoch_num', default = 180)
    parser.add_argument('--train_batchsize', default = 16)
    parser.add_argument('--eval_batchsize', default = 16)
    parser.add_argument('--save_epoch', default = 10)
    parser.add_argument('--eval_epoch', default = 10)
    parser.add_argument('--base_lr', default = 0.008)
    parser.add_argument('--decay_step', default = 60)
    parser.add_argument('--cls_lr_ratio', default = 10.0)
    parser.add_argument('--train_num_workers', default = 16)
    parser.add_argument('--eval_num_workers', default = 16)
    parser.add_argument('--crop_reso', default = 448)
    parser.add_argument('--swap_window', default = 7)

    args = parser.parse_args()

    dataset_transforms = Init_transforms(args)

    train_dataset, eval_dataset = Init_dataset(args)

    model = Init_model(args)

    return args