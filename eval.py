from __future__ import print_function, division
import os,time,datetime
import numpy as np
import datetime
from math import ceil
import torch
from torch import nn
def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def eval(model,data_loader,epoch_num):
    model.train(False)
    model.istrain=True
    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects3 = 0
    item_count = 0
    val_size = data_loader.__len__
    t0 = time.time()
    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    with torch.no_grad():
        for batch_cnt, data in enumerate(data_loader):
            input = data[0]
            labels = data[1]
            
            labels = labels.to('cuda').long()
            
            cls_out, adv_out, con_out = model(input)
            pro_out = cls_out + adv_out[:,0:200] + adv_out[:, 200:400]
            top3_val, top3_pos = torch.topk(pro_out, 3)
            
            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)
            
            item_count += top3_pos.size(0)
            
        print('item_count: ', item_count)
        val_acc1 = val_corrects1 / item_count
        val_acc2 = val_corrects2 / item_count
        val_acc3 = val_corrects3 / item_count
        t1 = time.time()
        since = t1-t0
        print('--'*30, flush=True)
        print('% 3d %s -acc@1: %.4f -acc@2: %.4f -acc@3: %.4f ||time: %d' % (epoch_num,  dt(),  val_acc1, val_acc2,  val_acc3, since), flush=True)
        print('--' * 30, flush=True)
    return val_acc1, val_acc2, val_acc3



