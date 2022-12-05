import os,time,datetime
import numpy as np
from math import ceil
import datetime
from tqdm import tqdm, trange
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import config
from dataset_CUB import collate_train, collate_val
from DCL_model import DCL_Network
from eval import eval
def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def train(model,
          epoch_num,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          checkpoint=1000):
    step=0
    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    checkpoint=checkpoint*train_epoch_step
    date_suffix = dt()
    con_loss_func = nn.L1Loss()
    cls_loss_func = nn.CrossEntropyLoss()
    alpha=1
    beta=1
    gamma=1
    for epoch in range(0, epoch_num-1):
        model.train(True)
        tqdm_tmp = tqdm(enumerate(data_loader['train']), total=train_epoch_step, leave = False)
        tqdm_tmp.set_description('Epoch %d' % epoch)
        for cnt, data in tqdm_tmp:
            total_loss=0
            step+=1
            model.train(True)
            
            img, cls_lables, adv_lables, loc = data
            
            cls_lables = cls_lables.to('cuda').long()
            adv_lables = adv_lables.to('cuda').long()
            loc = loc.to('cuda')
            
            optimizer.zero_grad()
            cls_out, adv_out, con_out=model(img)
            
            cls_loss = cls_loss_func(cls_out, cls_lables)
            adv_loss = cls_loss_func(adv_out, adv_lables)
            con_loss = con_loss_func(con_out, loc)
            
            total_loss = alpha * cls_loss + beta * adv_loss + gamma *con_loss
            total_loss.backward()
            torch.cuda.synchronize()

            optimizer.step()
            torch.cuda.synchronize()

            tqdm_tmp.set_postfix(loss=total_loss.detach().item(), cls_loss=cls_loss.detach().item(), adv_loss=adv_loss.detach().item(), con_loss=con_loss.detach().item())
            if step % checkpoint == 0:
                rec_loss = []
                print(32*'-', flush=True)
                print('current lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
                top1_acc, top2_acc, top3_acc  = eval(model, data_loader['val'], epoch)
                save_path = os.path.join(save_dir, 'weights_%d_%d_%.4f_%.4f.pth'%(epoch, cnt, top1_acc, top3_acc))
                torch.cuda.synchronize()
                '''
                torch.save(model.state_dict(), save_path)
                print('saved model to %s' % (save_path), flush=True)
                '''
                torch.cuda.empty_cache()
        
        exp_lr_scheduler.step(epoch)

def auto_load_resume(load_dir):
    folders = os.listdir(load_dir)
    date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
    choosed = folders[date_list.index(max(date_list))]
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    return os.path.join(load_dir, choosed, choosed_w)

if __name__ == '__main__':
    cudnn.benchmark = True
    args, train_dataset, val_dataset, model = config.Initialization()
    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(train_dataset,\
                                                batch_size=args.train_batchsize,\
                                                shuffle=True,\
                                                num_workers=args.train_num_workers,\
                                                collate_fn=collate_train,
                                                drop_last=False,
                                                pin_memory=True)

    dataloader['val'] = torch.utils.data.DataLoader(val_dataset,\
                                                batch_size=args.eval_batchsize,\
                                                shuffle=True,\
                                                num_workers=args.eval_num_workers,\
                                                collate_fn=collate_val,
                                                drop_last=False,
                                                pin_memory=True)

    if not args.auto_resume:
        print('train from imagenet pretrained models ...', flush=True)
    else:
        resume = auto_load_resume(args.save_dir)
        print('load from %s ...'%resume, flush=True)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(resume)
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    print('Set cache dir', flush=True)
    time = datetime.datetime.now()
    filename = '%d%d%d_DCB'%(time.month, time.day, time.hour)
    save_dir = os.path.join(args.save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.cuda()
    model = nn.DataParallel(model)

    ignored_params1 = list(map(id, model.module.cls_linear.parameters()))
    ignored_params2 = list(map(id, model.module.adv_linear.parameters()))
    ignored_params3 = list(map(id, model.module.con_conv.parameters()))

    ignored_params = ignored_params1 + ignored_params2 + ignored_params3

    base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())

    lr_ratio = args.cls_lr_ratio
    base_lr = args.base_lr

    optimizer = optim.SGD([{'params': base_params},
                               {'params': model.module.cls_linear.parameters(), 'lr': lr_ratio*base_lr},
                               {'params': model.module.adv_linear.parameters(), 'lr': lr_ratio*base_lr},
                               {'params': model.module.con_conv.parameters(), 'lr': lr_ratio*base_lr},
                              ], lr = base_lr, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)

    train(model,
          epoch_num=args.epoch_num,
          optimizer=optimizer,
          exp_lr_scheduler=exp_lr_scheduler,
          data_loader=dataloader,
          save_dir=args.save_dir,
          checkpoint=args.eval_epoch)