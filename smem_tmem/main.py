import parser
args = parser.parse_args()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import math
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

import sys
sys.path.append('..')
from util import utils
from model import loss
from model import network
from util.cmc import Video_Cmc

def validation(net, dataloader, args): # original
    
    net.eval()
    pbar = tqdm(total=len(dataloader), ncols=100, leave=True)
    pbar.set_description('Inference')

    gallery_features = []
    gallery_labels = []
    gallery_cams = []
    with torch.no_grad():
        for c, data in enumerate(dataloader):
            seqs = data[0].cuda()
            seqs = seqs.reshape((seqs.shape[0]//args.seq_len, args.seq_len, ) + seqs.shape[1:])
            label = data[1]
            cams = data[2]
            
            out = net(seqs)
            feat = out['val_bn']
            
            gallery_features.append(feat.cpu())
            gallery_labels.append(label)
            gallery_cams.append(cams)
            pbar.update(1)
    pbar.close()

    gallery_features = torch.cat(gallery_features, dim=0).numpy()
    gallery_labels = torch.cat(gallery_labels, dim=0).numpy()
    gallery_cams = torch.cat(gallery_cams, dim=0).numpy()
    
    Cmc, mAP = Video_Cmc(gallery_features, gallery_labels, gallery_cams, 
                         dataloader.dataset.query_idx, 10000)
    net.train()

    return Cmc[0], mAP

if __name__ == '__main__':
        
    # set transformation (H flip is inside dataset)
    train_transform = Compose(
        [Resize((256, 128)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         , utils.RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])])
    test_transform = Compose(
        [Resize((256, 128)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    print('\nDataloading starts !!')
    
    train_dataloader = utils.Get_Video_train_DataLoader(
        args.train_txt, args.train_info, train_transform, shuffle=True,num_workers=args.num_workers,
        seq_len=args.seq_len, track_per_class=args.track_per_class, class_per_batch=args.class_per_batch)
    
    test_dataloader = utils.Get_Video_test_DataLoader(
        args.test_txt, args.test_info, args.query_info, test_transform, batch_size=args.test_batch,
        shuffle=False, num_workers=args.num_workers, seq_len=args.seq_len, distractor=True)
    
    print('Dataloading ends !!\n')

    num_class = train_dataloader.dataset.n_id
    net = nn.DataParallel(
        network.STMN(args.feat_dim, num_class=num_class, stride=args.stride).cuda())

    if args.load_ckpt is not None:
        state = torch.load(args.load_ckpt)
        net.load_state_dict(state)
        
    # log 
    os.system('mkdir -p %s'%(args.ckpt))
    
    # optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, weight_decay = 1e-4)
    else:
        optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay = 1e-5)
    if args.lr_step_size != 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer,  args.lr_step_size,  0.1)
        
    if args.eval_only:
        cmc, map = validation(net, test_dataloader, args)
        print('R1 %.1f  mAP %.1f'%(cmc*100, map*100))

    else:
        best_cmc = 0
        loss = loss.Loss()
        for epoch in range(1, args.n_epochs+1):

            ############################### Validation ###############################
            if (epoch+1) % args.eval_freq == 0:
                cmc, map = validation(net, test_dataloader, args)
                print('R1 %.1f  mAP %.1f'%(cmc*100, map*100))

                f = open(os.path.join(args.ckpt, args.log_path), 'a')
                f.write('[Epoch %03d] R1 %.1f  mAP %.1f\n'%(epoch, cmc*100, map*100))
                torch.save(net.state_dict(), os.path.join(args.ckpt, 'ckpt_best.pth'))
                f.close()

            ############################### Training ###############################
            pbar = tqdm(total=len(train_dataloader), ncols=100, leave=True)
            pbar.set_description('Epoch %03d' %epoch)

            for batch_idx, data in enumerate(train_dataloader):

                seqs, labels = data # seqs: [B, T, C, H, W]
                num_batches = seqs.size()[0]
                seqs = seqs.cuda()
                labels = labels.cuda()

                # Forward
                out = net(seqs)
                loss_out = loss(out, labels)

                total_loss = 0
                total_loss += loss_out['track_id']
                total_loss += loss_out['trip']
                total_loss += out['smem']['loss']['mem_trip'].mean()
                total_loss += out['tmem']['loss']['mem_trip'].mean()

                # Backward
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                pbar.update(1)

            pbar.close()
            if args.lr_step_size !=0:
                scheduler.step()