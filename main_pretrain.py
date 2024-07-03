import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler


from torchvision.datasets import ImageFolder
from collections import OrderedDict

from module.model import ViTEncoderProjPredHeadMultiNoClsD3Momentum
from module.loss import contrastive_loss_patch

from module.augmentation import TwoCropsTransformBox

from utils.misc import AverageMeter, copy_files, cosine_scheduler, clip_gradients_by_history, clip_gradients
from utils.logger import Logger, console_logger

from config.pretrain.vit_small_pretrain import vit_small_pretrain



def train_epoch(model, optimizer, lr_scheduler, wd_scheduler, momentum_schedule, train_loader, epoch, \
                loggers, args, scaler, history_grad_norms): 
    model.train()
    
    logger_tb, logger_console = loggers

    data_time = AverageMeter('Data', ':6.3f')
    model_time = AverageMeter('Data', ':6.3f')
    loss_time = AverageMeter('Data', ':6.3f')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_patch = AverageMeter('Loss_patch', ':.4e')
    losses_divide = AverageMeter('Loss_divide', ':.4e')
    losses_l = AverageMeter('Loss_l', ':.4e')
    sims = AverageMeter('Sim', ':.4e')
    num_iter = len(train_loader)
    niter_global = epoch * num_iter
    
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        it = num_iter * epoch + i  # global training iteration
        m = momentum_schedule[it]
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler[it]
            if j == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_scheduler[it]

        batch_size = images[0].size(0)
        image1, image2, patch_indexs1, patch_indexs2, ious, patch_indexs3, patch_indexs4, ious2, patch_indexs5, patch_indexs6, ious3 = images
        image1 = image1.cuda(args.rank, non_blocking=True)
        image2 = image2.cuda(args.rank, non_blocking=True)
        patch_indexs1 = patch_indexs1.cuda(args.rank, non_blocking=True)
        patch_indexs2 = patch_indexs2.cuda(args.rank, non_blocking=True)
        ious = ious.cuda(args.rank, non_blocking=True)
        patch_indexs3 = patch_indexs3.cuda(args.rank, non_blocking=True)
        patch_indexs4 = patch_indexs4.cuda(args.rank, non_blocking=True)
        ious2 = ious2.cuda(args.rank, non_blocking=True)
        patch_indexs5 = patch_indexs5.cuda(args.rank, non_blocking=True)
        patch_indexs6 = patch_indexs6.cuda(args.rank, non_blocking=True)
        ious3 = ious3.cuda(args.rank, non_blocking=True)
        data_time.update(time.time() - end)
        if args.distributed:
            offset1 = batch_size * args.max_size *  torch.distributed.get_rank() 
            offset2 = batch_size * args.max_size2 *  torch.distributed.get_rank() 
            offset3 = batch_size * args.max_size3 *  torch.distributed.get_rank()
        else :
            offset1=0 
            offset2=0 
            offset3=0 
        labels1 = (torch.arange(batch_size*args.max_size, dtype=torch.long) + offset1).cuda() # distribute
        labels2 = (torch.arange(batch_size*args.max_size2, dtype=torch.long) + offset2).cuda() # distribute
        labels3 = (torch.arange(batch_size*args.max_size3, dtype=torch.long) + offset3).cuda() # distribute
        with autocast():
            p1, p2, z1, z2, dp1, dp2, dz1, dz2, lp1, lp2, lz1, lz2= model(image1, image2, patch_indexs1, patch_indexs2, patch_indexs3, patch_indexs4, \
                                patch_indexs5, patch_indexs6, (args.divide_size // args.patch_size), (args.divide_size2 // args.patch_size), m)

            model_time.update(time.time() - end)
 
            loss1, _ = contrastive_loss_patch(p1, z2.detach(), \
                args.temp, labels1, offset1, args.distributed, ious, batch_size, args.max_size)
            loss2, _ = contrastive_loss_patch(p2, z1.detach(), \
                args.temp, labels1, offset1, args.distributed, ious, batch_size, args.max_size) 
            loss5, _ = contrastive_loss_patch(dp1, dz2.detach(), \
                args.temp, labels2, offset2, args.distributed, ious2, batch_size, args.max_size2)
            loss6, _ = contrastive_loss_patch(dp2, dz1.detach(), \
                args.temp, labels2, offset2, args.distributed, ious2, batch_size, args.max_size2)

            loss3, _ = contrastive_loss_patch(lp1, lz2.detach(), \
                args.temp, labels3, offset3, args.distributed, ious3, batch_size, args.max_size3)
            loss4, _ = contrastive_loss_patch(lp2, lz1.detach(), \
                args.temp, labels3, offset3, args.distributed, ious3, batch_size, args.max_size3)

            loss = ( args.max_size * (loss1 + loss2) + args.max_size2 * (loss5 + loss6) + args.max_size3 * (loss3 + loss4)) / (args.max_size + args.max_size2 + args.max_size3)
            loss_time.update(time.time() - end)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.clip_grad:
            scaler.unscale_(optimizer)
            # history_grad_norms = clip_gradients_by_history(
            #     model.named_parameters(), 
            #     # model.module.base_encoder.patch_embed.named_parameters(), 
            #     history_grad_norms,
            #     1.2
            #     # args.max_grad_scale_low,
            #     # args.max_grad_scale_high
            # )
            clip_gradients(model, 3.0)
        scaler.step(optimizer)
        scaler.update()
        #-------------------------------------------------------------------------------------------#
        losses.update(loss.item())
        losses_patch.update((loss1+loss2).item())
        losses_divide.update((loss5+loss6).item())
        losses_l.update((loss3+loss4).item())
        batch_time.update(time.time() - end)
        
        end = time.time()

        niter_global += 1
        
        if args.rank == 0:
            logger_tb.add_scalar('Iter/loss', losses.val, niter_global)
        
        if (i + 1) % args.print_freq == 0 and logger_console is not None \
            and args.rank == 0:  
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            logger_console.info(f'Epoch [{epoch}][{i+1}/{num_iter}] - '
                        f'data_time: {data_time.avg:.3f},     '
                        f'model_time: {model_time.avg:.3f},     '
                        f'loss_time: {loss_time.avg:.3f},     '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'wd: {wd:.5f},     '
                        f'loss: {losses.val:.3f}({losses.avg:.3f})')


    if args.distributed:
        losses.synchronize_between_processes()
        losses_patch.synchronize_between_processes()
        losses_divide.synchronize_between_processes()
        losses_l.synchronize_between_processes()
    return losses.avg, losses_patch.avg, losses_divide.avg, losses_l.avg


def main_worker(gpu, ngpus_per_node, args):

    rank = args.rank * ngpus_per_node + gpu

    if args.distributed:
        dist.init_process_group(backend='nccl', init_method=args.init_method, rank=rank, world_size=args.world_size)
        torch.distributed.barrier()
    args.rank = rank

    #------------------------------logger-----------------------------#
    if args.rank == 0:
        args.exp_dir = f'./log/pretrain/{args.dataset}/ckpts_{args.arch}_loss_{args.loss_type}_epoch{args.nepoch}'\
            f'_temp{args.temp}_temp_cls{args.temp_cls}_lr_base{args.lr_base}_batch{args.batch_size}_img{args.img_size}_patch{args.patch_size}'\
            f'_divide_size{args.divide_size}_warmup{args.warmup_epoch}_cls_multi_no_cls_remove/'
        os.makedirs(args.exp_dir, exist_ok=True)
        log_root = args.exp_dir
        name = f'moco_{args.use_moco}_momentum{args.momentum}_r{args.r}_dim{args.dim}_mlp_dim{args.mlp_dim}_max_size{args.max_size}_max_size2_{args.max_size2}'\
            f'_m{args.m}m2_{args.m2}_projector_depth{args.projector_depth}_predictor_depth{args.predictor_depth}'    
        logger_tb = Logger(log_root, name)
        logger_console = console_logger(logger_tb.log_dir, 'console')
        dst_dir = os.path.join(logger_tb.log_dir, 'code/')
        copy_files('./', dst_dir, args.exclude_file_list)
    else:
        logger_tb,logger_console = None,None
 

    #---------------------------------model------------------------------#
    if args.arch == 'vit-small':
        model = ViTEncoderProjPredHeadMultiNoClsD3Momentum(img_size=args.img_size, patch_size=args.patch_size, embed_dim=384, depth=12, \
                num_heads=8, dim=args.dim, mlp_dim=args.mlp_dim, projector_depth=args.projector_depth, predictor_depth=args.predictor_depth, drop_path_rate=args.drop_path_rate)
    elif args.arch == 'vit-base':
        model = ViTEncoderProjPredHeadMultiNoClsD3Momentum(img_size=args.img_size, patch_size=args.patch_size, embed_dim=768, depth=12, \
                num_heads=12, dim=args.dim, mlp_dim=args.mlp_dim, projector_depth=args.projector_depth, predictor_depth=args.predictor_depth)
    model = model.cuda(args.rank)

    args.lr = args.lr_base * args.batch_size / 256

    if args.distributed :
        torch.cuda.set_device(args.rank)
        args.batch_size = int(args.batch_size / args.world_size)
        args.num_workers = int((args.num_workers + args.world_size - 1) / args.world_size) 
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.rank])

    train_set = ImageFolder(root=args.data_root, transform=TwoCropsTransformBox(args.patch_size, args.divide_size, args.divide_size2, \
                                                args.img_size, args.max_size, args.max_size2, args.max_size3))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=3,
                              persistent_workers = True) 

    #----------------------------optim---------------------------#
    parameters = model.module.parameters() \
        if isinstance(model, DDP) else model.parameters()

    optimizer = torch.optim.AdamW(parameters, 
        lr=args.lr, 
        betas=(0.9, 0.999), 
        weight_decay=args.weight_decay)

    lr_scheduler = cosine_scheduler(
        args.lr ,  # linear scaling rule
        args.min_lr,
        args.nepoch, len(train_loader),
        warmup_epochs=args.warmup_epoch,
    )

    wd_scheduler = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.nepoch, len(train_loader),
    )
    momentum_schedule = cosine_scheduler(args.momentum, 1,
                                               args.nepoch, len(train_loader))
    scaler = GradScaler()

    start_epoch=0

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            

    if args.rank==0 :
        path_save = os.path.join(args.exp_dir, logger_tb.log_name)

    history_grad_norms = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            history_grad_norms[name] = 0.0  

    for epoch in range(start_epoch, args.nepoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss, losses_patch, losses_divide, losses_l = train_epoch(model, optimizer, lr_scheduler, wd_scheduler, momentum_schedule, \
                                                       train_loader, epoch, (logger_tb, logger_console), args, scaler, history_grad_norms)
        lr = optimizer.param_groups[0]['lr']
        if args.rank == 0:
            logger_tb.add_scalar('Epoch/lr', lr, epoch + 1)
            logger_tb.add_scalar('Epoch/loss', loss, epoch + 1)
            logger_tb.add_scalar('Epoch/losses_patch', losses_patch, epoch + 1)
            logger_tb.add_scalar('Epoch/losses_divide', losses_divide, epoch + 1)
            logger_tb.add_scalar('Epoch/losses_l', losses_l, epoch + 1)

        if (epoch + 1) % args.save_freq == 0 and args.rank == 0:
            _epoch = epoch + 1
            state_dict = model.module.state_dict() \
                if isinstance(model, DDP) else model.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, f'{path_save}/checkpoint{_epoch:0>4d}.pth')
    
    if args.rank == 0: 
        state_dict = model.module.state_dict() \
                if isinstance(model, DDP) else model.state_dict()

        torch.save(state_dict, f'{path_save}/last.pth')


def main(args):
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.world_size * ngpus_per_node

    if args.distributed:
        mp.spawn(main_worker,args=(ngpus_per_node, args), nprocs=args.world_size)
    else:
        main_worker(args.rank, ngpus_per_node, args)


if __name__ == '__main__':  
    args = vit_small_pretrain() 
    main(args)