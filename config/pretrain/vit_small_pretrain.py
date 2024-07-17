import argparse

def vit_small_pretrain():
    args = argparse.Namespace()
    args.arch = 'vit-small'    
    args.data_root = '/path/to/dataset/'
    args.dataset = 'coco2017'
    if args.dataset == 'coco2017' or args.dataset == 'voc' or args.dataset == 'ade':
        args.nepoch = 800 
        args.batch_size = 256
        args.warmup_epoch = 10
        args.data_root = '/path/to/dataset/'
        args.drop_path_rate = 0.05
    elif args.dataset == 'imagenet':
        args.nepoch = 200 
        args.batch_size = 512
        args.warmup_epoch = 30
        args.data_root = '/path/to/dataset/'    
        args.drop_path_rate = 0.1
    args.image_size = 224
    args.patch_size = 16
    args.divide_size = 32
    args.divide_size2 = 112

    args.lr_base = 1e-3
    args.min_lr = 1e-6
    args.weight_decay = 0.05
    args.weight_decay_end = 0.05
    args.momentum_rate = 0.996

    args.max_size =10
    args.max_size2 =10
    args.max_size3 = 2

    args.projector_depth = 3
    args.predictor_depth = 2
    args.mlp_dim = 4096
    args.dim = 128
    args.temp = 0.1

    args.print_freq = 10
    args.save_freq = 10
    args.distributed = True
    args.num_workers = 48
    args.rank = 0
    args.world_size = 1
    args.init_method = 'tcp://localhost:18933'
    args.resume = False

    args.clip_grad = True

    args.exp_dir = f'./log/pretrain/{args.dataset}/ckpts_{args.dataset}_{args.arch}_loss_{args.loss_type}'\
        f'_temp{args.temp}_lr_base{args.lr_base}_batch{args.batch_size}_epoch{args.nepoch}'\
        f'_warmup{args.warmup_epoch}/'


    args.exclude_file_list= ['__pycache__', '.vscode', 'log', 'ckpt', '.git', 'out', 'dataset', 'weight']

    return args