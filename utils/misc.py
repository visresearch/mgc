
import os
import shutil
import random
import numpy as np
import torch
import torch.distributed as dist


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def clip_gradients_by_history(named_parameters, history_grad_norms, max_scale=1.1):
    for key, p in named_parameters:
        if p.grad is not None:
            grad_norm = p.grad.data.norm(2)
            max_norm = max_scale * history_grad_norms[key]

            if grad_norm > max_norm and history_grad_norms[key] > 1e-6:
                coeff = max_norm / (grad_norm + 1e-8)
                p.grad.data.mul_(coeff)
                history_grad_norms[key] = max_norm
            else:
                history_grad_norms[key] = grad_norm
       
    return history_grad_norms

def fix_random_seeds(seed=31):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        # self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n


    def synchronize_between_processes(self):  
        # pack = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        pack = torch.tensor([self.sum, self.count], device='cuda')
        dist.barrier()
        dist.all_reduce(pack)
        self.sum, self.count = pack.tolist()

    
    @property
    def avg(self):
        return self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        # return fmtstr.format(**self.__dict__)

        fmtstr = '{} {' + self.fmt + '} ({' + self.fmt + '})'
        return fmtstr.format(self.name, self.val, self.avg)


def copy_files(src_dir, dst_dir, exclude_file_list):
    fnames = os.listdir(src_dir)

    os.makedirs(dst_dir, exist_ok=True)

    for f in fnames:
        if f not in exclude_file_list:
            src = os.path.join(src_dir, f)
            if os.path.isdir(src):
                dst = os.path.join(dst_dir, f)
                print(f'copy {src} to {dst}')
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                print(f'copy {src} to {dst_dir}')
                shutil.copy(src, dst_dir)
            else:
                ValueError(f'{src} can not be copied')

    return

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

if __name__ == '__main__':
    import numpy as np
    meter = AverageMeter('test', ':.4e')

    a = np.arange(10.0)

    for i in range(10):
        meter.update(a[i], 1)
        print(meter)
    
