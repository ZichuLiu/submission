# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from SNGAN import datasets
from tqdm import tqdm
from functools import partialmethod

if 'PREEMPT' in os.environ:
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
from copy import deepcopy
from SNGAN.sngan_cifar10 import Generator, Discriminator
from SNGAN.train import *
import logging
import SNGAN.config as config
from SNGAN.utils import set_log_dir, create_logger, save_checkpoint
import random
import math
from torch import Tensor
from typing import List, Optional


def adam_no_update(params: List[Tensor],
                   grads: List[Tensor],
                   exp_avgs: List[Tensor],
                   exp_avg_sqs: List[Tensor],
                   max_exp_avg_sqs: List[Tensor],
                   state_steps: List[int],
                   *,
                   amsgrad: bool,
                   beta1: float,
                   beta2: float,
                   lr: float,
                   weight_decay: float,
                   eps: float):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.data = param.data - step_size * exp_avg / denom
        # param.addcdiv_(exp_avg, denom, value=-step_size)


class AdamPPM(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        # if not 0.0 <= betas[0] < 1.0:
        #     raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        # if not 0.0 <= betas[1] < 1.0:
        #     raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(torch.optim.Adam, self).__init__(params, defaults)

    # def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    #              weight_decay=0, amsgrad=False):
    #     torch.optim.Adam.__init__(self, params, lr=lr, betas=betas, eps=eps,
    #                               weight_decay=weight_decay, amsgrad=amsgrad)

    def step_no_update(self, closure=None):
        """Performs a single optimization step without updating the exponential moving average.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'].clone())
                    exp_avg_sqs.append(state['exp_avg_sq'].clone())

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adam_no_update(params_with_grad,
                           grads,
                           exp_avgs,
                           exp_avg_sqs,
                           max_exp_avg_sqs,
                           state_steps,
                           amsgrad=group['amsgrad'],
                           beta1=beta1,
                           beta2=beta2,
                           lr=group['lr'],
                           weight_decay=group['weight_decay'],
                           eps=group['eps'])
        return loss


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)

def main(args, valid_only=False):
    G = Generator(args).cuda()
    D = Discriminator(args).cuda()
    G_optimizer = AdamPPM(G.parameters(), args.lr_g, (args.beta1, args.beta2))
    D_optimizer = AdamPPM(D.parameters(), args.lr_d, (args.beta1, args.beta2))

    # Need args = args.in_channel here

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    G.apply(weights_init)
    D.apply(weights_init)

    gen_scheduler = LinearLrDecay(G_optimizer, args.lr_g, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(D_optimizer, args.lr_d, 0.0, 0, args.max_iter * args.n_critic)

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    gen_avg_param = copy_params(G)
    start_epoch = 0
    best_fid = 1e4
    best_is = 0

    g = torch.Generator()
    g.manual_seed(args.random_seed)
    # set up data_loader
    dataset = datasets.ImageDataset(args, generator=g, worker_init_fn=seed_worker)
    train_loader = dataset.train

    lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
    inception_score = 0
    best_ind = 0

    if args.load_path:
        print(f'=> resuming from {args.load_path}')

        checkpoint_file = os.path.join('logs', args.load_path, 'Model', '_609_checkpoint_best.pth')
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            start_epoch = checkpoint['epoch']
            best_is = checkpoint['best_fid']
            G.load_state_dict(checkpoint['gen_state_dict'])
            D.load_state_dict(checkpoint['dis_state_dict'])
            G_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            D_optimizer.load_state_dict(checkpoint['dis_optimizer'])
            avg_gen_net = deepcopy(G)
            avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
            gen_avg_param = copy_params(avg_gen_net)
            del avg_gen_net

            args.path_helper = checkpoint['path_helper']
            print("path helper", args.path_helper)
            logger = create_logger(args.path_helper['log_path'])
            logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        else:
            print(f"=>Can't find {checkpoint_file}")
            args.path_helper = set_log_dir('logs', args.exp_name)
            logger = create_logger(args.path_helper['log_path'])
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    if valid_only:
        for epoch in range(610, 611):
            print(f'=> resuming from {args.load_path}')
            # checkpoint_file = os.path.join(args.load_path, 'Model', '_' + str(epoch) + '_checkpoint_best.pth')
            checkpoint_file = os.path.join(args.load_path, 'checkpoint.pth')
            # checkpoint_file = args.load_path
            assert os.path.exists(checkpoint_file)

            checkpoint = torch.load(checkpoint_file)
            G.load_state_dict(checkpoint['gen_state_dict'])
            D.load_state_dict(checkpoint['dis_state_dict'])
            G_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            D_optimizer.load_state_dict(checkpoint['dis_optimizer'])
            avg_gen_net = deepcopy(G)
            avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
            gen_avg_param = copy_params(avg_gen_net)
            del avg_gen_net

            backup_param = copy_params(G)
            load_params(G, gen_avg_param)
            # inception_score = validate(args, fixed_z, 0, G, writer_dict)
            fid = get_fid(args, 'SNGAN/fid_stats_cifar10_train.npz', epoch, G, args.num_eval_imgs, args.gen_batch_size,
                          args.eval_batch_size, writer_dict=writer_dict, cls_idx=None)
            # _, fid = validate_fid(args, fixed_z, G, writer_dict, train_loader, 0)
            # print('inception score: {0}'.format(inception_score))
            logger.info(f'Inception score: {inception_score}, FID score: {fid} || @ epoch {epoch}.')
            load_params(G, backup_param)
            # if inception_score > best_is:
            #     best_is = inception_score
            #     is_best = True
            # else:
            #     is_best = False
            # logger.info('=> inception score: {0}, best inception score: {1}'.format(inception_score, best_is))
            if fid < best_fid:
                best_fid = fid
                best_ind = epoch
                is_best = True
            else:
                is_best = False
            logger.info(
                '=> inception score: {0}, best inception score at epoch {1}: {2}'.format(fid, best_ind, best_fid))
        return

    if os.path.exists(os.path.join(args.path_helper['ckpt_path'], 'complete')):
        print("training completed")
        return

    # train loop
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        if args.alg == 'ha':
            train_ha(args, G, D, G_optimizer, D_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
                     lr_schedulers)
        elif args.alg == 'sppm':
            train_sppm(args, G, D, G_optimizer, D_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
                       lr_schedulers)
        elif args.alg == 'ppm':
            train_ppm(args, G, D, G_optimizer, D_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
                      lr_schedulers)
        else:
            train(args, G, D, G_optimizer, D_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
                  lr_schedulers)

        # if epoch % args.val_freq == 0 or epoch == int(args.max_epoch) - 1:

        if epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch) - 1:
            backup_param = copy_params(G)
            load_params(G, gen_avg_param)
            avg_gen_net = deepcopy(G)
            # inception_score = validate(args, fixed_z, 0, G, writer_dict)
            fid = get_fid(args, 'SNGAN/fid_stats_cifar10_train.npz', epoch, G, args.num_eval_imgs, args.gen_batch_size,
                          args.eval_batch_size, writer_dict=writer_dict, cls_idx=None)
            # _, fid = validate_fid(args, fixed_z, G, writer_dict, train_loader, epoch)
            # print('inception score: {0}'.format(inception_score))
            logger.info(f'Inception score: {inception_score}, FID score: {fid} || @ epoch {epoch}.')
            load_params(G, backup_param)

            if fid < best_fid:
                best_fid = fid
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.model,
                    'gen_state_dict': G.state_dict(),
                    'dis_state_dict': D.state_dict(),
                    'avg_gen_state_dict': avg_gen_net.state_dict(),
                    'gen_optimizer': G_optimizer.state_dict(),
                    'dis_optimizer': D_optimizer.state_dict(),
                    'best_fid': best_is,
                    'path_helper': args.path_helper
                }, True, args.path_helper['ckpt_path'], epoch=epoch)

            logger.info('=> FID score: {0}, best FID score: {1}'.format(fid, best_fid))

        # for preempt
        avg_gen_net = deepcopy(G)
        load_params(avg_gen_net, gen_avg_param)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'gen_state_dict': G.state_dict(),
            'dis_state_dict': D.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': G_optimizer.state_dict(),
            'dis_optimizer': D_optimizer.state_dict(),
            'best_fid': best_is,
            'path_helper': args.path_helper
        }, False, args.path_helper['ckpt_path'], epoch=epoch)

    save_checkpoint({}, False, args.path_helper['ckpt_path'], 0, filename='complete')
    print("training completed")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = config.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main(args, valid_only=args.valid_only)

