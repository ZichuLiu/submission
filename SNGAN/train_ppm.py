import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
from imageio import imsave
import logging
from itertools import chain
from SNGAN.inception_score import get_inception_score
from SNGAN.fid_score_pytorch import calculate_fid
from SNGAN.torch_fid_score import calculate_fid_given_paths_torch
from pathlib import Path
logger = logging.getLogger(__name__)




def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1


def train_ppm(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
              epoch, writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0
    criterion = nn.BCEWithLogitsLoss()
    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()


    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # dis_optimizer.zero_grad()
        # gen_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z)

        ones = torch.autograd.Variable(torch.ones(real_imgs.size()[0])).cuda()
        zeros = torch.autograd.Variable(torch.zeros(fake_imgs.size()[0])).cuda()
        ones_g = torch.autograd.Variable(torch.ones(fake_imgs.size()[0])).cuda()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        g_loss = -torch.mean(fake_validity)
        # d_loss = criterion(real_validity.squeeze(), ones) + criterion(fake_validity.squeeze(), zeros)
        # g_loss = criterion(fake_validity.squeeze(), ones_g)

        grad_g_0 = torch.autograd.grad(g_loss, gen_net.parameters(), create_graph=True, retain_graph=True)
        grad_d_0 = torch.autograd.grad(d_loss, dis_net.parameters(), create_graph=True, retain_graph=True)

        g_init = [p.data for i, p in enumerate(gen_net.parameters())]
        d_init = [p.data for i, p in enumerate(dis_net.parameters())]

        for i, p in enumerate(gen_net.parameters()):
            p.grad = grad_g_0[i]
        for i, p in enumerate(dis_net.parameters()):
            p.grad = grad_d_0[i]
        gen_optimizer.step_no_update()
        dis_optimizer.step_no_update()

        # gi = [p.data for i, p in enumerate(gen_net.parameters())]
        # di = [p.data for i, p in enumerate(dis_net.parameters())]
        # for i, p in enumerate(gen_net.parameters()):
        #     p.data = g_init[i].data
        # for i, p in enumerate(dis_net.parameters()):
        #     p.data = d_init[i].data

        for iter in range(args.extra_steps - 1):
            real_validity = dis_net(real_imgs)
            fake_imgs = gen_net(z)
            fake_validity = dis_net(fake_imgs)
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            g_loss = -torch.mean(fake_validity)
            grad_g = torch.autograd.grad(g_loss, gen_net.parameters(), create_graph=True, retain_graph=True)
            grad_d = torch.autograd.grad(d_loss, dis_net.parameters(), create_graph=True, retain_graph=True)
            for i, p in enumerate(gen_net.parameters()):
                p.grad = grad_g[i]
            for i, p in enumerate(dis_net.parameters()):
                p.grad = grad_d[i]
            gen_optimizer.step_no_update()
            dis_optimizer.step_no_update()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z)
        fake_validity = dis_net(fake_imgs)
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        g_loss = -torch.mean(fake_validity)
        grad_g = torch.autograd.grad(g_loss, gen_net.parameters(), create_graph=True, retain_graph=True)
        grad_d = torch.autograd.grad(d_loss, dis_net.parameters(), create_graph=True, retain_graph=True)
        for i, p in enumerate(gen_net.parameters()):
            p.grad = grad_g[i]
        for i, p in enumerate(dis_net.parameters()):
            p.grad = grad_d[i]

        gen_optimizer.step()
        dis_optimizer.step()
        # adjust learning rate
        if schedulers:
            gen_scheduler, dis_scheduler = schedulers
            g_lr = gen_scheduler.step(global_steps)
            d_lr = dis_scheduler.step(global_steps)
            if global_steps % 20 == 0:
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

        # moving average weight
        for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            avg_p.mul_(args.ema).add_((1-args.ema), p.data)

        if global_steps % 20 == 0:
            writer.add_scalar('g_loss', g_loss.item(), global_steps)
        gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            if 'PREEMPT' in os.environ:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                      (
                      epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))
            else:
                tqdm.write(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                    (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1


def train_ha(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
             epoch,
             writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0
    alpha = args.ha_alpha
    beta = args.ha_beta
    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    rg = [torch.zeros_like(p.data, requires_grad=True) for p in gen_net.parameters()]
    rd = [torch.zeros_like(p.data, requires_grad=True) for p in dis_net.parameters()]
    rg_optimizer = torch.optim.Adam(rg, args.lr_r, (args.beta1, args.beta2))
    rd_optimizer = torch.optim.Adam(rd, args.lr_r, (args.beta1, args.beta2))
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        grad_d = torch.autograd.grad(d_loss, inputs=dis_net.parameters(), create_graph=True, retain_graph=True)
        for j, p in enumerate(dis_net.parameters()):
            p.grad = grad_d[j].data + beta * (p.data - rd[j])
        for j, p in enumerate(dis_net.parameters()):
            rd[j].grad = -alpha * (p.data - rd[j].data)
        # d_loss.backward()
        rd_optimizer.step()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            # gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
            grad_g = torch.autograd.grad(g_loss, inputs=gen_net.parameters(), create_graph=True, retain_graph=True)
            for j, p in enumerate(gen_net.parameters()):
                p.grad = grad_g[j].data + beta * (p.data - rg[j])
            for j, p in enumerate(gen_net.parameters()):
                rg[j].grad = -alpha * (p.data - rg[j].data)
            # g_loss.backward()
            rg_optimizer.step()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    if not os.path.exists(fid_buffer_dir):
        os.makedirs(fid_buffer_dir)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        # gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        gen_imgs = gen_net(z).data.cpu().numpy()
        # for img_idx, img in enumerate(gen_imgs):
        #     file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
        #     imsave(file_name, img)
        img_list.append(gen_imgs)

    # get inception score
    new_sample_list = list(chain.from_iterable(img_list))
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                                    resize=True, splits=10)

    # get fid score
    # logger.info('=> calculate fid score')
    # fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    # os.system('rm -r {}'.format(fid_buffer_dir))

    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    # writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean


def validate_fid(args, fixed_z, gen_net: nn.Module, writer_dict, train_loader, epoch):
    gen_net = gen_net.eval()
    global_steps = writer_dict['valid_global_steps']
    gen_net = gen_net.eval()
    eval_iter = args.num_eval_imgs // args.eval_batch_size

    # skip IS
    inception_score = 0

    # compute FID
    sample_list = []
    for i in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
        samples = gen_net(z)
        sample_list.append(samples.data.cpu().numpy())

    new_sample_list = list(chain.from_iterable(sample_list))
    fake_image_np = np.concatenate([img[None] for img in new_sample_list], 0)

    real_image_np = []
    for i, (images, _) in enumerate(train_loader):
        real_image_np += [images.data.numpy()]
        batch_size = real_image_np[0].shape[0]
        if len(real_image_np) * batch_size >= fake_image_np.shape[0]:
            break
    real_image_np = np.concatenate(real_image_np, 0)[:fake_image_np.shape[0]]
    fid_score = calculate_fid(real_image_np, fake_image_np, batch_size=300)
    var_fid = fid_score[0][2]
    fid = round(fid_score[0][1], 3)
    print('------------------------fid_score--------------------------')
    print(fid_score)

    # Generate a batch of images
    sample_dir = os.path.join(args.path_helper['sample_path'], 'sample_dir')
    Path(sample_dir).mkdir(exist_ok=True)

    sample_imgs = gen_net(fixed_z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0)
    img_grid = make_grid(sample_imgs, nrow=5).to('cpu', torch.uint8).numpy()
    file_name = os.path.join(sample_dir, f'epoch_{epoch}_fid_{fid}.png')
    imsave(file_name, img_grid.swapaxes(0, 1).swapaxes(1, 2))

    writer_dict['valid_global_steps'] = global_steps + 1
    return inception_score, fid

def get_fid(args, fid_stat, epoch, gen_net, num_img, gen_batch_size, val_batch_size, writer_dict=None, cls_idx=None):
    gen_net.eval()
    with torch.no_grad():
        # eval mode
        gen_net.eval()

#         eval_iter = num_img // gen_batch_size
#         img_list = []
#         for _ in tqdm(range(eval_iter), desc='sample images'):
#             z = torch.cuda.FloatTensor(np.random.normal(0, 1, (gen_batch_size, args.latent_dim)))

#             # Generate a batch of images
#             if args.n_classes > 0:
#                 if cls_idx is not None:
#                     label = torch.ones(z.shape[0]) * cls_idx
#                     label = label.type(torch.cuda.LongTensor)
#                 else:
#                     label = torch.randint(low=0, high=args.n_classes, size=(z.shape[0],), device='cuda')
#                 gen_imgs = gen_net(z, epoch)
#             else:
#                 gen_imgs = gen_net(z, epoch)
#             if isinstance(gen_imgs, tuple):
#                 gen_imgs = gen_imgs[0]
#             img_list += [gen_imgs]

#         img_list = torch.cat(img_list, 0)
        fid_score = calculate_fid_given_paths_torch(args, gen_net, fid_stat, gen_batch_size=gen_batch_size, batch_size=val_batch_size)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('FID_score', fid_score, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return fid_score