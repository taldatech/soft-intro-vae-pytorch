"""
Train 3D Soft-Intro VAE for image datasets
Author: Tal Daniel
Adapted from: https://github.com/MaciejZamorski/3d-AAE
"""
# imports
import argparse
import json
import logging
import random
from datetime import datetime
from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging
from metrics.jsd import jsd_between_point_cloud_sets
from datasets.transforms3d import RotateAxisAngle

from models.vae import SoftIntroVAE, reparameterize
from tqdm import tqdm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # DO NOT REMOVE
matplotlib.use("Agg")
cudnn.benchmark = True


def calc_jsd_valid(model, config, prior_std=1.0, split='valid'):
    model.eval()
    device = cuda_setup(config['cuda'], config['gpu'])
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'], split=split)
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    classes_selected = ('all' if not config['classes']
                        else ','.join(config['classes']))
    num_samples = len(dataset.point_clouds_names_valid)
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)
    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication

    x, _ = next(iter(data_loader))
    x = x.to(device)

    # We average JSD computation from 3 independet trials.
    js_results = []
    for _ in range(3):
        noise = prior_std * torch.randn(3 * num_samples, model.zdim)
        noise = noise.to(device)

        with torch.no_grad():
            x_g = model.decode(noise)
        if x_g.shape[-2:] == (3, 2048):
            x_g.transpose_(1, 2)

        jsd = jsd_between_point_cloud_sets(x, x_g, voxels=28)
        js_results.append(jsd)
    js_result = np.mean(js_results)
    return js_result


def remove_ticks_from_ax(ax):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def main(config):
    if config['seed'] >= 0:
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        print("random seed: ", config['seed'])

    results_dir = prepare_results_dir(config)
    starting_epoch = find_latest_epoch(results_dir) + 1

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    setup_logging(results_dir)
    log = logging.getLogger(__name__)
    logging.getLogger('matplotlib.font_manager').disabled = True

    device = cuda_setup(config['cuda'], config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')

    # load dataset
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'])
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')

    log.debug("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']),
        len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                   shuffle=config['shuffle'],
                                   num_workers=config['num_workers'],
                                   drop_last=True, pin_memory=True)
    scale = 1 / (3 * config['n_points'])
    # hyper-parameters
    valid_frequency = config["valid_frequency"]
    num_vae = config["num_vae"]
    beta_rec = config["beta_rec"]
    beta_kl = config["beta_kl"]
    beta_neg = config["beta_neg"]
    gamma_r = config["gamma_r"]
    apply_random_rotation = "rotate" in config["transforms"]
    if apply_random_rotation:
        print("applying random rotation to input shapes")

    # model
    model = SoftIntroVAE(config).to(device)

    if config['reconstruction_loss'].lower() == 'chamfer':
        from losses.chamfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    prior_std = config["prior_std"]
    prior_logvar = np.log(prior_std ** 2)
    print(f'prior: N(0, {prior_std ** 2:.3f})')

    # optimizers
    optimizer_e = getattr(optim, config['optimizer']['E']['type'])
    optimizer_e = optimizer_e(model.encoder.parameters(), **config['optimizer']['E']['hyperparams'])
    optimizer_d = getattr(optim, config['optimizer']['D']['type'])
    optimizer_d = optimizer_d(model.decoder.parameters(), **config['optimizer']['D']['hyperparams'])

    scheduler_e = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=[350, 450, 550], gamma=0.5)
    scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[350, 450, 550], gamma=0.5)

    if starting_epoch > 1:
        model.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}.pth')))

        optimizer_e.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_optim_e.pth')))
        optimizer_d.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_optim_d.pth')))

    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []
    exp_elbos_f = []
    exp_elbos_r = []
    diff_kls = []
    best_res = {"epoch": 0, "jsd": None}

    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()

        model.train()

        if epoch < num_vae:
            total_loss = 0.0
            pbar = tqdm(iterable=points_dataloader)
            for i, point_data in enumerate(pbar, 1):
                x, _ = point_data
                x = x.to(device)

                # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
                if x.size(-1) == 3:
                    x.transpose_(x.dim() - 2, x.dim() - 1)

                x_rec, mu, logvar = model(x)
                loss_rec = reconstruction_loss(x.permute(0, 2, 1) + 0.5, x_rec.permute(0, 2, 1) + 0.5)
                while len(loss_rec.shape) > 1:
                    loss_rec = loss_rec.sum(-1)
                loss_rec = loss_rec.mean()
                loss_kl = calc_kl(logvar, mu, logvar_o=prior_logvar, reduce="mean")
                loss = beta_rec * loss_rec + beta_kl * loss_kl

                optimizer_e.zero_grad()
                optimizer_d.zero_grad()
                loss.backward()
                total_loss += loss.item()
                optimizer_e.step()
                optimizer_d.step()

                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=loss_kl.data.cpu().item())

        else:
            batch_kls_real = []
            batch_kls_fake = []
            batch_kls_rec = []
            batch_rec_errs = []
            batch_exp_elbo_f = []
            batch_exp_elbo_r = []
            batch_diff_kls = []
            pbar = tqdm(iterable=points_dataloader)
            for i, point_data in enumerate(pbar, 1):
                x, _ = point_data
                x = x.to(device)

                # random rotation
                if apply_random_rotation:
                    angle = torch.rand(size=(x.shape[0],)) * 180
                    rotate_transform = RotateAxisAngle(angle, axis="Z", device=device)
                    x = rotate_transform.transform_points(x)

                # change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
                if x.size(-1) == 3:
                    x.transpose_(x.dim() - 2, x.dim() - 1)

                noise_batch = prior_std * torch.randn(size=(config['batch_size'], model.zdim)).to(device)

                # ----- update E ----- #
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.decoder.parameters():
                    param.requires_grad = False

                fake = model.sample(noise_batch)

                real_mu, real_logvar = model.encode(x)
                z = reparameterize(real_mu, real_logvar)
                x_rec = model.decoder(z)

                loss_rec = reconstruction_loss(x.permute(0, 2, 1) + 0.5, x_rec.permute(0, 2, 1) + 0.5)
                while len(loss_rec.shape) > 1:
                    loss_rec = loss_rec.sum(-1)
                loss_rec = loss_rec.mean()

                loss_real_kl = calc_kl(real_logvar, real_mu, logvar_o=prior_logvar, reduce="mean")

                rec_rec, rec_mu, rec_logvar = model(x_rec.detach())
                rec_fake, fake_mu, fake_logvar = model(fake.detach())

                kl_rec = calc_kl(rec_logvar, rec_mu, logvar_o=prior_logvar, reduce="none")
                kl_fake = calc_kl(fake_logvar, fake_mu, logvar_o=prior_logvar, reduce="none")

                loss_rec_rec_e = reconstruction_loss(x_rec.detach().permute(0, 2, 1) + 0.5,
                                                     rec_rec.permute(0, 2, 1) + 0.5)
                while len(loss_rec_rec_e.shape) > 1:
                    loss_rec_rec_e = loss_rec_rec_e.sum(-1)
                loss_rec_fake_e = reconstruction_loss(fake.permute(0, 2, 1) + 0.5, rec_fake.permute(0, 2, 1) + 0.5)
                while len(loss_rec_fake_e.shape) > 1:
                    loss_rec_fake_e = loss_rec_fake_e.sum(-1)

                expelbo_rec = (-2 * scale * (beta_rec * loss_rec_rec_e + beta_neg * kl_rec)).exp().mean()
                expelbo_fake = (-2 * scale * (beta_rec * loss_rec_fake_e + beta_neg * kl_fake)).exp().mean()

                loss_margin = scale * beta_kl * loss_real_kl + 0.25 * (expelbo_rec + expelbo_fake)

                lossE = scale * beta_rec * loss_rec + loss_margin
                optimizer_e.zero_grad()
                lossE.backward()
                optimizer_e.step()

                # ----- update D ----- #
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = True

                fake = model.sample(noise_batch)
                with torch.no_grad():
                    z = reparameterize(real_mu.detach(), real_logvar.detach())
                rec = model.decoder(z.detach())
                loss_rec = reconstruction_loss(x.permute(0, 2, 1) + 0.5, rec.permute(0, 2, 1) + 0.5)
                while len(loss_rec.shape) > 1:
                    loss_rec = loss_rec.sum(-1)
                loss_rec = loss_rec.mean()

                rec_mu, rec_logvar = model.encode(rec)
                z_rec = reparameterize(rec_mu, rec_logvar)

                fake_mu, fake_logvar = model.encode(fake)
                z_fake = reparameterize(fake_mu, fake_logvar)

                rec_rec = model.decode(z_rec.detach())
                rec_fake = model.decode(z_fake.detach())

                loss_rec_rec = reconstruction_loss(rec.detach().permute(0, 2, 1) + 0.5, rec_rec.permute(0, 2, 1) + 0.5)
                while len(loss_rec_rec.shape) > 1:
                    loss_rec_rec = loss_rec.sum(-1)
                loss_rec_rec = loss_rec_rec.mean()
                loss_fake_rec = reconstruction_loss(fake.detach().permute(0, 2, 1) + 0.5,
                                                    rec_fake.permute(0, 2, 1) + 0.5)
                while len(loss_fake_rec.shape) > 1:
                    loss_fake_rec = loss_rec.sum(-1)
                loss_fake_rec = loss_fake_rec.mean()

                lossD_rec_kl = calc_kl(rec_logvar, rec_mu, logvar_o=prior_logvar, reduce="mean")
                lossD_fake_kl = calc_kl(fake_logvar, fake_mu, logvar_o=prior_logvar, reduce="mean")

                lossD = scale * (loss_rec * beta_rec + (
                        lossD_rec_kl + lossD_fake_kl) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (
                                         loss_rec_rec + loss_fake_rec))

                optimizer_d.zero_grad()
                lossD.backward()
                optimizer_d.step()

                if torch.isnan(lossD):
                    raise SystemError("loss is Nan")

                diff_kl = -loss_real_kl.data.cpu() + lossD_fake_kl.data.cpu()
                batch_diff_kls.append(diff_kl)
                batch_kls_real.append(loss_real_kl.data.cpu().item())
                batch_kls_fake.append(lossD_fake_kl.cpu().item())
                batch_kls_rec.append(lossD_rec_kl.data.cpu().item())
                batch_rec_errs.append(loss_rec.data.cpu().item())
                batch_exp_elbo_f.append(expelbo_fake.data.cpu())
                batch_exp_elbo_r.append(expelbo_rec.data.cpu())

                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=loss_real_kl.data.cpu().item(),
                                 diff_kl=diff_kl.item(), expelbo_f=expelbo_fake.cpu().item())

        pbar.close()
        scheduler_e.step()
        scheduler_d.step()
        if epoch > num_vae - 1:
            kls_real.append(np.mean(batch_kls_real))
            kls_fake.append(np.mean(batch_kls_fake))
            kls_rec.append(np.mean(batch_kls_rec))
            rec_errs.append(np.mean(batch_rec_errs))
            exp_elbos_f.append(np.mean(batch_exp_elbo_f))
            exp_elbos_r.append(np.mean(batch_exp_elbo_r))
            diff_kls.append(np.mean(batch_diff_kls))
            # epoch summary
            print('#' * 50)
            print(f'Epoch {epoch} Summary:')
            print(f'beta_rec: {beta_rec}, beta_kl: {beta_kl}, beta_neg: {beta_neg}')
            print(
                f'rec: {rec_errs[-1]:.3f}, kl: {kls_real[-1]:.3f}, kl_fake: {kls_fake[-1]:.3f}, kl_rec: {kls_rec[-1]:.3f}')
            print(
                f'diff_kl: {diff_kls[-1]:.3f}, exp_elbo_f: {exp_elbos_f[-1]:.4e}, exp_elbo_r: {exp_elbos_r[-1]:.4e}')
            if best_res['jsd'] is not None:
                print(f'best jsd: {best_res["jsd"]}, epoch: {best_res["epoch"]}')
            print(f'time: {datetime.now() - start_epoch_time}')
            print('#' * 50)
        # save intermediate results
        model.eval()
        with torch.no_grad():
            noise_batch = prior_std * torch.randn(size=(5, model.zdim)).to(device)
            fake = model.sample(noise_batch).data.cpu().numpy()
            x_rec, _, _ = model(x, deterministic=True)
            x_rec = x_rec.data.cpu().numpy()

        fig = plt.figure(dpi=350)
        for k in range(5):
            ax = fig.add_subplot(3, 5, k + 1, projection='3d')
            ax = plot_3d_point_cloud(x[k][0].data.cpu().numpy(), x[k][1].data.cpu().numpy(),
                                     x[k][2].data.cpu().numpy(),
                                     in_u_sphere=True, show=False, axis=ax, show_axis=True, s=4, color='dodgerblue')
            remove_ticks_from_ax(ax)

        for k in range(5):
            ax = fig.add_subplot(3, 5, k + 6, projection='3d')
            ax = plot_3d_point_cloud(x_rec[k][0],
                                     x_rec[k][1],
                                     x_rec[k][2],
                                     in_u_sphere=True, show=False, axis=ax, show_axis=True, s=4, color='dodgerblue')
            remove_ticks_from_ax(ax)

        for k in range(5):
            ax = fig.add_subplot(3, 5, k + 11, projection='3d')
            ax = plot_3d_point_cloud(fake[k][0], fake[k][1], fake[k][2],
                                     in_u_sphere=True, show=False, axis=ax, show_axis=True, s=4, color='dodgerblue')
            remove_ticks_from_ax(ax)

        fig.savefig(join(results_dir, 'samples', f'figure_{epoch}'))
        plt.close(fig)

        if epoch % valid_frequency == 0:
            print("calculating valid jsd...")
            model.eval()
            with torch.no_grad():
                jsd = calc_jsd_valid(model, config, prior_std=prior_std)
            print(f'epoch: {epoch}, jsd: {jsd:.4f}')
            if best_res['jsd'] is None:
                best_res['jsd'] = jsd
                best_res['epoch'] = epoch
            elif best_res['jsd'] > jsd:
                print(f'epoch: {epoch}: best jsd updated: {best_res["jsd"]} -> {jsd}')
                best_res['jsd'] = jsd
                best_res['epoch'] = epoch
                # save
                torch.save(model.state_dict(), join(weights_path, f'{epoch:05}_jsd_{jsd:.4f}.pth'))

        if epoch % config['save_frequency'] == 0:
            torch.save(model.state_dict(), join(weights_path, f'{epoch:05}.pth'))
            torch.save(optimizer_e.state_dict(),
                       join(weights_path, f'{epoch:05}_optim_e.pth'))
            torch.save(optimizer_d.state_dict(),
                       join(weights_path, f'{epoch:05}_optim_d.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    args.config = './config/soft_intro_vae_hp.json'
    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)
