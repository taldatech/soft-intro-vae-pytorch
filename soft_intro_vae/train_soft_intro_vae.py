"""
Train Soft-Intro VAE for image datasets
Author: Tal Daniel
"""

# imports
# torch and friends
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torchvision import transforms

# standard
import os
import random
import time
import numpy as np
from tqdm import tqdm
import pickle
from dataset import ImageDatasetFromFile, DigitalMonstersDataset
from metrics.fid_score import calculate_fid_given_dataset
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

"""
Models
"""


class ResidualBlock(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    "if inc is not outc" -> "if inc != outc"
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc != outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        return output


class Encoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 cond_dim=10):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        self.cond_dim = cond_dim
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        print("num fc features: ", num_fc_features)
        if self.conditional:
            self.fc = nn.Linear(num_fc_features + self.cond_dim, 2 * zdim)
        else:
            self.fc = nn.Linear(num_fc_features, 2 * zdim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x, o_cond=None):
        y = self.main(x).view(x.size(0), -1)
        if self.conditional and o_cond is not None:
            y = torch.cat([y, o_cond], dim=1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 conv_input_size=None, cond_dim=10):
        super(Decoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        cc = channels[-1]
        self.conv_input_size = conv_input_size
        if conv_input_size is None:
            num_fc_features = cc * 4 * 4
        else:
            num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]
        self.cond_dim = cond_dim
        if self.conditional:
            self.fc = nn.Sequential(
                nn.Linear(zdim + self.cond_dim, num_fc_features),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(zdim, num_fc_features),
                nn.ReLU(True),
            )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z, y_cond=None):
        z = z.view(z.size(0), -1)
        if self.conditional and y_cond is not None:
            y_cond = y_cond.view(y_cond.size(0), -1)
            z = torch.cat([z, y_cond], dim=1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


class SoftIntroVAE(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 cond_dim=10):
        super(SoftIntroVAE, self).__init__()

        self.zdim = zdim
        self.conditional = conditional
        self.cond_dim = cond_dim

        self.encoder = Encoder(cdim, zdim, channels, image_size, conditional=conditional, cond_dim=cond_dim)

        self.decoder = Decoder(cdim, zdim, channels, image_size, conditional=conditional,
                               conv_input_size=self.encoder.conv_output_size, cond_dim=cond_dim)

    def forward(self, x, o_cond=None, deterministic=False):
        if self.conditional and o_cond is not None:
            mu, logvar = self.encode(x, o_cond=o_cond)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z, y_cond=o_cond)
        else:
            mu, logvar = self.encode(x)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z)
        return mu, logvar, z, y

    def sample(self, z, y_cond=None):
        y = self.decode(z, y_cond=y_cond)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu"), y_cond=None):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decode(z, y_cond=y_cond)

    def encode(self, x, o_cond=None):
        if self.conditional and o_cond is not None:
            mu, logvar = self.encoder(x, o_cond=o_cond)
        else:
            mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z, y_cond=None):
        if self.conditional and y_cond is not None:
            y = self.decoder(z, y_cond=y_cond)
        else:
            y = self.decoder(z)
        return y


"""
Helpers
"""


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


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def str_to_list(x):
    return [int(xi) for xi in x.split(',')]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)


def record_image(writer, image_list, cur_iter, num_rows=8):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=num_rows), cur_iter)


def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)


def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "./saves/" + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))


"""
Train Functions
"""


def train_soft_intro_vae(dataset='cifar10', z_dim=128, lr_e=2e-4, lr_d=2e-4, batch_size=128, num_workers=4,
                         start_epoch=0, exit_on_negative_diff=False,
                         num_epochs=250, num_vae=0, save_interval=50, recon_loss_type="mse",
                         beta_kl=1.0, beta_rec=1.0, beta_neg=1.0, test_iter=1000, seed=-1, pretrained=None,
                         device=torch.device("cpu"), num_row=8, gamma_r=1e-8, with_fid=False):
    """
    :param dataset: dataset to train on: ['cifar10', 'mnist', 'fmnist', 'svhn', 'monsters128', 'celeb128', 'celeb256', 'celeb1024']
    :param z_dim: latent dimensions
    :param lr_e: learning rate for encoder
    :param lr_d: learning rate for decoder
    :param batch_size: batch size
    :param num_workers: num workers for the loading the data
    :param start_epoch: epoch to start from
    :param exit_on_negative_diff: stop run if mean kl diff between fake and real is negative after 50 epochs
    :param num_epochs: total number of epochs to run
    :param num_vae: number of epochs for vanilla vae training
    :param save_interval: epochs between checkpoint saving
    :param recon_loss_type: type of reconstruction loss ('mse', 'l1', 'bce')
    :param beta_kl: beta coefficient for the kl divergence
    :param beta_rec: beta coefficient for the reconstruction loss
    :param beta_neg: beta coefficient for the kl divergence in the expELBO function
    :param test_iter: iterations between sample image saving
    :param seed: seed
    :param pretrained: path to pretrained model, to continue training
    :param device: device to run calculation on - torch.device('cuda:x') or torch.device('cpu')
    :param num_row: number of images in a row gor the sample image saving
    :param gamma_r: coefficient for the reconstruction loss for fake data in the decoder
    :param with_fid: calculate FID during training (True/False)
    :return:
    """
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    # --------------build models -------------------------
    if dataset == 'cifar10':
        image_size = 32
        channels = [64, 128, 256]
        train_set = CIFAR10(root='./cifar10_ds', train=True, download=True, transform=transforms.ToTensor())
        ch = 3
    elif dataset == 'celeb128':
        channels = [64, 128, 256, 512, 512]
        image_size = 128
        ch = 3
        output_height = 128
        train_size = 162770
        data_root = '../data/celeb256/img_align_celeba'
        image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        train_list = image_list[:train_size]
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    elif dataset == 'celeb256':
        channels = [64, 128, 256, 512, 512, 512]
        image_size = 256
        ch = 3
        output_height = 256
        train_size = 162770
        data_root = '../data/celeb256/img_align_celeba'
        image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        train_list = image_list[:train_size]
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    elif dataset == 'celeb1024':
        channels = [16, 32, 64, 128, 256, 512, 512, 512]
        image_size = 1024
        ch = 3
        output_height = 1024
        train_size = 29000
        data_root = './' + dataset
        image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        train_list = image_list[:train_size]
        assert len(train_list) > 0

        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    elif dataset == 'monsters128':
        channels = [64, 128, 256, 512, 512]
        image_size = 128
        ch = 3
        data_root = './monsters_ds/'
        train_set = DigitalMonstersDataset(root_path=data_root, output_height=image_size)
    elif dataset == 'svhn':
        image_size = 32
        channels = [64, 128, 256]
        train_set = SVHN(root='./svhn', split='train', transform=transforms.ToTensor(), download=True)
        ch = 3
    elif dataset == 'fmnist':
        image_size = 28
        channels = [64, 128]
        train_set = FashionMNIST(root='./fmnist_ds', train=True, download=True, transform=transforms.ToTensor())
        ch = 1
    elif dataset == 'mnist':
        image_size = 28
        channels = [64, 128]
        train_set = MNIST(root='./mnist_ds', train=True, download=True, transform=transforms.ToTensor())
        ch = 1
    else:
        raise NotImplementedError("dataset is not supported")

    model = SoftIntroVAE(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size).to(device)
    if pretrained is not None:
        load_model(model, pretrained, device)
    print(model)

    fig_dir = './figures_' + dataset
    os.makedirs(fig_dir, exist_ok=True)

    optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
    optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)

    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)

    scale = 1 / (ch * image_size ** 2)  # normalize by images size (channels * height * width)

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers)

    start_time = time.time()

    cur_iter = 0
    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []
    exp_elbos_f = []
    exp_elbos_r = []
    best_fid = None
    for epoch in range(start_epoch, num_epochs):
        if with_fid and ((epoch == 0) or (epoch >= 100 and epoch % 20 == 0) or epoch == num_epochs - 1):
            with torch.no_grad():
                print("calculating fid...")
                fid = calculate_fid_given_dataset(train_data_loader, model, batch_size, cuda=True, dims=2048,
                                                  device=device, num_images=50000)
                print("fid:", fid)
                if best_fid is None:
                    best_fid = fid
                elif best_fid > fid:
                    print("best fid updated: {} -> {}".format(best_fid, fid))
                    best_fid = fid
                    # save
                    save_epoch = epoch
                    prefix = dataset + "_soft_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                        beta_rec) + "_" + "fid_" + str(fid) + "_"
                    save_checkpoint(model, save_epoch, cur_iter, prefix)

        diff_kls = []
        # save models
        if epoch % save_interval == 0 and epoch > 0:
            save_epoch = (epoch // save_interval) * save_interval
            prefix = dataset + "_soft_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                beta_rec) + "_"
            save_checkpoint(model, save_epoch, cur_iter, prefix)

        model.train()

        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []
        batch_exp_elbo_f = []
        batch_exp_elbo_r = []

        pbar = tqdm(iterable=train_data_loader)

        for batch in pbar:
            # --------------train------------
            if dataset in ["cifar10", "svhn", "fmnist", "mnist"]:
                batch = batch[0]
            if epoch < num_vae:
                if len(batch.size()) == 3:
                    batch = batch.unsqueeze(0)

                batch_size = batch.size(0)

                real_batch = batch.to(device)

                # =========== Update E, D ================

                real_mu, real_logvar, z, rec = model(real_batch)

                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
                loss_kl = calc_kl(real_logvar, real_mu, reduce="mean")

                loss = beta_rec * loss_rec + beta_kl * loss_kl

                optimizer_d.zero_grad()
                optimizer_e.zero_grad()
                loss.backward()
                optimizer_e.step()
                optimizer_d.step()

                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=loss_kl.data.cpu().item())

                if cur_iter % test_iter == 0:
                    vutils.save_image(torch.cat([real_batch, rec], dim=0).data.cpu(),
                                      '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)

            else:
                if len(batch.size()) == 3:
                    batch = batch.unsqueeze(0)

                b_size = batch.size(0)
                noise_batch = torch.randn(size=(b_size, z_dim)).to(device)

                real_batch = batch.to(device)

                # =========== Update E ================
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.decoder.parameters():
                    param.requires_grad = False

                fake = model.sample(noise_batch)

                real_mu, real_logvar = model.encode(real_batch)
                z = reparameterize(real_mu, real_logvar)
                rec = model.decoder(z)

                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")

                lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

                rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
                fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

                kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
                kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")

                loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction='none')
                while len(loss_rec_rec_e.shape) > 1:
                    loss_rec_rec_e = loss_rec_rec_e.sum(-1)
                loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction='none')
                while len(loss_rec_fake_e.shape) > 1:
                    loss_rec_fake_e = loss_rec_fake_e.sum(-1)

                expelbo_rec = (-2 * scale * (beta_rec * loss_rec_rec_e + beta_neg * kl_rec)).exp().mean()
                expelbo_fake = (-2 * scale * (beta_rec * loss_rec_fake_e + beta_neg * kl_fake)).exp().mean()

                lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
                lossE_real = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl)

                lossE = lossE_real + lossE_fake
                optimizer_e.zero_grad()
                lossE.backward()
                optimizer_e.step()

                # ========= Update D ==================
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = True

                fake = model.sample(noise_batch)
                rec = model.decoder(z.detach())
                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")

                rec_mu, rec_logvar = model.encode(rec)
                z_rec = reparameterize(rec_mu, rec_logvar)

                fake_mu, fake_logvar = model.encode(fake)
                z_fake = reparameterize(fake_mu, fake_logvar)

                rec_rec = model.decode(z_rec.detach())
                rec_fake = model.decode(z_fake.detach())

                loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type,
                                                        reduction="mean")
                loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type,
                                                         reduction="mean")

                lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
                lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")

                lossD = scale * (loss_rec * beta_rec + (
                        lossD_rec_kl + lossD_fake_kl) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (
                                         loss_rec_rec + loss_fake_rec))

                optimizer_d.zero_grad()
                lossD.backward()
                optimizer_d.step()
                if torch.isnan(lossD) or torch.isnan(lossE):
                    raise SystemError

                dif_kl = -lossE_real_kl.data.cpu() + lossD_fake_kl.data.cpu()
                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=lossE_real_kl.data.cpu().item(),
                                 diff_kl=dif_kl.item(), expelbo_f=expelbo_fake.cpu().item())

                diff_kls.append(-lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item())
                batch_kls_real.append(lossE_real_kl.data.cpu().item())
                batch_kls_fake.append(lossD_fake_kl.cpu().item())
                batch_kls_rec.append(lossD_rec_kl.data.cpu().item())
                batch_rec_errs.append(loss_rec.data.cpu().item())
                batch_exp_elbo_f.append(expelbo_fake.data.cpu())
                batch_exp_elbo_r.append(expelbo_rec.data.cpu())

                if cur_iter % test_iter == 0:
                    _, _, _, rec_det = model(real_batch, deterministic=True)
                    max_imgs = min(batch.size(0), 16)
                    vutils.save_image(
                        torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
                        '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)

            cur_iter += 1
        e_scheduler.step()
        d_scheduler.step()
        pbar.close()
        if exit_on_negative_diff and epoch > 50 and np.mean(diff_kls) < -1.0:
            print(
                f'the kl difference [{np.mean(diff_kls):.3f}] between fake and real is negative (no sampling improvement)')
            print("try to lower beta_neg hyperparameter")
            print("exiting...")
            raise SystemError("Negative KL Difference")

        if epoch > num_vae - 1:
            kls_real.append(np.mean(batch_kls_real))
            kls_fake.append(np.mean(batch_kls_fake))
            kls_rec.append(np.mean(batch_kls_rec))
            rec_errs.append(np.mean(batch_rec_errs))
            exp_elbos_f.append(np.mean(batch_exp_elbo_f))
            exp_elbos_r.append(np.mean(batch_exp_elbo_r))
            # epoch summary
            print('#' * 50)
            print(f'Epoch {epoch} Summary:')
            print(f'beta_rec: {beta_rec}, beta_kl: {beta_kl}, beta_neg: {beta_neg}')
            print(
                f'rec: {rec_errs[-1]:.3f}, kl: {kls_real[-1]:.3f}, kl_fake: {kls_fake[-1]:.3f}, kl_rec: {kls_rec[-1]:.3f}')
            print(
                f'diff_kl: {np.mean(diff_kls):.3f}, exp_elbo_f: {exp_elbos_f[-1]:.4e}, exp_elbo_r: {exp_elbos_r[-1]:.4e}')
            print(f'time: {time.time() - start_time}')
            print('#' * 50)
        if epoch == num_epochs - 1:
            with torch.no_grad():
                _, _, _, rec_det = model(real_batch, deterministic=True)
                noise_batch = torch.randn(size=(b_size, z_dim)).to(device)
                fake = model.sample(noise_batch)
                max_imgs = min(batch.size(0), 16)
                vutils.save_image(
                    torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
                    '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)

            # plot graphs
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.arange(len(kls_real)), kls_real, label="kl_real")
            ax.plot(np.arange(len(kls_fake)), kls_fake, label="kl_fake")
            ax.plot(np.arange(len(kls_rec)), kls_rec, label="kl_rec")
            ax.plot(np.arange(len(rec_errs)), rec_errs, label="rec_err")
            ax.legend()
            plt.savefig('./soft_intro_train_graphs.jpg')
            with open('./soft_intro_train_graphs_data.pickle', 'wb') as fp:
                graph_dict = {"kl_real": kls_real, "kl_fake": kls_fake, "kl_rec": kls_rec, "rec_err": rec_errs}
                pickle.dump(graph_dict, fp)
            # save models
            prefix = dataset + "_soft_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                beta_rec) + "_"
            save_checkpoint(model, epoch, cur_iter, prefix)
            model.train()


if __name__ == '__main__':
    """
    Recommended hyper-parameters:
    - CIFAR10: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
    - SVHN: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
    - MNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
    - FashionMNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
    - Monsters: beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, z_dim: 128, batch_size: 16
    - CelebA-HQ: beta_kl: 1.0, beta_rec: 0.5, beta_neg: 1024, z_dim: 256, batch_size: 8
    """
    beta_kl = 1.0
    beta_rec = 1.0
    beta_neg = 256
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print("betas: ", beta_kl, beta_neg, beta_rec)
    try:
        train_soft_intro_vae(dataset="monsters128", z_dim=128, batch_size=16, num_workers=0, num_epochs=400,
                             num_vae=0, beta_kl=beta_kl, beta_neg=beta_neg, beta_rec=beta_rec,
                             device=device, save_interval=50, start_epoch=0, lr_e=2e-4, lr_d=2e-4,
                             pretrained=None,
                             test_iter=1000, with_fid=False)
    except SystemError:
        print("Error, probably loss is NaN, try again...")
