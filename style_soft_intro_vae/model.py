# Copyright 2020-2021 Tal Daniel
# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This code is adapted from Stanislav Pidhorskyi's ALAE model:
https://github.com/podgorskiy/ALAE
"""

import random
from net import *
import numpy as np

"""
Helpers
"""


def set_model_require_grad(model, val):
    for param in model.parameters():
        param.requires_grad = val


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


def calc_kl(logvar, mu, mu_o=10, is_outlier=False, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param is_outlier: if True, calculates with mu_neg
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if is_outlier:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp() + 2 * mu * mu_o - mu_o.pow(2)).sum(1)
    else:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


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


class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)


class SoftIntroVAEModelTL(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="",
                 encoder="", beta_kl=1.0, beta_rec=1.0, beta_neg=1.0, scale=1 / (3 * 256 ** 2), gamma_r=1e-8):
        super(SoftIntroVAEModelTL, self).__init__()

        self.layer_count = layer_count
        self.beta_kl = beta_kl
        self.beta_rec = beta_rec
        self.beta_neg = beta_neg
        self.scale = scale
        self.gamma_r = gamma_r
        self.last_kl_diff = torch.tensor(0.0)
        self.last_rec_loss = torch.tensor(0.0)
        self.last_real_kl = torch.tensor(0.0)
        self.last_fake_kl = torch.tensor(0.0)
        self.last_expelbo_fake = torch.tensor(0.0)
        self.last_expelbo_rec = torch.tensor(0.0)

        self.mapping_tl = MAPPINGS["MappingToLatent"](
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=3)

        self.mapping_fl = MAPPINGS["MappingFromLatent"](
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = GENERATORS[generator](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.encoder = ENCODERS[encoder](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.dlatent_avg = DLatent(latent_size, self.mapping_fl.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, blend_factor, z=None, count=32, mixing=True, noise=True, return_styles=False,
                 no_truncation=False):
        """
        Sampling from the prior and decoding
        :param lod:
        :param blend_factor:
        :param z:
        :param count:
        :param mixing:
        :param noise:
        :param return_styles:
        :param no_truncation:
        :return:
        """
        if z is None:
            z = torch.randn(count, self.latent_size)
        styles = self.mapping_fl(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_fl.num_layers, 1)

        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)

        if mixing and self.style_mixing_prob is not None:
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size)
                styles2 = self.mapping_fl(z2)[:, 0]
                styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1]).repeat(1, self.mapping_fl.num_layers, 1)

                layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(layer_idx < mixing_cutoff, styles, styles2)

        if (self.truncation_psi is not None) and not no_truncation:
            layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32)
            coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
            styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder.forward(styles, lod, blend_factor, noise)
        if return_styles:
            return s, rec
        else:
            return rec

    def encode(self, x, lod, blend_factor):
        y = self.encoder(x, lod, blend_factor)
        y = self.mapping_tl(y)
        mu, logvar = y[:, 0, :], y[:, 1, :]
        z = reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x, lod, blend_factor, d_train, e_train, fake_rec_coef=1e-8):
        if e_train:
            # train encoder
            set_model_require_grad(self.encoder, True)
            set_model_require_grad(self.mapping_tl, True)
            set_model_require_grad(self.decoder, False)
            set_model_require_grad(self.mapping_fl, False)

            fake = self.generate(lod, blend_factor, count=x.shape[0], noise=True, no_truncation=True)

            z_real, mu_real, logvar_real = self.encode(x, lod, blend_factor)
            s, rec = self.generate(lod, blend_factor, z=z_real, mixing=False, noise=True, return_styles=True,
                                   no_truncation=True)

            loss_rec = calc_reconstruction_loss(x, rec, loss_type="mse", reduction="mean")

            lossE_real_kl = calc_kl(logvar_real, mu_real, reduce="mean")

            z_rec, mu_rec, logvar_rec = self.encode(rec.detach(), lod, blend_factor)
            s_rec, rec_rec = self.generate(lod, blend_factor, z=z_rec, mixing=False, noise=True, return_styles=True,
                                           no_truncation=True)

            z_fake, mu_fake, logvar_fake = self.encode(fake.detach(), lod, blend_factor)
            s_fake, rec_fake = self.generate(lod, blend_factor, z=z_fake, mixing=False, noise=True, return_styles=True,
                                             no_truncation=True)

            kl_rec = calc_kl(logvar_rec, mu_rec, reduce="none")
            kl_fake = calc_kl(logvar_fake, mu_fake, reduce="none")

            loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type="mse", reduction='none')
            while len(loss_rec_rec_e.shape) > 1:
                loss_rec_rec_e = loss_rec_rec_e.sum(-1)
            loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type="mse", reduction='none')
            while len(loss_rec_fake_e.shape) > 1:
                loss_rec_fake_e = loss_rec_fake_e.sum(-1)

            expelbo_rec = (-2 * self.scale * (self.beta_rec * loss_rec_rec_e + self.beta_neg * kl_rec)).exp().mean()
            expelbo_fake = (-2 * self.scale * (self.beta_rec * loss_rec_fake_e + self.beta_neg * kl_fake)).exp().mean()

            lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
            lossE_real = self.scale * (self.beta_rec * loss_rec + self.beta_kl * lossE_real_kl)

            loss = lossE_real + lossE_fake

            self.last_rec_loss = loss_rec.data.cpu()
            self.last_real_kl = lossE_real_kl.data.cpu()
            self.last_expelbo_fake = expelbo_fake.data.cpu()
            self.last_expelbo_rec = expelbo_rec.data.cpu()
            return loss

        elif d_train:
            # train decoder
            set_model_require_grad(self.encoder, False)
            set_model_require_grad(self.mapping_tl, False)
            set_model_require_grad(self.decoder, True)
            set_model_require_grad(self.mapping_fl, True)

            fake = self.generate(lod, blend_factor, count=x.shape[0], noise=True, no_truncation=True)
            z_real, mu_real, logvar_real = self.encode(x, lod, blend_factor)
            s, rec = self.generate(lod, blend_factor, z=z_real.detach(), mixing=False, noise=True, return_styles=True,
                                   no_truncation=True)
            loss_rec = calc_reconstruction_loss(x, rec, loss_type="mse", reduction="mean")

            z_rec, mu_rec, logvar_rec = self.encode(rec, lod, blend_factor)
            z_fake, mu_fake, logvar_fake = self.encode(fake, lod, blend_factor)

            _, rec_rec = self.generate(lod, blend_factor, z=z_rec.detach(), mixing=False, noise=True,
                                       return_styles=True, no_truncation=True)
            _, rec_fake = self.generate(lod, blend_factor, z=z_fake.detach(), mixing=False, noise=True,
                                        return_styles=True, no_truncation=True)

            loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type="mse", reduction="mean")
            loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type="mse", reduction="mean")

            lossD_rec_kl = calc_kl(logvar_rec, mu_rec, reduce="mean")
            lossD_fake_kl = calc_kl(logvar_fake, mu_fake, reduce="mean")

            self.last_rec_loss = loss_rec.data.cpu()
            self.last_kl_diff = self.last_fake_kl - self.last_real_kl
            self.last_fake_kl = lossD_fake_kl.data.cpu()

            loss = self.scale * (loss_rec * self.beta_rec + (
                    lossD_rec_kl + lossD_fake_kl) * 0.5 * self.beta_kl + self.gamma_r * 0.5 * self.beta_rec * (
                                         loss_rec_rec + loss_fake_rec))
            return loss
        else:
            # vanilla vae
            set_model_require_grad(self.encoder, True)
            set_model_require_grad(self.mapping_tl, True)
            set_model_require_grad(self.decoder, True)
            set_model_require_grad(self.mapping_fl, True)

            z_real, mu_real, logvar_real = self.encode(x, lod, blend_factor)
            s, rec = self.generate(lod, blend_factor, z=z_real, mixing=False, noise=True, return_styles=True,
                                   no_truncation=True)

            loss_rec = calc_reconstruction_loss(x, rec, loss_type="mse", reduction="mean")
            loss_kl = calc_kl(logvar_real, mu_real, reduce="mean")

            self.last_rec_loss = loss_rec.data.cpu()
            self.last_real_kl = loss_kl.data.cpu()

            loss = self.beta_rec * loss_rec + self.beta_kl * loss_kl
            return loss

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_tl.parameters()) + list(self.mapping_fl.parameters()) + list(
                self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_tl.parameters()) + list(other.mapping_fl.parameters()) + list(
                other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)
