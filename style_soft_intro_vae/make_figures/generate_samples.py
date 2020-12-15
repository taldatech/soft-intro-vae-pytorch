# Copyright 2020-2021 Tal Daniel
# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from net import *
from model import SoftIntroVAEModelTL
from launcher import run
from dataloader import *
from checkpointer import Checkpointer
# from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from PIL import Image
import PIL
from pathlib import Path
from tqdm import tqdm


def millify(n):
    millnames = ['', 'k', 'M', 'G', 'T', 'P']
    n = float(n)
    millidx = max(0, min(len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.1f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])


def count_parameters(model, print_func=print, verbose=False):
    for n, p in model.named_parameters():
        if p.requires_grad and verbose:
            print_func(n, millify(p.numel()))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_samples(cfg, model, save_dir, num_samples, device=torch.device("cpu")):
    for i in tqdm(range(num_samples)):
        samplez = torch.randn(size=(1, cfg.MODEL.LATENT_SPACE_SIZE)).float().to(device)
        image = model.generate(cfg.DATASET.MAX_RESOLUTION_LEVEL - 2, 1, samplez, 1, mixing=True)[0]
        im = image.data.cpu().numpy()
        im = im.transpose(1, 2, 0)
        im = im * 0.5 + 0.5
        image = PIL.Image.fromarray(np.clip(im * 255, 0, 255).astype(np.uint8), 'RGB')
        image.save(os.path.join(save_dir, 'image_{}.jpg'.format(i)))


def sample(cfg, logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(2)
    model = SoftIntroVAEModelTL(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        beta_kl=cfg.MODEL.BETA_KL,
        beta_rec=cfg.MODEL.BETA_REC,
        beta_neg=cfg.MODEL.BETA_NEG[cfg.MODEL.LAYER_COUNT - 1],
        scale=cfg.MODEL.SCALE
    )

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl

    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters decoder:")
    print(count_parameters(decoder))

    logger.info("Trainable parameters encoder:")
    print(count_parameters(encoder))

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    checkpointer.load()

    model.eval()

    path = './make_figures/output'
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, cfg.NAME), exist_ok=True)
    with torch.no_grad():
        generate_samples(cfg, model, path, 5, device=device)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='SoftIntroVAEVAE-generations',
        default_config='./configs/ffhq256.yaml',
        world_size=gpu_count, write_log=False)
