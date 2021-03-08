import argparse
import json
import logging
import random
from importlib import import_module
from os.path import join

import numpy as np
import torch
from torch.distributions import Beta
from torch.utils.data import DataLoader

from datasets.shapenet.shapenet import ShapeNetDataset
from models.vae import SoftIntroVAE, reparameterize
from utils.util import cuda_setup


def prepare_model(config, path_to_weights, device=torch.device("cpu")):
    model = SoftIntroVAE(config).to(device)
    model.load_state_dict(torch.load(path_to_weights, map_location=device))
    model.eval()
    return model


def main(eval_config):
    # Load hyperparameters as they were during training
    train_results_path = join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    with open(join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    device = cuda_setup(config['cuda'], config['gpu'])
    print("using device: ", device)

    #
    # Dataset
    #
    dataset_name = train_config['dataset'].lower()
    if dataset_name == 'shapenet':
        dataset = ShapeNetDataset(root_dir=train_config['data_dir'],
                                  classes=train_config['classes'], split='test')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    classes_selected = ('all' if not train_config['classes']
                        else ','.join(train_config['classes']))

    #
    # Models
    #

    model = prepare_model(config, path_to_weights, device=device)
    model.eval()

    num_samples = len(dataset.point_clouds_names_test)
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication

    x, _ = next(iter(data_loader))
    x = x.to(device)

    np.save(join(train_results_path, 'results', f'_X'), x)

    prior_std = config["prior_std"]

    for i in range(3):
        noise = prior_std * torch.randn(3 * num_samples, model.zdim)
        noise = noise.to(device)

        with torch.no_grad():
            x_g = model.decode(noise)
        if x_g.shape[-2:] == (3, 2048):
            x_g.transpose_(1, 2)
        np.save(join(train_results_path, 'results', f'_Xg_{i}'), x_g)

    with torch.no_grad():
        mu_z, logvar_z = model.encode(x)
        data_z = reparameterize(mu_z, logvar_z)
        # x_rec = model.decode(data_z)  # stochastic
        x_rec = model.decode(mu_z)  # deterministic
    if x_rec.shape[-2:] == (3, 2048):
        x_rec.transpose_(1, 2)

    np.save(join(train_results_path, 'results', f'_Xrec'), x_rec)


if __name__ == '__main__':
    path_to_weights = './results/vae/soft_intro_vae_chair/weights/00350_jsd_0.106.pth'
    config_path = 'config/soft_intro_vae_hp.json'
    config = None
    if config_path is not None and config_path.endswith('.json'):
        with open(config_path) as f:
            config = json.load(f)
    assert config is not None

    main(config)
