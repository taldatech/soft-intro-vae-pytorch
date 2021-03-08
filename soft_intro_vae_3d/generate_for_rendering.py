"""
Generate point clouds from a trained model for rendering using Mitsuba
"""

import json
import numpy as np
import os

import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader

from utils.util import cuda_setup
from models.vae import SoftIntroVAE, reparameterize


def generate_from_model(model, num_samples, prior_std=0.2, device=torch.device("cpu")):
    model.eval()
    noise = prior_std * torch.randn(size=(num_samples, model.zdim)).to(device)
    with torch.no_grad():
        x_g = model.decode(noise)
    if x_g.shape[-2:] == (3, 2048):
        x_g.transpose_(1, 2)
    return x_g


def interpolate(model, data, num_steps=20, device=torch.device("cpu")):
    assert data.shape[0] >= 2, "must supply at least 2 data points"
    model.eval()
    steps = np.linspace(0, 1, num_steps)
    data = data[:2].to(device)
    # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
    if data.size(-1) == 3:
        data.transpose_(data.dim() - 2, data.dim() - 1)
    mu_z, logvar_z = model.encode(data)
    data_z = reparameterize(mu_z, logvar_z)
    interpolations = [data_z[0][None,]]
    for step in steps:
        interpolation = step * data_z[1] + (1 - step) * data_z[0]
        interpolations.append(interpolation[None,])
    interpolations.append(data_z[1][None,])
    interpolations = torch.cat(interpolations, dim=0)
    data_interpolation = model.decode(interpolations)
    return data_interpolation


def save_point_cloud_np(save_path, data_tensor):
    # Change dim
    if data_tensor.size(-1) != 3:
        data_tensor.transpose_(data_tensor.dim() - 1, data_tensor.dim() - 2)
    data_np = data_tensor.data.cpu().numpy()
    np.save(save_path, data_np)
    print(f'saved data @ {save_path}')


def prepare_model(config, path_to_weights, device=torch.device("cpu")):
    model = SoftIntroVAE(config).to(device)
    model.load_state_dict(torch.load(path_to_weights, map_location=device))
    model.eval()
    return model


def prepare_data(config, split='train', batch_size=32):
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'], split=split)
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4,
                             drop_last=False, pin_memory=True)
    return data_loader


def prepare_dataset(config, split='train', batch_size=32):
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'], split=split)
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    return dataset


if __name__ == "__main__":
    """
    cars + airplane: cars:[60, 5150], planes: [6450, 6550, 6950]
    """
    save_path = './results/generated_data'
    os.makedirs(save_path, exist_ok=True)
    path_generated = os.path.join(save_path, 'generated.npy')
    path_interpolated = os.path.join(save_path, 'interpolations.npy')
    path_to_weights = './results/vae/soft_intro_vae_chair/weights/01618_jsd_0.0175.pth'
    config_path = 'config/soft_intro_vae_hp.json'
    config = None
    if config_path is not None and config_path.endswith('.json'):
        with open(config_path) as f:
            config = json.load(f)
    assert config is not None
    device = cuda_setup(config['cuda'], config['gpu'])
    print("using device: ", device)
    model = prepare_model(config, path_to_weights, device=device)
    dataset = prepare_dataset(config, split='train', batch_size=config['batch_size'])
    batch = torch.stack([torch.from_numpy(dataset[60][0]), torch.from_numpy(dataset[6450][0])], dim=0)
    # generate
    x_g = generate_from_model(model, num_samples=5, device=device)
    save_point_cloud_np(path_generated, x_g)
    print(f'save generations in {path_generated}')
    # interpolate
    x_interpolated = interpolate(model, batch, num_steps=50, device=device)
    save_point_cloud_np(path_interpolated, x_interpolated)
    print(f'save interpolations in {path_interpolated}')
    print("use these .npy files to render beautiful point clouds with Mitsuba, see the 'render' directory for instructions")
