"""
Test a trained model (on the test split of the data)
"""

import json
import numpy as np

import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader

from utils.util import cuda_setup
from metrics.jsd import jsd_between_point_cloud_sets

from models.vae import SoftIntroVAE


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
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)
    return data_loader


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

    # We average JSD computation from 3 independent trials.
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


if __name__ == "__main__":
    path_to_weights = './results/vae/soft_intro_vae_chair/weights/00350_jsd_0.106.pth'
    config_path = 'config/soft_intro_vae_hp.json'
    config = None
    if config_path is not None and config_path.endswith('.json'):
        with open(config_path) as f:
            config = json.load(f)
    assert config is not None
    device = cuda_setup(config['cuda'], config['gpu'])
    print("using device: ", device)
    model = prepare_model(config, path_to_weights, device=device)
    test_jsd = calc_jsd_valid(model, config, prior_std=config["prior_std"], split='test')
    print(f'test jsd: {test_jsd}')
