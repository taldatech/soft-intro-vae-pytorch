import argparse
import json
import logging
import random
import re
from datetime import datetime
from os import listdir
from os.path import join

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.shapenet import ShapeNetDataset
from metrics.jsd import jsd_between_point_cloud_sets
from utils.util import cuda_setup, setup_logging
from models.vae import SoftIntroVAE, reparameterize


def _get_epochs_by_regex(path, regex):
    reg = re.compile(regex)
    return {int(w[:5]) for w in listdir(path) if reg.match(w)}


def main(eval_config):
    # Load hyperparameters as they were during training
    train_results_path = join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    with open(join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    if train_config['seed'] >= 0:
        random.seed(train_config['seed'])
        torch.manual_seed(train_config['seed'])
        torch.cuda.manual_seed(train_config['seed'])
        np.random.seed(train_config['seed'])
        torch.backends.cudnn.deterministic = True
        print("random seed: ", train_config['seed'])

    setup_logging(join(train_results_path, 'results'))
    log = logging.getLogger(__name__)

    log.debug('Evaluating Jensen-Shannon divergences on validation set on all '
              'saved epochs.')

    weights_path = join(train_results_path, 'weights')

    # Find all epochs that have saved model weights
    v_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{5})\.pth')
    epochs = sorted(v_epochs)
    log.debug(f'Testing epochs: {epochs}')

    device = cuda_setup(eval_config['cuda'], eval_config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    dataset_name = train_config['dataset'].lower()
    if dataset_name == 'shapenet':
        dataset = ShapeNetDataset(root_dir=train_config['data_dir'],
                                  classes=train_config['classes'], split='valid')
    # elif dataset_name == 'faust':
    #     from datasets.dfaust import DFaustDataset
    #     dataset = DFaustDataset(root_dir=train_config['data_dir'],
    #                             classes=train_config['classes'], split='valid')
    # elif dataset_name == 'mcgill':
    #     from datasets.mcgill import McGillDataset
    #     dataset = McGillDataset(root_dir=train_config['data_dir'],
    #                             classes=train_config['classes'], split='valid')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    classes_selected = ('all' if not train_config['classes']
                        else ','.join(train_config['classes']))
    log.debug(f'Selected {classes_selected} classes. Loaded {len(dataset)} '
              f'samples.')

    # if 'distribution' in train_config:
    #     distribution = train_config['distribution']
    # elif 'distribution' in eval_config:
    #     distribution = eval_config['distribution']
    # else:
    #     log.warning('No distribution type specified. Assumed normal = N(0, 0.2)')
    #     distribution = 'normal'

    #
    # Models

    model = SoftIntroVAE(train_config).to(device)
    model.eval()

    num_samples = len(dataset.point_clouds_names_valid)
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    # noise = torch.FloatTensor(3 * num_samples, train_config['z_size'], 1)
    noise = torch.randn(3 * num_samples, model.zdim)
    noise = noise.to(device)

    x, _ = next(iter(data_loader))
    x = x.to(device)

    results = {}

    for epoch in reversed(epochs):
        try:
            model.load_state_dict(torch.load(
                join(weights_path, f'{epoch:05}.pth')))

            start_clock = datetime.now()

            # We average JSD computation from 3 independet trials.
            js_results = []
            for _ in range(3):
                # if distribution == 'normal':
                #     noise.normal_(0, 0.2)
                # elif distribution == 'beta':
                #     noise_np = np.random.beta(train_config['z_beta_a'],
                #                               train_config['z_beta_b'],
                #                               noise.shape)
                #     noise = torch.tensor(noise_np).float().round().to(device)

                with torch.no_grad():
                    x_g = model.decode(noise)
                if x_g.shape[-2:] == (3, 2048):
                    x_g.transpose_(1, 2)

                jsd = jsd_between_point_cloud_sets(x, x_g, voxels=28)
                js_results.append(jsd)

            js_result = np.mean(js_results)
            log.debug(f'Epoch: {epoch} JSD: {js_result: .6f} '
                      f'Time: {datetime.now() - start_clock}')
            results[epoch] = js_result
        except KeyboardInterrupt:
            log.debug(f'Interrupted during epoch: {epoch}')
            break

    results = pd.DataFrame.from_dict(results, orient='index', columns=['jsd'])
    log.debug(f"Minimum JSD at epoch {results.idxmin()['jsd']}: "
              f"{results.min()['jsd']: .6f}")


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    args = parser.parse_args()

    args.config = './config/soft_intro_vae_hp.json'
    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None

    main(evaluation_config)
