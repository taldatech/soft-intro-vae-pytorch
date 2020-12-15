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

import torch.utils.data
from torchvision.utils import save_image
from net import *
from model import SoftIntroVAEModelTL
from launcher import run
from checkpointer import Checkpointer
from defaults import get_cfg_defaults
import lreq
from dataloader import *

lreq.use_implicit_lreq.set(True)


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


def place(canvas, image, x, y):
    im_size = image.shape[2]
    if len(image.shape) == 4:
        image = image[0]
    canvas[:, y: y + im_size, x: x + im_size] = image * 0.5 + 0.5


def save_sample(model, sample, i):
    os.makedirs('results', exist_ok=True)

    with torch.no_grad():
        model.eval()
        x_rec = model.generate(model.generator.layer_count - 1, 1, z=sample)

        def save_pic(x_rec):
            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample,
                       'sample_%i_lr.png' % i, nrow=16)

        save_pic(x_rec)


def sample(cfg, logger):
    torch.cuda.set_device(0)
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
    model.cuda(0)
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

    extra_checkpoint_data = checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        z, mu, _ = model.encode(x, layer_count - 1, 1)
        styles = model.mapping_fl(mu)
        return styles

    def decode(x):
        return model.decoder(x, layer_count - 1, 1, noise=True)

    rnd = np.random.RandomState(5)

    dataset = TFRecordsDataset(cfg, logger, rank=0, world_size=1, buffer_size_mb=10, channels=cfg.MODEL.CHANNELS,
                               train=False)

    dataset.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, 10)
    b = iter(make_dataloader(cfg, logger, dataset, 10, 0, numpy=True))

    def make(sample):
        canvas = []
        with torch.no_grad():
            for img in sample:
                x = torch.tensor(np.asarray(img, dtype=np.float32), device='cpu',
                                 requires_grad=True).cuda() / 127.5 - 1.
                if x.shape[0] == 4:
                    x = x[:3]
                latents = encode(x[None, ...].cuda())
                f = decode(latents)
                r = torch.cat([x[None, ...].detach().cpu(), f.detach().cpu()], dim=3)
                canvas.append(r)
        return canvas

    sample = next(b)
    canvas = make(sample)
    canvas = torch.cat(canvas, dim=0)

    save_image(canvas * 0.5 + 0.5, './make_figures/reconstructions_ffhq_real_1.png', nrow=2, pad_value=1.0)

    sample = next(b)
    canvas = make(sample)
    canvas = torch.cat(canvas, dim=0)

    save_image(canvas * 0.5 + 0.5, './make_figures/reconstructions_ffhq_real_2.png', nrow=2, pad_value=1.0)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='SoftIntroVAE-reconstruction-ffhq',
        default_config='./configs/ffhq256.yaml',
        world_size=gpu_count, write_log=False)
