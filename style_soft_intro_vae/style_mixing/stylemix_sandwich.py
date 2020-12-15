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
from model import SandwichModelTL
from launcher import run
from checkpointer import Checkpointer
# from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq

from PIL import Image

lreq.use_implicit_lreq.set(True)

src_len = 5
dst_len = 6


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
    image = image.cpu().detach().numpy()
    im_size = image.shape[1]
    canvas[:, y * im_size: (y + 1) * im_size, x * im_size: (x + 1) * im_size] = image * 0.5 + 0.5


def main(cfg, logger):
    with torch.no_grad():
        _main(cfg, logger)


def _main(cfg, logger):
    torch.cuda.set_device(0)
    model = SandwichModelTL(
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

    logger.info("Trainable parameters generator:")
    print(count_parameters(decoder))

    logger.info("Trainable parameters discriminator:")
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
    last_epoch = list(extra_checkpoint_data['auxiliary']['scheduler'].values())[0]['last_epoch']
    logger.info("Model trained for %d epochs" % last_epoch)

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        layer_count = cfg.MODEL.LAYER_COUNT

        zlist = []
        for i in range(x.shape[0]):
            z, mu, _ = model.encode(x[i][None, ...], layer_count - 1, 1)
            styles = model.mapping_fl(z)
            # Z, _ = model.encode(x[i][None, ...], layer_count - 1, 1)
            zlist.append(styles)
        z = torch.cat(zlist, dim=0)
        # Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return z

    def decode(x):
        decoded = []
        for i in range(x.shape[0]):
            r = model.decoder(x[i][None, ...], layer_count - 1, 1, noise=True)
            decoded.append(r)
        return torch.cat(decoded)

    path = cfg.DATASET.STYLE_MIX_PATH
    im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)

    src_originals = []
    for i in range(src_len):
        try:
            im = np.asarray(Image.open(os.path.join(path, 'src/%d.png' % i)))
        except FileNotFoundError:
            im = np.asarray(Image.open(os.path.join(path, 'src/%d.jpg' % i)))
        im = im.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]
        factor = x.shape[2] // im_size
        if factor != 1:
            x = torch.nn.functional.avg_pool2d(x[None, ...], factor, factor)[0]
        assert x.shape[2] == im_size
        src_originals.append(x)
    src_originals = torch.stack([x for x in src_originals])
    dst_originals = []
    for i in range(dst_len):
        try:
            im = np.asarray(Image.open(os.path.join(path, 'dst/%d.png' % i)))
        except FileNotFoundError:
            im = np.asarray(Image.open(os.path.join(path, 'dst/%d.jpg' % i)))
        im = im.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]
        factor = x.shape[2] // im_size
        if factor != 1:
            x = torch.nn.functional.avg_pool2d(x[None, ...], factor, factor)[0]
        assert x.shape[2] == im_size
        dst_originals.append(x)
    dst_originals = torch.stack([x for x in dst_originals])

    src_latents = encode(src_originals)
    src_images = decode(src_latents)

    dst_latents = encode(dst_originals)
    dst_images = decode(dst_latents)

    canvas = np.zeros([3, im_size * (dst_len + 1), im_size * (src_len + 1)])

    os.makedirs('./style_mixing/output/%s/' % cfg.NAME, exist_ok=True)

    for i in range(src_len):
        save_image(src_originals[i] * 0.5 + 0.5, './style_mixing/output/%s/source_%d.png' % (cfg.NAME, i))
        place(canvas, src_originals[i], 1 + i, 0)

    for i in range(dst_len):
        save_image(dst_originals[i] * 0.5 + 0.5, './style_mixing/output/%s/dst_coarse_%d.png' % (cfg.NAME, i))
        place(canvas, dst_originals[i], 0, 1 + i)

    style_ranges = [range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, layer_count * 2)]

    def mix_styles(style_src, style_dst, r):
        style = style_dst.clone()
        style[:, r] = style_src[:, r]
        return style

    for row in range(dst_len):
        row_latents = torch.stack([dst_latents[row]] * src_len)
        style = mix_styles(src_latents, row_latents, style_ranges[row])
        rec = model.decoder(style, layer_count - 1, 1, noise=True)
        for j in range(rec.shape[0]):
            save_image(rec[j] * 0.5 + 0.5, './style_mixing/output/%s/rec_coarse_%d_%d.png' % (cfg.NAME, row, j))
            place(canvas, rec[j], 1 + j, 1 + row)

    save_image(torch.Tensor(canvas), './style_mixing/output/%s/stylemix.png' % cfg.NAME)


if __name__ == "__main__":
    gpu_count = 1
    run(main, get_cfg_defaults(), description='SandwichStyleVAE-style-mixing',
        default_config='./configs/celeba-hq256-generate.yaml',
        world_size=gpu_count, write_log=False)
