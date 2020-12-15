import zipfile
import tqdm
#from defaults import get_cfg_defaults
import sys
import logging
from torch.nn import functional as F
import torch
#from dlutils import download

from scipy import misc
# from net import *
import numpy as np
import pickle
import random
import argparse
import os
# from dlutils.pytorch.cuda_helper import *
import tensorflow as tf
import imageio
from PIL import Image
from yacs.config import CfgNode as CN
from pathlib import Path


_C = CN()

_C.NAME = ""
_C.PPL_CELEBA_ADJUSTMENT = False
_C.OUTPUT_DIR = "results"

_C.DATASET = CN()
_C.DATASET.PATH = 'celeba/data_fold_%d_lod_%d.pkl'
_C.DATASET.PATH_TEST = ''
_C.DATASET.FFHQ_SOURCE = '/data/datasets/ffhq-dataset/tfrecords/ffhq/ffhq-r%02d.tfrecords'
_C.DATASET.PART_COUNT = 1
_C.DATASET.PART_COUNT_TEST = 1
_C.DATASET.SIZE = 70000
_C.DATASET.SIZE_TEST = 10000
_C.DATASET.FLIP_IMAGES = True
_C.DATASET.SAMPLES_PATH = 'dataset_samples/faces/realign128x128'

_C.DATASET.STYLE_MIX_PATH = 'style_mixing/test_images/set_celeba/'

_C.DATASET.MAX_RESOLUTION_LEVEL = 10

_C.MODEL = CN()

_C.MODEL.LAYER_COUNT = 6
_C.MODEL.START_CHANNEL_COUNT = 64
_C.MODEL.MAX_CHANNEL_COUNT = 512
_C.MODEL.LATENT_SPACE_SIZE = 256
_C.MODEL.DLATENT_AVG_BETA = 0.995
_C.MODEL.TRUNCATIOM_PSI = 0.7
_C.MODEL.TRUNCATIOM_CUTOFF = 8
_C.MODEL.STYLE_MIXING_PROB = 0.9
_C.MODEL.MAPPING_LAYERS = 5
_C.MODEL.CHANNELS = 3
_C.MODEL.GENERATOR = "GeneratorDefault"
_C.MODEL.ENCODER = "EncoderDefault"
_C.MODEL.MAPPING_TO_LATENT = "MappingToLatent"
_C.MODEL.MAPPING_FROM_LATENT = "MappingFromLatent"
_C.MODEL.Z_REGRESSION = False
_C.MODEL.BETA_KL = 1.0
_C.MODEL.BETA_REC = 1.0
_C.MODEL.BETA_NEG = 1024
_C.MODEL.SCALE = 1 / (3 * 256 ** 2)

_C.TRAIN = CN()

_C.TRAIN.EPOCHS_PER_LOD = 15

_C.TRAIN.BASE_LEARNING_RATE = 0.0015
_C.TRAIN.ADAM_BETA_0 = 0.0
_C.TRAIN.ADAM_BETA_1 = 0.99
_C.TRAIN.LEARNING_DECAY_RATE = 0.1
_C.TRAIN.LEARNING_DECAY_STEPS = []
_C.TRAIN.TRAIN_EPOCHS = 110
_C.TRAIN.NUM_VAE = 1

_C.TRAIN.LOD_2_BATCH_8GPU = [512, 256, 128,   64,   32,    32]
_C.TRAIN.LOD_2_BATCH_4GPU = [512, 256, 128,   64,   32,    16]
_C.TRAIN.LOD_2_BATCH_2GPU = [256, 256, 128,   64,   32,    16]
_C.TRAIN.LOD_2_BATCH_1GPU = [128, 128, 128,   64,   32,    16]


_C.TRAIN.SNAPSHOT_FREQ = [300, 300, 300, 100, 50, 30, 20, 20, 10]

_C.TRAIN.REPORT_FREQ = [100, 80, 60, 30, 20, 10, 10, 5, 5]

_C.TRAIN.LEARNING_RATES = [0.002]


def get_cfg_defaults():
    return _C.clone()



def prepare_celeba(cfg, logger, train=True):
    if train:
        directory = os.path.dirname(cfg.DATASET.PATH)
    else:
        directory = os.path.dirname(cfg.DATASET.PATH_TEST)

    os.makedirs(directory, exist_ok=True)

    images = []
    # The official way of generating CelebA-HQ can be challenging.
    # Please refer to this page: https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download
    # You can get pre-generated dataset from: https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P
    source_path = str(Path.home()) + '/../../mnt/data/tal/celeb256/data256x256'
    for filename in tqdm.tqdm(os.listdir(source_path)):
        images.append((int(filename[:-4]), filename))

    print("Total count: %d" % len(images))
    if train:
        images = images[:cfg.DATASET.SIZE]
    else:
        images = images[cfg.DATASET.SIZE: cfg.DATASET.SIZE + cfg.DATASET.SIZE_TEST]

    count = len(images)
    print("Count: %d" % count)

    random.seed(0)
    random.shuffle(images)

    folds = cfg.DATASET.PART_COUNT
    celeba_folds = [[] for _ in range(folds)]

    count_per_fold = count // folds
    for i in range(folds):
        celeba_folds[i] += images[i * count_per_fold: (i + 1) * count_per_fold]

    for i in range(folds):
        if train:
            path = cfg.DATASET.PATH
        else:
            path = cfg.DATASET.PATH_TEST

        writers = {}
        for lod in range(cfg.DATASET.MAX_RESOLUTION_LEVEL, 1, -1):
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            part_path = path % (lod, i)
            os.makedirs(os.path.dirname(part_path), exist_ok=True)
            tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
            writers[lod] = tfr_writer

        for label, filename in tqdm.tqdm(celeba_folds[i]):
            img = np.asarray(Image.open(os.path.join(source_path, filename)))
            img = img.transpose((2, 0, 1))
            for lod in range(cfg.DATASET.MAX_RESOLUTION_LEVEL, 1, -1):
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))}))
                writers[lod].write(ex.SerializeToString())

                image = torch.tensor(np.asarray(img, dtype=np.float32)).view(1, 3, img.shape[1], img.shape[2])
                image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8).view(3, image.shape[2] // 2, image.shape[3] // 2).numpy()

                img = image_down


def run():
    parser = argparse.ArgumentParser(description="Adversarial, hierarchical style VAE")
    parser.add_argument(
        "--config-file",
        default="/home/tal/tmp/StyleSandwichVAE2/configs/celeba-hq256.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # data_root = str(Path.home()) + '/../../mnt/data/tal/ffhq_ds/ffhq-dataset/images1024x1024'
    cfg.MODEL.PATH = str(Path.home()) + '/../../mnt/data/tal/celebhq_256_tfrecords/celeba-r%02d.tfrecords.%03d'
    cfg.MODEL.PATH_TEST = str(Path.home()) + '/../../mnt/data/tal/celebhq_256_test_tfrecords/celeba-r%02d.tfrecords.%03d'
    cfg.SAMPLES_PATH = str(Path.home()) + '/../../mnt/data/tal/celeb256/data256x256/'
    cfg.freeze()
    

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    prepare_celeba(cfg, logger, True)
    prepare_celeba(cfg, logger, False)


if __name__ == '__main__':
    run()

