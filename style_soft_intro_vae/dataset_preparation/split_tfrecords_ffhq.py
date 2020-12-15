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

import os
import sys
import argparse
import logging
import tensorflow as tf
# from defaults import get_cfg_defaults

from yacs.config import CfgNode as CN

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
_C.MODEL.BETA_NEG = [2048, 2048, 1024, 512, 512, 128, 128, 64, 64]
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

_C.TRAIN.LOD_2_BATCH_8GPU = [512, 256, 128, 64, 32, 32]
_C.TRAIN.LOD_2_BATCH_4GPU = [512, 256, 128, 64, 32, 16]
_C.TRAIN.LOD_2_BATCH_2GPU = [256, 256, 128, 64, 32, 16]
_C.TRAIN.LOD_2_BATCH_1GPU = [128, 128, 128, 64, 32, 16]

_C.TRAIN.SNAPSHOT_FREQ = [300, 300, 300, 100, 50, 30, 20, 20, 10]

_C.TRAIN.REPORT_FREQ = [100, 80, 60, 30, 20, 10, 10, 5, 5]

_C.TRAIN.LEARNING_RATES = [0.002]

def get_cfg_defaults():
    return _C.clone()


def split_tfrecord(cfg, logger):
    tfrecord_path = cfg.DATASET.FFHQ_SOURCE

    ffhq_train_size = 60000

    part_size = ffhq_train_size // cfg.DATASET.PART_COUNT

    logger.info("Splitting into % size parts" % part_size)

    for i in range(2, cfg.DATASET.MAX_RESOLUTION_LEVEL + 1):
        with tf.Graph().as_default(), tf.Session() as sess:
            ds = tf.data.TFRecordDataset(tfrecord_path % i)
            ds = ds.batch(part_size)
            batch = ds.make_one_shot_iterator().get_next()
            part_num = 0
            while True:
                try:
                    records = sess.run(batch)
                    if part_num < cfg.DATASET.PART_COUNT:
                        part_path = cfg.DATASET.PATH % (i, part_num)
                        os.makedirs(os.path.dirname(part_path), exist_ok=True)
                        with tf.python_io.TFRecordWriter(part_path) as writer:
                            for record in records:
                                writer.write(record)
                    else:
                        part_path = cfg.DATASET.PATH_TEST % (i, part_num - cfg.DATASET.PART_COUNT)
                        os.makedirs(os.path.dirname(part_path), exist_ok=True)
                        with tf.python_io.TFRecordWriter(part_path) as writer:
                            for record in records:
                                writer.write(record)
                    part_num += 1
                except tf.errors.OutOfRangeError:
                    break


def run():
    parser = argparse.ArgumentParser(description="ALAE. Split FFHQ into parts for training and testing")
    parser.add_argument(
        "--config-file",
        default="/home/tal/tmp/StyleSandwichVAE2/configs/ffhq256.yaml",
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

    fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    split_tfrecord(cfg, logger)


if __name__ == '__main__':
    run()

