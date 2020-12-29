# style-soft-intro-vae-pytorch

Implementation of Style Soft-IntroVAE for image data.

This codes builds upon the original Adversarial Latent Autoencoders (ALAE) implementation by Stanislav Pidhorskyi.
Please see the [official repository](https://github.com/podgorskiy/ALAE) for a more detailed explanation of the files and how to get the datasets.
The authors would like to thank Stanislav Pidhorskyi, Donald A. Adjeroh and Gianfranco Doretto for their great work which inspired this implementation.

<p align="center">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/celebahq_samples.png" width="200">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/ffhq_recons.png" height="133">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/style_sintro_vae.PNG" width="400">
</p>

- [style-soft-intro-vae-pytorch](#style-soft-intro-vae-pytorch)
  * [Requirements](#requirements)
  * [Training](#training)
  * [Datasets](#datasets)
  * [Pretrained models](#pretrained-models)
  * [Recommended hyperparameters](#recommended-hyperparameters)
  * [What to expect](#what-to-expect)
  * [Files and directories in the repository](#files-and-directories-in-the-repository)
  * [Credits](#credits)

## Requirements

* Please see ALAE's repository for explanation of the requirements and how to get them.
* We provide an `environment.yml` file for `conda`, which installs all that is needed to run the files, in an environment called `tf_torch`.
  * `conda env create -f environment.yml`
* The required packages are located in the `requirements.txt` file.
  * `pip install -r requirements.txt`
  * If you installed the environment using the `environment.yml` file, there is no need to use the `requirements.txt` file.
* As in the original ALAE repository, the code is organized in such a way that all scripts must be run from the root of the repository.
  * If you use an IDE (e.g. PyCharm or Visual Studio Code), just set Working Directory to point to the root of the repository.
  * If you want to run from the command line, then you also need to set PYTHONPATH variable to point to the root of the repository.
    * Run `$ export PYTHONPATH=$PYTHONPATH:$(pwd)` in the root directory.

## Training

* This implementation uses the [DareBlopy](https://github.com/podgorskiy/DareBlopy) package to load the data.
  * `pip install dareblopy` (in addition to the `pip install -r requirements.txt`)
  * TL;DR: read TFRecords files in PyTorch, for better utilization of the data loading, train faster.
  
To run training: 

`python train_style_soft_intro_vae.py -c <config>`

* Configs are located in the `configs` directory, edit them to change the hyperparameters.
* It will run multi-GPU training on all available GPUs. It uses DistributedDataParallel for parallelism. If only one GPU available, it will run on single GPU, no special care is needed.
  * To modify the visible GPUs, edit the line: `os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"` in the `train_style_soft_intro_vae.py` file.

Examples:

`python train_style_soft_intro_vae.py` (FFHQ)

`python train_style_soft_intro_vae.py -c ./configs/celeba-hq256` 


## Datasets
* CelebHQ: please follow [ALAE](https://github.com/podgorskiy/ALAE#datasets) instructions.
* FFHQ: please see the following repository [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset).

## Pretrained models
|Dataset | Filename | Where to Put| Links|
|------------|------|----|---|
|CelebA-HQ (256x256)|`celebahq_fid_18.63_epoch_230.pth` |`training_artifacts/celeba-hq256`|[MEGA.co.nz](https://mega.nz/file/sJkS2BAC#aGFwJIPvOTIP147GwBGHOJgwRMC_NYKHT_QK7abb0VE), [Mediafire](https://www.mediafire.com/file/fgf0a85z5d0jtu5/celebahq_fid_18.63_epoch_230.pth/file) |
|FFHQ (256x256)|`ffhq_fid_17.55_epoch_270.pth` |`training_artifacts/ffhq` | [MEGA.co.nz](https://mega.nz/file/YJ1SkBwI#9t0ZEZTC0WWG0NUsJg2OwZujOuUXKn_ehP6fba1pV7o), [Mediafire](https://www.mediafire.com/file/x6jkyg4rlkqc4hl/ffhq_fid_17.55_epoch_270.pth/file) |

* In config files, `OUTPUT_DIR` points to where weights are saved to and read from. For example: `OUTPUT_DIR: training_artifacts/celeba-hq256`.

* In `OUTPUT_DIR` it saves a file `last_checkpoint` which contains the path to the actual `.pth` pickle with model weight. If you want to test the model with a specific weight file, you can simply modify `last_checkpoint` file.

## Recommended hyperparameters

|Dataset | `beta_kl` | `beta_rec`| `beta_neg`|`z_dim`|
|------------|------|----|---|----|
|FFHQ (256x256)|0.2|0.05| 512|512|
|CelebA-HQ (256x256)|0.2|0.1| 512|512|


## What to expect

* During the training, figures of samples and reconstructions are saved locally.
* During training, statistics are printed (reconstruction error, KLD, expELBO).
* In the final resolution stage (256x256), FID will be calculated every 10 epcohs.
* Tips:
    * KL of fake/rec samples should be > KL of real data (by a fair margin).
    * It is usually better to choose `beta_kl` >= `beta_rec`.
    * We stick to ALAE's original architecture hyperparameters, and mostly didn't change their configs.
  

## Files and directories in the repository

* For a full description, please ALAE's repository.

|File name         | Purpose |
|----------------------|------|
|`train_style_soft_intro_vae.py`| main training function|
|`checkpointer.py`| module for saving/restoring model weights, optimizer state and loss history.|
|`custom_adam.py`| customized adam optimizer for learning rate equalization and zero second beta.|
|`dataloader.py`| module with dataset classes, loaders, iterators, etc.|
|`defaults.py`| definition for config variables with default values.|
|`launcher.py`| helper for running multi-GPU, multiprocess training. Sets up config and logging.|
|`lod_driver.py`| helper class for managing growing/stabilizing network.|
|`lreq.py`| custom `Linear`, `Conv2d` and `ConvTranspose2d` modules for learning rate equalization.|
|`model.py`| module with high-level model definition.|
|`net.py`| definition of all network blocks for multiple architectures.|
|`registry.py`| registry of network blocks for selecting from config file.|
|`scheduler.py`| custom schedulers with warm start and aggregating several optimizers.|
|`tracker.py`| module for plotting losses.|
|`utils.py`| decorator for async call, decorator for caching, registry for network blocks.|
|`configs/celeba-hq256.yaml`, `configs/ffhq256.yaml`| config file for CelebA-HQ and FFHQ datasets at 256x256 resolution.|
|`dataset_preparation/`| folder with scripts for dataset preparation (creating and splitting TFRecords files).|
|`make_figures/`| scripts for making various figures.|
|`metrics/fid_score.py`, `metrics/inception.py`| functions for FID calculation from datasets, using the pretrained Inception network|
|` training_artifacts/`| default folder for saving checkpoints/sample outputs/plots.|

## Credits
* Adversarial Latent Autoencoders, Pidhorskyi et al., CVPR 2020 - [Code](https://github.com/podgorskiy/ALAE), [Paper](https://arxiv.org/abs/2004.04467).

