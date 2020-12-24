# soft-intro-vae-pytorch-images

Implementation of Soft-IntroVAE for image data.

A step-by-step tutorial can be found in [Soft-IntroVAE Jupyter Notebook Tutorials](https://github.com/taldatech/soft-intro-vae-pytorch/tree/main/soft_intro_vae_tutorial).

<p align="center">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/cifar10_samples.png" width="200">
</p>

- [soft-intro-vae-pytorch-images](#soft-intro-vae-pytorch-images)
  * [Training](#training)
  * [Datasets](#datasets)
  * [Recommended hyperparameters](#recommended-hyperparameters)
  * [What to expect](#what-to-expect)
  * [Files and directories in the repository](#files-and-directories-in-the-repository)
  * [Tutorial](#tutorial)

## Training 

`main.py --help`


You should use the `main.py` file with the following arguments:

|Argument                 | Description                                 |Legal Values |
|-------------------------|---------------------------------------------|-------------|
|-h, --help       | shows arguments description             			| 			|
|-d, --dataset     | dataset to train on 				               	|str: 'cifar10', 'mnist', 'fmnist', 'svhn', 'monsters128', 'celeb128', 'celeb256', 'celeb1024'	|
|-n, --num_epochs	| total number of epochs to run			| int: default=250|
|-z, --z_dim| latent dimensions										| int: default=128|
|-s, --seed| random state to use. for random: -1 						| int: -1 , 0, 1, 2 ,....|
|-v, --num_vae| number of iterations for vanilla vae training 				| int: default=0|
|-l, --lr| learning rate 												| float: defalut=2e-4 |
|-r, --beta_rec | beta coefficient for the reconstruction loss |float: default=1.0|
|-k, --beta_kl| beta coefficient for the kl divergence							| float: default=1.0|
|-e, --beta_neg| beta coefficient for the kl divergence in the expELBO function | float: default=256.0|
|-g, --gamma_r| coefficient for the reconstruction loss for fake data in the decoder		| float: default=1e-8|
|-b, --batch_size| batch size 											| int: default=32 |
|-p, --pretrained     | path to pretrained model, to continue training	 	|str: default="None"	|
|-c, --device| device: -1 for cpu, 0 and up for specific cuda device						|int: default=-1|
|-f, --fid| if specified, FID wil be calculated during training				|bool: default=False|

Examples:

`python main.py --dataset cifar10 --device 0 --lr 2e-4 --num_epochs 250 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 256 --z_dim 128 --batch_size 32`

`python main.py --dataset mnist --device 0 --lr 2e-4 --num_epochs 200 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 256 --z_dim 32 --batch_size 128`

## Datasets
* CelebHQ: please follow [ALAE](https://github.com/podgorskiy/ALAE#datasets) instructions.
* Digital-Monsters dataset: we curated a “Digital Monsters” dataset: ~4000 images of Pokemon, Digimon and Nexomon (yes, it’s a thing). We currently don't provide a download link for this dataset (not because we are bad people), but please contact us if you wish to create it yourself.

On the left is a sample from the (very diverse) Digital-Monsters dataset (we used augmentations to enrich it), and on the right, samples generated from S-IntroVAE.
We hope this does not give you nightmares.

<p align="center">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/monsters_data.png" width="320">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/monsters_generated_c.png" width="320">
</p>

## Recommended hyperparameters

|Dataset | `beta_kl` | `beta_rec`| `beta_neg`|`z_dim`|`batch_size`|
|------------|------|----|---|----|---|
|CIFAR10 (`cifar10`)|1.0|1.0| 256|128| 32|
|SVHN (`svhn`)|1.0|1.0| 256|128| 32|
|MNIST (`mnist`)|1.0|1.0|256|32|128|
|FashionMNIST (`fmnist`)|1.0|1.0|256|32|128|
|Monsters (`monsters128`)|0.2|0.2|256|128|16|
|CelebA-HQ (`celeb256`)|0.5|1.0|1024|256|8|


## What to expect

* During the training, figures of samples and reconstructions are saved locally.
* During training, statistics are printed (reconstruction error, KLD, expELBO).
* At the end of each epoch, a summary of statistics will be printed.
* Tips:
    * KL of fake/rec samples should be >= KL of real data.
    * It is usually better to choose `beta_kl` >= `beta_rec`.
    * FID calculation is not so fast, so turn it off if you don't care about it. 

## Files and directories in the repository

|File name         | Purpose |
|----------------------|------|
|`main.py`| general purpose main application for training Soft-IntroVAE for image data|
|`train_soft_intro_vae.py`| main training function, datasets and architectures|
|`datasets.py`| classes for creating PyTorch dataset classes from images|
|`metrics/fid.py`, `metrics/inception.py`| functions for FID calculation from datasets, using the pretrained Inception network|


## Tutorial
* [Jupyter Notebook tutorial for image datasets](https://github.com/taldatech/soft-intro-vae-pytorch/blob/main/soft_intro_vae_tutorial/soft_intro_vae_image_code_tutorial.ipynb)
  * [Open in Colab](https://colab.research.google.com/github/taldatech/soft-intro-vae-pytorch/blob/main/soft_intro_vae_tutorial/soft_intro_vae_image_code_tutorial.ipynb)


