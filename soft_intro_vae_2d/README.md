# soft-intro-vae-pytorch-2d

Implementation of Soft-IntroVAE for tabular (2D) data.

A step-by-step tutorial can be found in [Soft-IntroVAE Jupyter Notebook Tutorials](https://github.com/taldatech/soft-intro-vae-pytorch/tree/main/soft_intro_vae_tutorial).

<p align="center">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/samples_plot_png_f.PNG" width="200">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/density_plot_png_f.PNG" width="200">
</p>

- [soft-intro-vae-pytorch-2d](#soft-intro-vae-pytorch-2d)
  * [Training](#training)
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
|-d, --dataset     | dataset to train on 				               	|str: '8Gaussians', '2spirals', 'checkerboard', rings'	|
|-n, --num_iter	| total number of iterations to run				| int: default=30000|
|-z, --z_dim| latent dimensions										| int: default=2|
|-s, --seed| random state to use. for random: -1 						| int: -1 , 0, 1, 2 ,....|
|-v, --num_vae| number of iterations for vanilla vae training 				| int: default=2000|
|-l, --lr| learning rate 												| float: defalut=2e-4 |
|-r, --beta_rec | beta coefficient for the reconstruction loss |float: default=0.2|
|-k, --beta_kl| beta coefficient for the kl divergence							| float: default=0.3|
|-e, --beta_neg| beta coefficient for the kl divergence in the expELBO function | float: default=0.9|
|-g, --gamma_r| coefficient for the reconstruction loss for fake data in the decoder		| float: default=1e-8|
|-b, --batch_size| batch size 											| int: default=512 |
|-p, --pretrained     | path to pretrained model, to continue training	 	|str: default="None"	|
|-c, --device| device: -1 for cpu, 0 and up for specific cuda device						|int: default=-1|


Examples:

`python main.py --dataset 8Gaussians --device 0 --seed 92 --lr 2e-4 --num_vae 2000 --num_iter 30000 --beta_kl 0.3 --beta_rec 0.2 --beta_neg 0.9`

`python main.py --dataset rings --device -1 --seed -1 --lr 2e-4 --num_vae 2000 --num_iter 30000 --beta_kl 0.2 --beta_rec 0.2 --beta_neg 1.0`

## Recommended hyperparameters

|Dataset | `beta_kl` | `beta_rec`| `beta_neg`|
|------------|------|----|---|
|`8Gaussians`|0.3|0.2| 0.9|
|`2spirals`|0.5|0.2|1.0|
|`checkerboard`|0.1|0.2|0.2|
|`rings`|0.2|0.2|1.0|


## What to expect

* During the training, figures of samples and density plots are saved locally.
* During training, statistics are printed (reconstruction error, KLD, expELBO).
* At the end of the training, the following quantities are calculated, printed and saved to a `.txt` file: grid-normalized ELBO (gnELBO), KL, JSD
* Tips:
    * KL of fake/rec samples should be >= KL of real data 
    * You will see that the deterministic reconstruction error is printed in parenthesis, it should be lower than the stochastic reconstruction error.
    * We found that for the 2D datasets, it better to initialize the networks with vanilla vae training (about 2000 iterations is good).
    

## Files and directories in the repository

|File name         | Purpose |
|----------------------|------|
|`main.py`| general purpose main application for training Soft-IntroVAE for 2D data|
|`train_soft_intro_vae_2d.py`| main training function, datasets and architectures|


## Tutorial
* [Jupyter Notebook tutorial for 2D datasets](https://github.com/taldatech/soft-intro-vae-pytorch/blob/main/soft_intro_vae_tutorial/soft_intro_vae_2d_code_tutorial.ipynb)
  * [Open in Colab](https://colab.research.google.com/github/taldatech/soft-intro-vae-pytorch/blob/main/soft_intro_vae_tutorial/soft_intro_vae_2d_code_tutorial.ipynb)