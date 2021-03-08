# 3d-soft-intro-vae-pytorch

Implementation of 3D Soft-IntroVAE for point clouds.

This codes builds upon the code base of [3D-AAE](https://github.com/MaciejZamorski/3d-AAE) 
from the paper "Adversarial Autoencoders for Compact Representations of 3D Point Clouds"
by Maciej Zamorski, Maciej Zięba, Piotr Klukowski, Rafał Nowak, Karol Kurach, Wojciech Stokowiec, and Tomasz Trzciński
<p align="center">
  <img src="https://github.com/taldatech/soft-intro-vae-web/raw/main/assets/3d_airplane.jpg" width="250">
  <img src="https://github.com/taldatech/soft-intro-vae-web/raw/main/assets/3d_chair.jpg" width="250">
</p>

<p align="center">
  <img src="https://github.com/taldatech/soft-intro-vae-web/raw/main/assets/3d_plane_to_car.gif" width="300">
</p>

- [3d-soft-intro-vae-pytorch](#3d-soft-intro-vae-pytorch)
  * [Requirements](#requirements)
  * [Training](#training)
  * [Testing](#testing)
  * [Rendering](#rendering)
  * [Datasets](#datasets)
  * [Pretrained models](#pretrained-models)
  * [Recommended hyperparameters](#recommended-hyperparameters)
  * [What to expect](#what-to-expect)
  * [Files and directories in the repository](#files-and-directories-in-the-repository)
  * [Credits](#credits)

## Requirements

* The required packages are located in the `requirements.txt` file, nothing special.
  * `pip install -r requirements.txt`
* We provide an `environment.yml` file for `conda` (at the repo's root), which installs all that is needed to run the files.
  * `conda env create -f environment.yml`

## Training

To run training: 

* Modify the hyperparameters in `/config/soft_intro_vae_hp.json`

* Run: `python train_soft_intro_vae_3d.py`

## Testing

* To test the generations from a trained model in terms of JSD, modify `path_to_weights` and `config_path` in `test_model.py` and run it: `python test_model.py`.
* To produce reconstructed and generated point clouds in a form of NumPy array to be used with validation methods from ["Learning Representations and Generative Models For 3D Point Clouds" repository](https://github.com/optas/latent_3d_points/blob/master/notebooks/compute_evaluation_metrics.ipynb)
modify `path_to_weights` and `config_path` in `evaluation/generate_data_for_metrics.py` and run: `python evaluation/generate_data_for_metrics.py` 

## Rendering
* To render beautiful point clouds from a trained model, we provide a script that uses Mitsuba 2 renderer. Instructions can be found in `/render`.

## Datasets
* We currently support [ShapeNet](https://shapenet.org/), which will be downloaded automatically on first run.

## Pretrained models
|Dataset/Class | Filename | Validation Sample JSD| Links|
|------------|------|----|---|
|ShapeNet-Chair|`chair_01618_jsd_0.0175.pth` |0.0175|[MEGA.co.nz](https://mega.nz/file/RJ8mmIjL#DKuvWImRZdzKL_JN9JwwsvZw3F4Iv0i5g0qaLiSL84Q), [Mediafire](http://www.mediafire.com/file/i9ozb2yv4bv1i76/chair_01618_jsd_0.0175.pth/file) |
|ShapeNet-Table|`table_01592_jsd_0.0143.pth` |0.0143 | [MEGA.co.nz](https://mega.nz/file/ZQ8GjSQB#ctGaJXgvUsgaMYQm1R3bfMUzKld7nGO-oUAGGA9EOX8), [Mediafire](http://www.mediafire.com/file/hvygeusesaa58y2/table_01592_jsd_0.0143.pth/file) |
|ShapeNet-Car|`car_01344_jsd_0.0113.pth` |0.0113 | [MEGA.co.nz](https://mega.nz/file/kZ0AQQQL#hecHNlPyh0ww3_RZOvrXCE48yr5ZmfL3RZ01MSz2NwU), [Mediafire](http://www.mediafire.com/file/ja1p9wjnc58uab4/car_01344_jsd_0.0113.pth/file) |
|ShapeNet-Airplane|`airplane_00536_jsd_0.0191.pth` |0.0191 | [MEGA.co.nz](https://mega.nz/file/xA9g0ajA#jyhBgPQC4VxLwgDPfk-xo_xAbCUQofzVz9jdP0OUvDc), [Mediafire](http://www.mediafire.com/file/79ett5dhhwm2yl8/airplane_00536_jsd_0.0191.pth/file) |


## Recommended hyperparameters

|Dataset | `beta_kl` | `beta_rec`| `beta_neg`|`z_dim`|
|------------|------|----|---|----|
|ShapeNet|0.2|1.0| 20.0|128|


## What to expect

* During the training, figures of samples and reconstructions are saved locally.
  * First row - real, second row - reconstructions, third row - random samples
* During training, statistics are printed (reconstruction error, KLD, expELBO).
* Checkpoint is saved every epoch, and JSD is calculated on the validation split.
* Tips:
    * KL of fake/rec samples should be > KL of real data (by a fair margin).
    * Currently, this code only supports the Chamfer Distance loss, which requires high `beta_rec`.
    * Since the practice is to train on a single class, it is usually better to use a narrower Gaussian for the prior (e.g., N(0, 0.2)).
  

## Files and directories in the repository

|File name         | Purpose |
|----------------------|------|
|`train_soft_intro_vae_3d.py`| main training function.|
|`generate_for_rendering.py`| generate samples (+interpolation) from a trained model for rendering with Mitsuba.|
|`test_model.py`| test sampling JSD of a trained model (w.r.t the test split).|
|`config/soft_intro_vae_hp.json`| contains the hyperparmeters of the model.|
|`/datasets`| directory containing various datasets files (e.g., data loader for ShapeNet).|
|`/evaluations`| directory containing evaluation scrips (e.g., generating data for evaluation metrics).|
|`/losses/chamfer_loss.py`| PyTorch implementation of the Chamfer distance loss function.|
|`/metrics/jsd.py`| functions to measure JSD between point clouds.|
|`/models/vae.py`| VAE module and architecture.|
|`/render`| directory containing scripts and instructions to render point clouds with Mitsuba 2 renderer.|
|`/utils`| various utility functions to process the data|

## Credits
* Adversarial Autoencoders for Compact Representations of 3D Point Clouds, Zamorski et al., 2018 - [Code](https://github.com/MaciejZamorski/3d-AAE), [Paper](https://arxiv.org/abs/1811.07605).

