# soft-intro-vae-pytorch

<h1 align="center">
  <br>
[CVPR 2021 Oral] Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders
  <br>
</h1>
  <p align="center">
    <a href="https://taldatech.github.io">Tal Daniel</a> •
    <a href="https://avivt.github.io/avivt/">Aviv Tamar</a>

  </p>
<h4 align="center">Official repository of the paper</h4>

<h4 align="center">CVPR 2021 Oral</h4>

<h4 align="center"><a href="https://taldatech.github.io/soft-intro-vae-web">Project Website</a> • <a href="https://www.youtube.com/watch?v=1NfsSYoHnBg">Video</a></h4>

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/soft-intro-vae-pytorch"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</h4>


<p align="center">
  <img src="https://github.com/taldatech/soft-intro-vae-web/raw/main/assets/ffhq_samples.png" height="120">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/celebahq_recons.png" height="120">
</p>
<p align="center">
  <img src="https://github.com/taldatech/soft-intro-vae-web/raw/main/assets/3d_plane_to_car.gif" height="100">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/density_plot_png_f.PNG" height="100">
</p>

# Soft-IntroVAE

> **Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders**<br>
> Tal Daniel, Aviv Tamar<br>
>
> **Abstract:** *The recently introduced introspective variational autoencoder (IntroVAE) exhibits outstanding image generations, and allows for amortized inference using an image encoder. The main idea in IntroVAE is to train a VAE adversarially, using the VAE encoder to discriminate between generated and real data samples. However, the original IntroVAE loss function relied on a particular hinge-loss formulation that is very hard to stabilize in practice, and its theoretical convergence analysis ignored important terms in the loss. In this work, we take a step towards better understanding of the IntroVAE model, its practical implementation, and its applications. We propose the Soft-IntroVAE, a modified IntroVAE that replaces the hinge-loss terms with a smooth exponential loss on generated samples. This change significantly improves training stability, and also enables theoretical analysis of the complete algorithm. Interestingly, we show that the IntroVAE converges to a distribution that minimizes a sum of KL distance from the data distribution and an entropy term. We discuss the implications of this result, and demonstrate that it induces competitive image generation and reconstruction. Finally, we describe two applications of Soft-IntroVAE to unsupervised image translation and out-of-distribution detection, and demonstrate compelling results.*

## Citation
Daniel, Tal, and Aviv Tamar. "Soft-IntroVAE: Analyzing and Improving the Introspective Variational Autoencoder." arXiv preprint arXiv:2012.13253 (2020).
>
    @InProceedings{Daniel_2021_CVPR,
    author    = {Daniel, Tal and Tamar, Aviv},
    title     = {Soft-IntroVAE: Analyzing and Improving the Introspective Variational Autoencoder},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {4391-4400}
}

<h4 align="center">Preprint on ArXiv: <a href="https://arxiv.org/abs/2012.13253">2012.13253</a></h4>


- [soft-intro-vae-pytorch](#soft-intro-vae-pytorch)
- [Soft-IntroVAE](#soft-introvae)
  * [Citation](#citation)
  * [Prerequisites](#prerequisites)
  * [Repository Organization](#repository-organization)
  * [Credits](#credits)
    

## Prerequisites

* For your convenience, we provide an `environemnt.yml` file which installs the required packages in a `conda` environment name `torch`.
    * Use the terminal or an Anaconda Prompt and run the following command `conda env create -f environment.yml`.
* For Style-SoftIntroVAE, more packages are required, and we provide them in the `style_soft_intro_vae` directory.


|Library         | Version |
|----------------------|----|
|`Python`|  `3.6 (Anaconda)`|
|`torch`|  >= `1.2` (tested on `1.7`)|
|`torchvision`|  >= `0.4`|
|`matplotlib`|  >= `2.2.2`|
|`numpy`|  >= `1.17`|
|`opencv`|  >= `3.4.2`|
|`tqdm`| >= `4.36.1`|
|`scipy`| >= `1.3.1`|



## Repository Organization

|File name         | Content |
|----------------------|------|
|`/soft_intro_vae`| directory containing implementation for image data|
|`/soft_intro_vae_2d`| directory containing implementations for 2D datasets|
|`/soft_intro_vae_3d`| directory containing implementations for 3D point clouds data|
|`/soft_intro_vae_bootstrap`| directory containing implementation for image data using bootstrapping (using a target decoder)|
|`/style_soft_intro_vae`| directory containing implementation for image data using ALAE's style-based architecture|
|`/soft_intro_vae_tutorials`| directory containing Jupyter Noteboook tutorials for the various types of Soft-IntroVAE|

## Related Projects

* March 2022: `augmentation-enhanced-soft-intro-vae` - <a href="https://github.com/baruch1192/augmentation-enhanced-Soft-Intro-VAE">GitHub</a> - using differentiable augmentations to improve image generation FID score.


## Credits
* Adversarial Latent Autoencoders, Pidhorskyi et al., CVPR 2020 - [Code](https://github.com/podgorskiy/ALAE), [Paper](https://arxiv.org/abs/2004.04467).
* FID is calculated natively in PyTorch using Seitzer implementation - [Code](https://github.com/mseitzer/pytorch-fid)



