# soft-intro-vae-pytorch

<h1 align="center">
  <br>
Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders
  <br>
</h1>
  <p align="center">
    <a href="https://github.com/taldatech">Tal Daniel</a> •
    <a href="https://avivt.github.io/avivt/">Aviv Tamar</a>

  </p>
<h4 align="center">Official repository of the paper</h4>

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/soft-intro-vae-pytorch"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</h4>


<p align="center">
  <img src="https://github.com/taldatech/soft-intro-vae-web/raw/main/assets/ffhq_samples.png" style="height:250px">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/celebahq_recons.png" style="height:250px">
</p>

# Soft-IntroVAE

> **Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders**<br>
> Tal Daniel, Aviv Tamar<br>
>
> **Abstract:** *The recently introduced introspective variational autoencoder (IntroVAE) exhibits outstanding image generations, and allows for amortized inference using an image encoder. The main idea in IntroVAE is to train a VAE adversarially, using the VAE encoder to discriminate between generated and real data samples. However, the original IntroVAE loss function relied on a particular hinge-loss formulation that is very hard to stabilize in practice, and its theoretical convergence analysis ignored important terms in the loss.
In this work, we take a step towards better understanding of the IntroVAE model, its practical implementation, and its applications. We propose the Soft-IntroVAE, a modified IntroVAE that replaces the hinge-loss terms with a smooth exponential loss on generated samples. This change significantly improves training stability, and also enables theoretical analysis of the complete algorithm. Interestingly, we show that the IntroVAE converges to a distribution that minimizes a sum of KL distance from the data distribution and an entropy term. We discuss the implications of this result, and demonstrate that it induces competitive image generation and reconstruction. Finally, we describe an application of Soft-IntroVAE to unsupervised image translation, and demonstrate compelling results.*

## Citation
* Stanislav Pidhorskyi, Donald A. Adjeroh, and Gianfranco Doretto. Adversarial Latent Autoencoders. In *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020. 
>
    @InProceedings{pidhorskyi2020adversarial,
     author   = {Pidhorskyi, Stanislav and Adjeroh, Donald A and Doretto, Gianfranco},
     booktitle = {Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)},
     title    = {Adversarial Latent Autoencoders},
     year     = {2020},
     note     = {[to appear]},
    }
<h4 align="center">Preprint on arXiv: <a href="link">Number</a></h4>


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
|`tensorboard`|  >= `1.10.0`|
|`tensorboardX`|  >= `1.4`|
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
|`/soft_intro_vae_bootstrap`| directory containing implementation for image data using bootstrapping (using a target decoder)|
|`/style_soft_intro_vae`| directory containing implementation for image data using ALAE's style-based architecture|
|`/soft_intro_vae_tutorials`| directory containing Jupyter Noteboook tutorials for the various types of Soft-IntroVAE|


## Credits
* Adversarial Latent Autoencoders, Pidhorskyi et al., CVPR 2020 - [Code](https://github.com/podgorskiy/ALAE), [Paper](https://arxiv.org/abs/2004.04467).
* FID is calculated natively in PyTorch using Seitzer implementation - [Code](https://github.com/mseitzer/pytorch-fid)



