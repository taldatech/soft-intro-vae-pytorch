"""
Main function for arguments parsing
Author: Tal Daniel
"""
# imports
import torch
import argparse
from train_soft_intro_vae_2d import train_soft_intro_vae_toy

if __name__ == "__main__":
    """
        Recommended hyper-parameters:
        - 8Gaussians: beta_kl: 0.3, beta_rec: 0.2, beta_neg: 0.9, z_dim: 2, batch_size: 512
        - 2spirals: beta_kl: 0.5, beta_rec: 0.2, beta_neg: 1.0, z_dim: 2, batch_size: 512
        - checkerboard: beta_kl: 0.1, beta_rec: 0.2, beta_neg: 0.2, z_dim: 2, batch_size: 512
        - rings: beta_kl: 0.2, beta_rec: 0.2, beta_neg: 1.0, z_dim: 2, batch_size: 512
    """
    parser = argparse.ArgumentParser(description="train Soft-IntroVAE 2D")
    parser.add_argument("-d", "--dataset", type=str,
                        help="dataset to train on: ['8Gaussians', '2spirals', 'checkerboard', rings']")
    parser.add_argument("-n", "--num_iter", type=int, help="total number of iterations to run", default=30_000)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=2)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=2e-4)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=512)
    parser.add_argument("-v", "--num_vae", type=int, help="number of iterations for vanilla vae training", default=2000)
    parser.add_argument("-r", "--beta_rec", type=float, help="beta coefficient for the reconstruction loss",
                        default=0.2)
    parser.add_argument("-k", "--beta_kl", type=float, help="beta coefficient for the kl divergence",
                        default=0.3)
    parser.add_argument("-e", "--beta_neg", type=float,
                        help="beta coefficient for the kl divergence in the expELBO function", default=0.9)
    parser.add_argument("-g", "--gamma_r", type=float,
                        help="coefficient for the reconstruction loss for fake data in the decoder", default=1e-8)
    parser.add_argument("-s", "--seed", type=int, help="seed", default=-1)
    parser.add_argument("-p", "--pretrained", type=str, help="path to pretrained model, to continue training",
                        default="None")
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device",
                        default=-1)
    args = parser.parse_args()

    device = torch.device("cpu") if args.device <= -1 else torch.device("cuda:" + str(args.device))
    pretrained = None if args.pretrained == "None" else args.pretrained
    if args.dataset == '8Gaussians':
        scale = 1
    else:
        scale = 2
    # train
    model = train_soft_intro_vae_toy(z_dim=args.z_dim, lr_e=args.lr, lr_d=args.lr, batch_size=args.batch_size,
                                     n_iter=args.num_iter, num_vae=args.num_vae, save_interval=5000,
                                     recon_loss_type="mse", beta_kl=args.beta_kl, beta_rec=args.beta_rec,
                                     beta_neg=args.beta_neg, test_iter=5000, seed=args.seed, scale=scale,
                                     device=device, dataset=args.dataset)
