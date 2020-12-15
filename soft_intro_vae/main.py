"""
Main function for arguments parsing
Author: Tal Daniel
"""
# imports
import torch
import argparse
from train_soft_intro_vae import train_soft_intro_vae

if __name__ == "__main__":
    """
        Recommended hyper-parameters:
        - CIFAR10: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
        - SVHN: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
        - MNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
        - FashionMNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
        - Monsters: beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, z_dim: 128, batch_size: 16
        - CelebA-HQ: beta_kl: 1.0, beta_rec: 0.5, beta_neg: 1024, z_dim: 256, batch_size: 8
    """
    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-d", "--dataset", type=str,
                        help="dataset to train on: ['cifar10', 'mnist', 'fmnist', 'svhn', 'monsters128', 'celeb128', "
                             "'celeb256', 'celeb1024']")
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=250)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=128)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=2e-4)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("-v", "--num_vae", type=int, help="number of epochs for vanilla vae training", default=0)
    parser.add_argument("-r", "--beta_rec", type=float, help="beta coefficient for the reconstruction loss",
                        default=1.0)
    parser.add_argument("-k", "--beta_kl", type=float, help="beta coefficient for the kl divergence",
                        default=1.0)
    parser.add_argument("-e", "--beta_neg", type=float,
                        help="beta coefficient for the kl divergence in the expELBO function", default=1.0)
    parser.add_argument("-g", "--gamma_r", type=float,
                        help="coefficient for the reconstruction loss for fake data in the decoder", default=1e-8)
    parser.add_argument("-s", "--seed", type=int, help="seed", default=-1)
    parser.add_argument("-p", "--pretrained", type=str, help="path to pretrained model, to continue training",
                        default="None")
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device",
                        default=-1)
    parser.add_argument('-f', "--fid", action='store_true', help="if specified, FID wil be calculated during training")
    args = parser.parse_args()

    device = torch.device("cpu") if args.device <= -1 else torch.device("cuda:" + str(args.device))
    pretrained = None if args.pretrained == "None" else args.pretrained
    train_soft_intro_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_workers=0,
                         num_epochs=args.num_epochs,
                         num_vae=args.num_vae, beta_kl=args.beta_kl, beta_neg=args.beta_neg, beta_rec=args.beta_rec,
                         device=device, save_interval=50, start_epoch=0, lr_e=args.lr, lr_d=args.lr,
                         pretrained=pretrained, seed=args.seed,
                         test_iter=1000, with_fid=args.fid)
