import torch
import torch.nn as nn

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['D']['use_bias']
        self.relu_slope = config['model']['D']['relu_slope']
        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=2048 * 3, bias=self.use_bias),
        )

    def forward(self, input):
        output = self.model(input.squeeze())
        output = output.view(-1, 3, 2048)
        return output


class EncoderNoBatchNorm(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
                      bias=self.use_bias),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        self.std_layer = nn.Linear(256, self.z_size, bias=True)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        return mu, logvar


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        self.std_layer = nn.Linear(256, self.z_size, bias=True)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        return mu, logvar


class SoftIntroVAE(nn.Module):
    def __init__(self, config):
        super(SoftIntroVAE, self).__init__()

        self.zdim = config['z_size']

        self.encoder = Encoder(config)

        self.decoder = Decoder(config)

    def forward(self, x, deterministic=False):
        mu, logvar = self.encoder(x)
        if deterministic:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        y = self.decoder(z)
        return y, mu, logvar

    def sample(self, z):
        y = self.decode(z)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu")):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decode(z)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        y = self.decoder(z)
        return y


class SoftIntroVAEBootstrap(nn.Module):
    def __init__(self, config):
        super(SoftIntroVAEBootstrap, self).__init__()

        self.zdim = config['z_size']

        self.encoder = Encoder(config)

        self.decoder = Decoder(config)

        self.target_decoder = Decoder(config)

    def forward(self, x, deterministic=False, use_target_decoder=True):
        mu, logvar = self.encoder(x)
        if deterministic:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        if use_target_decoder:
            y = self.target_decoder(z)
        else:
            y = self.decoder(z)
        return y, mu, logvar

    def sample(self, z, use_target_decoder=False):
        if use_target_decoder:
            y = self.decode_target(z)
        else:
            y = self.decode(z)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu")):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decode(z)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        y = self.decoder(z)
        return y

    def decode_target(self, z):
        y = self.target_decoder(z)
        return y
