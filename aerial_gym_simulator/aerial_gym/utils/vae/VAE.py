import torch
import torch.nn as nn


class ImgDecoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=64, with_logits=False):
        """
        Parameters
        ----------
        latent_dim: int
            The latent dimension.
        """

        super(ImgDecoder, self).__init__()
        print("[ImgDecoder] Starting create_model")
        self.with_logits = with_logits
        self.n_channels = input_dim
        self.dense = nn.Linear(latent_dim, 512)
        self.dense1 = nn.Linear(512, 9 * 15 * 128)
        # Pytorch docs: output_padding is only used to find output shape, but does not actually add zero-padding to output
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=5, stride=2, padding=(2, 2), output_padding=(0, 1), dilation=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            64, 32, kernel_size=6, stride=4, padding=(2, 2), output_padding=(0, 0), dilation=1
        )
        self.deconv6 = nn.ConvTranspose2d(
            32, 16, kernel_size=6, stride=2, padding=(0, 0), output_padding=(0, 1)
        )
        self.deconv7 = nn.ConvTranspose2d(
            16, self.n_channels, kernel_size=4, stride=2, padding=2
        )  # tanh activation or sigmoid
        print("[ImgDecoder] Done with create_model")
        print("Defined decoder.")

    def forward(self, z):
        return self.decode(z)

    def decode(self, z):
        x = self.dense(z)
        x = torch.relu(x)
        x = self.dense1(x)
        x = x.view(x.size(0), 128, 9, 15)

        x = self.deconv1(x)
        x = torch.relu(x)

        x = self.deconv2(x)
        x = torch.relu(x)

        x = self.deconv4(x)
        x = torch.relu(x)

        x = self.deconv6(x)
        x = torch.relu(x)

        x = self.deconv7(x)
        # print(f"- After deconv 7, mean: {x.mean():.3f} var: {x.var():.3f}")
        if self.with_logits:
            return x

        x = torch.sigmoid(x)
        # print(f"- After sigmoid, mean: {x.mean():.3f} var: {x.var():.3f}")
        return x


class ImgEncoder(nn.Module):
    """
    ResNet8 architecture as encoder.
    """

    def __init__(self, input_dim, latent_dim):
        """
        Parameters:
        ----------
        input_dim: int
            Number of input channels in the image.
        latent_dim: int
            Number of latent dimensions
        """
        super(ImgEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.define_encoder_dronet()
        self.elu = nn.ELU()
        print(f"Defined encoder with input_dim: {input_dim} and latent_dim: {latent_dim}")

    def define_encoder(self):
        # define conv functions
        self.conv0 = nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=2, padding=2)
        self.conv0_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2)
        nn.init.xavier_uniform_(self.conv0_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv0_1.bias)

        self.conv1_0 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv1_1.bias)

        self.conv2_0 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv2_1.bias)

        self.conv3_0 = nn.Conv2d(128, 128, kernel_size=5, stride=2)

        self.conv0_jump_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv1_jump_3 = nn.Conv2d(64, 128, kernel_size=5, stride=4, padding=(2, 1))

        self.dense0 = nn.Linear(3 * 6 * 128, 512)
        self.dense1 = nn.Linear(512, 2 * self.latent_dim)

        print("Encoder network initialized.")

    def define_encoder_dronet(self):
        # first convolutional layer: 32 filters of size 5x5, 200x200, /2
        self.conv0 = nn.Conv2d(in_channels=self.input_dim, out_channels=32, kernel_size=5, stride=2, padding=2, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv0_bn = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv0_relu = nn.ReLU(inplace=False)
        # Max pooling layer: max_pool 2x2, 32, 100x100, /2
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # Three blocks of Depthwise separable convolutions
        # Block 1: depthwise separable convolution, 32, 32, 25x25, /2
        self.conv1_dw = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1, groups=32, bias=False, padding_mode='zeros')
        self.conv1_dw_bn = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1_dw_relu = nn.ReLU6(inplace=False)
        # Block 1: pointwise separable convolution, 32, 32, 25x25, /1
        self.conv1_pw = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias=False)
        self.conv1_pw_bn = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1_pw_relu = nn.ReLU6(inplace=False)
        # Block 2: depthwise separable convolution, 32, 32, 50x50, /2
        self.conv2_dw = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, groups=32, bias=False, padding_mode='zeros')
        self.conv2_dw_bn = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_dw_relu = nn.ReLU6(inplace=False)
        # Block 2: pointwise separable convolution, 32, 32, 50x50, /1
        self.conv2_pw = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias=False)
        self.conv2_pw_bn = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_pw_relu = nn.ReLU6(inplace=False)

        # Block 3: depthwise separable convolution, 32, 32, 50x50, /2
        self.conv3_dw = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1, groups=32, bias=False, padding_mode='zeros')
        self.conv3_dw_bn = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_dw_relu = nn.ReLU6(inplace=False)
        # Block 3: pointwise separable convolution, 32, 64, 50x50, /1
        self.conv3_pw = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=False)
        self.conv3_pw_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_pw_relu = nn.ReLU6(inplace=False)
        # Block 4: depthwise separable convolution, 64, 64, 13x13, /2
        self.conv4_dw = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=64, bias=False, padding_mode='zeros')
        self.conv4_dw_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_dw_relu = nn.ReLU6(inplace=False)
        # Block 4: pointwise separable convolution, 64, 64, 13x13, /1
        self.conv4_pw = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False)
        self.conv4_pw_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_pw_relu = nn.ReLU6(inplace=False)

        # Block 5: depthwise separable convolution, 64, 64, 13x13, /2
        self.conv5_dw = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, groups=64, bias=False, padding_mode='zeros')
        self.conv5_dw_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_dw_relu = nn.ReLU6(inplace=False)
        # Block 5: pointwise separable convolution, 64, 128, 13x13, /1
        self.conv5_pw = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, bias=False)
        self.conv5_pw_bn = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_pw_relu = nn.ReLU6(inplace=False)
        # Block 6: depthwise separable convolution, 128, 128, 7x7, /1
        self.conv6_dw = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=128, bias=False, padding_mode='zeros')
        self.conv6_dw_bn = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6_dw_relu = nn.ReLU6(inplace=False)
        # Block 6: pointwise separable convolution, 128, 128, 7x7, /1
        self.conv6_pw = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, bias=False)
        self.conv6_pw_bn = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6_pw_relu = nn.ReLU6(inplace=False)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        # Fully connected layer
        self.fc = nn.Linear(in_features=128*7*7, out_features=512, bias=False)
        self.fc_sig = nn.Sigmoid()
        # Latent layer
        self.fc_latent = nn.Linear(in_features=512, out_features=2 * self.latent_dim)

        print("Encoder network initialized: DRONETv3")

    def forward(self, img):
        return self.encode_dronet(img)
    
    def encode_dronet(self, img):
        # First convolutional layer
        x0_0 = self.conv0(img)
        x0_1 = self.conv0_bn(x0_0)
        x0_1 = self.conv0_relu(x0_1)
        # Max pooling layer
        x0_1 = self.max_pool(x0_1)
        # Three blocks of Depthwise separable convolutions
        # Block 1
        x1_0 = self.conv1_dw(x0_1)
        x1_1 = self.conv1_dw_bn(x1_0)
        x1_1 = self.conv1_dw_relu(x1_1)
        x1_2 = self.conv1_pw(x1_1)
        x1_3 = self.conv1_pw_bn(x1_2)
        x1_3 = self.conv1_pw_relu(x1_3)
        # Block 2
        x2_0 = self.conv2_dw(x1_3)
        x2_1 = self.conv2_dw_bn(x2_0)
        x2_1 = self.conv2_dw_relu(x2_1)
        x2_2 = self.conv2_pw(x2_1)
        x2_3 = self.conv2_pw_bn(x2_2)
        x2_3 = self.conv2_pw_relu(x2_3)
        # Block 3
        x3_0 = self.conv3_dw(x2_3)
        x3_1 = self.conv3_dw_bn(x3_0)
        x3_1 = self.conv3_dw_relu(x3_1)
        x3_2 = self.conv3_pw(x3_1)
        x3_3 = self.conv3_pw_bn(x3_2)
        x3_3 = self.conv3_pw_relu(x3_3)
        # Block 4
        x4_0 = self.conv4_dw(x3_3)
        x4_1 = self.conv4_dw_bn(x4_0)
        x4_1 = self.conv4_dw_relu(x4_1)
        x4_2 = self.conv4_pw(x4_1)
        x4_3 = self.conv4_pw_bn(x4_2)
        x4_3 = self.conv4_pw_relu(x4_3)
        # Block 5
        x5_0 = self.conv5_dw(x4_3)
        x5_1 = self.conv5_dw_bn(x5_0)
        x5_1 = self.conv5_dw_relu(x5_1)
        x5_2 = self.conv5_pw(x5_1)
        x5_3 = self.conv5_pw_bn(x5_2)
        x5_3 = self.conv5_pw_relu(x5_3)
        # Block 6
        x6_0 = self.conv6_dw(x5_3)
        x6_1 = self.conv6_dw_bn(x6_0)
        x6_1 = self.conv6_dw_relu(x6_1)
        x6_2 = self.conv6_pw(x6_1)
        x6_3 = self.conv6_pw_bn(x6_2)
        x6_3 = self.conv6_pw_relu(x6_3)
        # Dropout layer
        x6_3 = self.dropout(x6_3)
        # Fully connected layer
        x = x6_3.view(x6_3.size(0), -1)
        x = self.fc(x)
        x = self.fc_sig(x)
        # Latent layer
        x = self.fc_latent(x)
        return x

    def encode(self, img):
        """
        Encodes the input image.
        """

        # conv0
        x0_0 = self.conv0(img)
        x0_1 = self.conv0_1(x0_0)
        x0_1 = self.elu(x0_1)

        x1_0 = self.conv1_0(x0_1)
        x1_1 = self.conv1_1(x1_0)

        x0_jump_2 = self.conv0_jump_2(x0_1)

        x1_1 = x1_1 + x0_jump_2

        x1_1 = self.elu(x1_1)

        x2_0 = self.conv2_0(x1_1)
        x2_1 = self.conv2_1(x2_0)

        x1_jump3 = self.conv1_jump_3(x1_1)

        x2_1 = x2_1 + x1_jump3

        x2_1 = self.elu(x2_1)

        x3_0 = self.conv3_0(x2_1)

        x = x3_0.view(x3_0.size(0), -1)

        x = self.dense0(x)
        x = self.elu(x)
        x = self.dense1(x)
        return x


class Lambda(nn.Module):
    """Lambda function that accepts tensors as input."""

    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class VAE(nn.Module):
    """Variational Autoencoder for reconstruction of depth images."""

    def __init__(self, input_dim=1, latent_dim=64, with_logits=False, inference_mode=False):
        """
        Parameters
        ----------
        input_dim: int
            The number of input channels in an image.
        latent_dim: int
            The latent dimension.
        """

        super(VAE, self).__init__()

        self.with_logits = with_logits
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.inference_mode = inference_mode
        self.encoder = ImgEncoder(input_dim=self.input_dim, latent_dim=self.latent_dim)
        self.img_decoder = ImgDecoder(
            input_dim=1, latent_dim=self.latent_dim, with_logits=self.with_logits
        )

        self.mean_params = Lambda(lambda x: x[:, : self.latent_dim])  # mean parameters
        self.logvar_params = Lambda(lambda x: x[:, self.latent_dim :])  # log variance parameters

    def forward(self, img):
        """Do a forward pass of the VAE. Generates a reconstructed image based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """

        # encode
        z = self.encoder(img)

        # reparametrization trick
        mean = self.mean_params(z)
        logvar = self.logvar_params(z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = mean + eps * std

        # decode
        img_recon = self.img_decoder(z_sampled)
        return img_recon, mean, logvar, z_sampled

    def forward_test(self, img):
        """Do a forward pass of the VAE. Generates a reconstructed image based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """

        # encode
        z = self.encoder(img)

        # reparametrization trick
        mean = self.mean_params(z)
        logvar = self.logvar_params(z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = mean + eps * std

        # decode
        img_recon = self.img_decoder(z_sampled)
        return img_recon, mean, logvar, z_sampled

    def encode(self, img):
        """Do a forward pass of the VAE. Generates a latent vector based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """
        z = self.encoder(img)

        means = self.mean_params(z)
        logvars = self.logvar_params(z)
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(logvars)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = means + eps * std

        return z_sampled, means, std

    def decode(self, z):
        """Do a forward pass of the VAE. Generates a reconstructed image based on z
        Parameters
        ----------
        z: torch.Tensor
            The latent vector.
        """
        img_recon = self.img_decoder(z)
        if self.with_logits:
            return torch.sigmoid(img_recon)
        return img_recon

    def set_inference_mode(self, mode):
        self.inference_mode = mode
