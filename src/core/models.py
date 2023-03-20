# Define you transfomer model here
import torch
import torch.nn as nn
from math import floor


class UMMT(nn.Module):
    pass

class MLP(nn.Module):
    """
    Two-layer MLP network
    # TODO: Add link to MICCAI paper

    Attributes
    ----------
    fc_1: torch.nn.Module, first FC linear layer
    fc_2: torch.nn.Module, second FC linear layer
    bn: torch.nn.Module, batch normalization layer
    dropout_p: float, dropout ratio

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 80,
                 output_dim: int = 128,
                 dropout_p: int = 0.0):
        """
        :param input_dim: int, dimension of input embeddings
        :param hidden_dim: int, dimension of hidden embeddings
        :param output_dim: int, dimension of output embeddings
        :param dropout_p: float, dropout used in between layers
        """

        super().__init__()

        # Linear layers
        self.fc_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc_2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        # Initialize batch norm
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.bn_out = nn.BatchNorm1d(output_dim)

        # Dropout params
        self.dropout_p = dropout_p

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor
        :return: transformed embeddings
        """

        # Two FC layers
        x = F.relu(self.bn_out(self.fc_2(F.dropout(F.relu(self.bn(self.fc_1(x))),
                                                   p=self.dropout_p,
                                                   training=self.training))))

        return x


class CNNResBlock(nn.Module):
    """
    3D convolution block with residual connections
    #TODO: Add link to code for MICCAI paper

    Attributes
    ----------
    conv: torch.nn.Conv3d, PyTorch Conv3D model
    bn: torch.nn.BatchNorm3d, PyTorch 3D batch normalization layer
    pool: torch.nn.AvgPool3d, PyTorch average 3D pooling layer
    dropout: torch.nn.Dropout3D, PyTorch 3D dropout layer
    one_by_one_cnn: torch.nn.Conv3d, pyTorch 1*1 conv model to equalize the number of channels for residual addition

    Methods
    -------
    forward(x): model's forward propagation
    """
    def __init__(self,
                 in_channels: int,
                 padding: int,
                 out_channels: int = 128,
                 kernel_size: int = 3,
                 pool_size: int = 2,
                 out_size: int = None,
                 cnn_dropout_p: float = 0.0):
        """
        :param in_channels: int, number of input channels
        :param padding: int, 0 padding dims
        :param out_channels: int, number of filters to use
        :param kernel_size: int, filter size
        :param pool_size: int, pooling kernel size for the spatial dims (if out_size is kept as None)
        :param out_size: int, output frame dimension for adaptive pooling
        :param cnn_dropout_p: float, cnn dropout rate
        """

        super().__init__()

        # Check if a Conv would be needed to make the channel dim the same
        # for the residual
        self.one_by_one_cnn = None
        if in_channels != out_channels:
            # noinspection PyTypeChecker
            self.one_by_one_cnn = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1)

        # 2D conv layer
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(padding, padding))

        # Other operations
        self.bn = nn.BatchNorm2d(out_channels)
        if out_size is None:
            self.pool = nn.MaxPool2d(kernel_size=(pool_size, pool_size))
        else:
            self.pool = nn.AdaptiveMaxPool2d(output_size=(out_size, out_size))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(p=cnn_dropout_p)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor of shape N*1*64*T*H*W
        :return: Tensor of shape N*out_channels*T*H'*W'
        """

        # Make the number of channels equal for input and output if needed
        if self.one_by_one_cnn is not None:
            residual = self.one_by_one_cnn(x)
        else:
            residual = x

        x = self.conv(x)
        x = self.bn(x)
        x = x + residual
        x = self.pool(x)
        x = self.activation(x)

        return self.dropout(x)


class CNN(nn.Module):
    """
    3D convolution network
    # TODO: Add link to MICCAI paper

    Attributes
    ----------
    conv: torch.nn.Sequential, the convolutional network containing residual blocks
    output_fc: torch.nn.Sequential, the FC layer applied to the output of convolutional network

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 out_channels: list,
                 kernel_sizes: list = None,
                 pool_sizes: list = None,
                 fc_output_dim: list = None,
                 cnn_dropout_p: float = 0.0):
        """
        :param out_channels: list, output channels for each layer
        :param kernel_sizes: list, kernel sizes for each layer
        :param pool_sizes: list, pooling kernel sizes for each layer
        :param fc_output_dim: int, the output dimension of output FC layer (set to None for no output fc)
        :param cnn_dropout_p: float, dropout ratio of the CNN
        """

        super().__init__()

        n_conv_layers = len(out_channels)

        # Default list arguments
        if kernel_sizes is None:
            kernel_sizes = [3]*n_conv_layers
        if pool_sizes is None:
            pool_sizes = [1]*n_conv_layers

        # Ensure input params are list
        if type(out_channels) is not list:
            out_channels = [out_channels]*n_conv_layers
        else:
            assert len(out_channels) == n_conv_layers, 'Provide channel parameter for all layers.'
        if type(kernel_sizes) is not list:
            kernel_sizes = [kernel_sizes]*n_conv_layers
        else:
            assert len(kernel_sizes) == n_conv_layers, 'Provide kernel size parameter for all layers.'
        if type(pool_sizes) is not list:
            pool_sizes = [pool_sizes]*n_conv_layers
        else:
            assert len(pool_sizes) == n_conv_layers, 'Provide pool size parameter for all layers.'

        # Compute paddings to preserve temporal dim
        paddings = list()
        for kernel_size in kernel_sizes:
            paddings.append(floor((kernel_size - 1) / 2))

        # Conv layers
        convs = list()

        # Add first layer
        convs.append(nn.Sequential(CNNResBlock(in_channels=1,
                                               padding=paddings[0],
                                               out_channels=out_channels[0],
                                               kernel_size=kernel_sizes[0],
                                               pool_size=pool_sizes[0],
                                               cnn_dropout_p=cnn_dropout_p)))

        # Add subsequent layers
        for layer_num in range(1, n_conv_layers):
            convs.append(nn.Sequential(CNNResBlock(in_channels=out_channels[layer_num-1],
                                                   padding=paddings[layer_num],
                                                   out_channels=out_channels[layer_num],
                                                   kernel_size=kernel_sizes[layer_num],
                                                   pool_size=pool_sizes[layer_num],
                                                   cnn_dropout_p=cnn_dropout_p)))
        # Change to sequential
        self.conv = nn.Sequential(*convs)

        # Output linear layer
        self.output_fc = None
        if fc_output_dim is not None:
            self.output_fc = nn.Sequential(nn.AdaptiveAvgPool3d((None, 1, 1)),
                                           nn.Flatten(start_dim=2),
                                           nn.Linear(out_channels[-1], fc_output_dim),
                                           nn.ReLU(inplace=True))

    def forward(self,
                x):
        """
        Forward path of the CNN3D network

        :param x: torch.tensor, input torch.tensor of image frames

        :return: Vector embeddings of input images of shape (num_samples, output_dim)
        """

        # CNN layers
        x = self.conv(x)

        # FC layer
        if self.output_fc is not None:
            x = self.output_fc(x)

        return x

