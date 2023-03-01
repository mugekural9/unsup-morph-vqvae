"""Proposed residual neural nets architectures suited for MNIST"""

from typing import List

import torch
import torch.nn as nn

from ....base import BaseAEConfig
from ....base.base_utils import ModelOutput
from ...base_architectures import BaseDecoder, BaseEncoder
from ..utils import ResBlock


class Encoder_LSTM_VQVAE(BaseEncoder):
    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = 128
        self.enc_nh = 512
        self.num_layers = 1
        self.vocabsize = 50 #?
        self.enc_dropout_in = 0.2
        self.latent_dim = args.latent_dim
        self.n_channels = 1
        self.bidirectional = True
    

        layers = nn.ModuleList()
        layers.append(nn.Dropout(self.enc_dropout_in))
        layers.append(nn.Embedding(self.vocabsize, self.input_dim))
        layers.append(nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.enc_nh,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional))

        self.layers = layers
        self.depth = len(layers)


    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""

        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.pre_qantized(out)

        return output



class Decoder_ResNet_VQVAE_MNIST(BaseDecoder):
    """
    A ResNet decoder suited for MNIST and Vector Quantized VAE models.

    .. code-block::

        >>> from pythae.models.nn.benchmarks.mnist import Decoder_ResNet_VQVAE_MNIST
        >>> from pythae.models import VQVAEConfig
        >>> model_config = VQVAEConfig(input_dim=(1, 28, 28), latent_dim=16)
        >>> decoder = Decoder_ResNet_VQVAE_MNIST(model_config)
        >>> decoder
        ... Decoder_ResNet_VQVAE_MNIST(
        ...   (layers): ModuleList(
        ...     (0): ConvTranspose2d(16, 128, kernel_size=(1, 1), stride=(1, 1))
        ...     (1): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     (2): Sequential(
        ...       (0): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (1): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (2): ReLU()
        ...     )
        ...     (3): Sequential(
        ...       (0): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        ...       (1): ReLU()
        ...     )
        ...     (4): Sequential(
        ...       (0): ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        ...       (1): Sigmoid()
        ...     )
        ...   )
        ... )

    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VQVAE
        >>> model = VQVAE(model_config=model_config, decoder=decoder)
        >>> model.decoder == decoder
        ... True

    .. note::

        Please note that this decoder is suitable for **all** models.

        .. code-block::

            >>> import torch
            >>> input = torch.randn(2, 16, 4, 4)
            >>> out = decoder(input)
            >>> out.reconstruction.shape
            ... torch.Size([2, 1, 28, 28])
    """

    def __init__(self, args: BaseAEConfig):
        BaseDecoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(nn.ConvTranspose2d(self.latent_dim, 128, 1, 1))

        layers.append(nn.ConvTranspose2d(128, 128, 3, 2, padding=1))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    64, self.n_channels, 3, 2, padding=1, output_padding=1
                ),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output
