"""
    To implement code of your network using operation from ops.py.
"""

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchsummary import summary as torch_summary

from src.utils import load_class


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: Optional[Dict] = None,
    ) -> None:
        """[summary]

        Args:
            in_features (int): [description]
            out_features (int): [description]
            bias (bool, optional): [description]. Defaults to False.
            activation (Optional[Dict], optional): [description]. Defaults to None.
        """
        super(LinearBlock, self).__init__()

        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )

        self.activation = activation
        if self.activation:
            self.activation = getattr(nn, activation["type"])(**activation["args"])

    def forward(self, x):
        x = self.linear(x)

        if self.activation:
            x = self.activation(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation: Optional[Dict] = None,
        pool: Optional[Dict] = None,
    ) -> None:
        """[summary]

        Args:
            in_channels (int): [description]
            out_channels (int): [description]
            kernel_size (int, optional): [description]. Defaults to 3.
            stride (int, optional): [description]. Defaults to 1.
            padding (int, optional): [description]. Defaults to 0.
            dilation (int, optional): [description]. Defaults to 1.
            groups (int, optional): [description]. Defaults to 1.
            bias (bool, optional): [description]. Defaults to True.
            padding_mode (str, optional): [description]. Defaults to "zeros".
            batch_norm (bool, optional): [description]. Defaults to False.
            activation (Optional[Dict], optional): [description]. Defaults to None.
            pool (Optional[Dict], optional): [description]. Defaults to None.
        """
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.activation = activation
        if self.activation:
            self.activation = getattr(nn, activation["type"])(**activation["args"])

        self.pool = pool
        if self.pool:
            # yaml not supported tuple. omegaconf too
            pool_dict = dict(pool)

            kernel_size = tuple(list(pool.args.kernel_size))

            old_args = pool_dict.pop("args", None)
            new_args = {}
            for key in old_args.keys():
                if key == "kernel_size":
                    continue
                new_args.update({key: old_args[key]})
            new_args.update({"kernel_size": kernel_size})
            pool_dict.update({"args": new_args})

            self.pool = getattr(nn, pool_dict["type"])(**pool_dict["args"])

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)

        if self.activation:
            x = self.activation(x)

        if self.pool:
            x = self.pool(x)

        return x


def _build_linear_layers(linear_layers_config: DictConfig) -> torch.nn.ModuleList:
    return nn.ModuleList([LinearBlock(**params) for params in linear_layers_config])


def _build_conv_layers(conv_layers_config: DictConfig) -> torch.nn.ModuleList:
    return nn.ModuleList([ConvBlock(**params) for params in conv_layers_config])


def _build_output_layer(output_layer_config) -> torch.nn.Module:
    return load_class(
        module=nn, name=output_layer_config["type"], args=output_layer_config["args"]
    )


class LeNet(nn.Module):
    def __init__(self, model_config: DictConfig) -> None:
        """[summary]

        Args:
            model_config (DictConfig): [description]
        """
        super(LeNet, self).__init__()

        self._width: int = model_config.params.width
        self._height: int = model_config.params.height
        self._channels: int = model_config.params.channels

        self.input_shape: tuple = (self._channels, self._height, self._width)
        self.in_channels: int = self._channels

        self.conv_layers: nn.ModuleList = _build_conv_layers(
            conv_layers_config=model_config.params.feature_layers.conv
        )

        self.linear_layers: nn.ModuleList = _build_linear_layers(
            linear_layers_config=model_config.params.feature_layers.linear
        )

        self.output_layer = _build_output_layer(
            output_layer_config=model_config.params.output_layer
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.view(x.size()[0], -1)

        for linear_layer in self.linear_layers:
            x = linear_layer(x)

        x = self.output_layer(x)

        return x

    def loss(self, x, y):
        return self.loss_fn(x, y)

    def summary(self):
        # torchsummary only supported [cuda, cpu]. not cuda:0
        device = str(self.device).split(":")[0]
        torch_summary(
            self,
            input_size=(self._channels, self._height, self._width),
            device=device,
        )

    @property
    def device(self):
        devices = {param.device for param in self.parameters()} | {
            buf.device for buf in self.buffers()
        }
        if len(devices) != 1:
            raise RuntimeError(
                "Cannot determine device: {} different devices found".format(
                    len(devices)
                )
            )
        return next(iter(devices))
