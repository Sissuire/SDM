import torch.nn as nn


def _conv_1d_layer_(in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
    return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)


def _conv_3d_layer_(in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=padding,
            stride=stride, bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.ELU(alpha=1.0, inplace=False))


def _conv_3d_layer_down_(in_channels, out_channels, kernel_size, padding, pool_size=(1, 2, 2), stride=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=padding,
            stride=stride, bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.ELU(alpha=1.0, inplace=False),
        nn.AvgPool3d(kernel_size=pool_size, stride=pool_size))


def _conv_2d_layer_down_(in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias),
        nn.BatchNorm2d(out_channels),
        # nn.ELU(alpha=1.0, inplace=False),
        nn.ELU(),
        nn.AvgPool2d(kernel_size=2, stride=2))


def _conv_2d_layer_(in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ELU()
        # nn.ELU(alpha=1.0, inplace=False),
    )


def _fc_layer_(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.Dropout(0.2),
        nn.ELU(alpha=1.0, inplace=False))
