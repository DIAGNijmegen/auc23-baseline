import torch
import torch.nn as nn


class DepthWiseConv(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.depth_wise = Conv(dim, in_channels, in_channels, kernel_size, groups=in_channels, **kwargs)
        self.point_wise = Conv(dim, in_channels, out_channels, 1, groups=in_channels, **kwargs)

    def forward(self, x):
        x = self.depth_wise(x)
        return self.point_wise(x)


class Conv(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, **kwargs):
        super(Conv, self).__init__()
        if dim == 1:
            self.model = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        elif dim == 2:
            self.model = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        elif dim == 3:
            self.model = nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
        else:
            raise NotImplementedError('Not implemented for ' + str(dim) + 'D data.')

    def forward(self, x):
        return self.model(x)


class BatchNorm(nn.Module):
    def __init__(self, dim, num_features, no_bn, **kwargs):
        super(BatchNorm, self).__init__()
        if no_bn:
            self.model = nn.Sequential()
        elif dim == 1:
            self.model = nn.BatchNorm1d(num_features, **kwargs)
        elif dim == 2:
            self.model = nn.BatchNorm2d(num_features, **kwargs)
        elif dim == 3:
            self.model = nn.BatchNorm3d(num_features, **kwargs)
        else:
            raise NotImplementedError('Not implemented for ' + str(dim) + 'D data.')

    def forward(self, x):
        return self.model(x)


class AvgPool(nn.Module):
    def __init__(self, dim, kernel_size, **kwargs):
        super(AvgPool, self).__init__()
        if dim == 1:
            self.model = nn.AvgPool1d(kernel_size, **kwargs)
        elif dim == 2:
            self.model = nn.AvgPool2d(kernel_size, **kwargs)
        elif dim == 3:
            self.model = nn.AvgPool3d(kernel_size, **kwargs)
        else:
            raise NotImplementedError('Not implemented for ' + str(dim) + 'D data.')

    def forward(self, x):
        return self.model(x)


class GlobalSumPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x.view(x.size(0), x.size(1), -1), dim=2)


class GlobalRunningAvgPool(nn.Module):
    def __init__(self, momentum=0.99):
        super().__init__()
        self.register_buffer('running_mean', torch.tensor(0.))
        self.momentum = momentum

    def update_mean_nr_features(self, numel):
        if self.running_mean == 0:
            self.running_mean += numel
        else:
            self.running_mean = self.running_mean*self.momentum + numel*(1-self.momentum)

    def forward(self, x):
        self.update_mean_nr_features(x.shape.numel())
        x = torch.sum(x.view(x.shape[0], x.shape[1], -1), dim=2)
        x = x / self.running_mean
        return x


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)


class AdaptiveAvgPool(nn.Module):
    def __init__(self, dim, output_size, **kwargs):
        super(AdaptiveAvgPool, self).__init__()
        if dim == 1:
            self.model = nn.AdaptiveAvgPool1d(output_size, **kwargs)
        elif dim == 2:
            self.model = nn.AdaptiveAvgPool2d(output_size, **kwargs)
        elif dim == 3:
            self.model = nn.AdaptiveAvgPool3d(output_size, **kwargs)
        else:
            raise NotImplementedError('Not implemented for ' + str(dim) + 'D data.')

    def forward(self, x):
        return self.model(x)


class MaxPool(nn.Module):
    def __init__(self, dim, kernel_size, **kwargs):
        super(MaxPool, self).__init__()
        if dim == 1:
            self.model = nn.MaxPool1d(kernel_size, **kwargs)
        elif dim == 2:
            self.model = nn.MaxPool2d(kernel_size, **kwargs)
        elif dim == 3:
            self.model = nn.MaxPool3d(kernel_size, **kwargs)
        else:
            raise NotImplementedError('Not implemented for ' + str(dim) + 'D data.')

    def forward(self, x):
        return self.model(x)


class EmulateNonPaddedConvWithK3(nn.Module):
    def __init__(self, dim, kernel3):
        super(EmulateNonPaddedConvWithK3, self).__init__()
        if dim not in [1, 2, 3]:
            raise RuntimeError("Only implemented for dim = 1, 2, or 3")
        self.dim = dim
        self.kernel3 = kernel3

    def forward(self, x):
        if self.dim == 1:
            return x[:, :, self.kernel3:-self.kernel3]
        if self.dim == 2:
            return x[:, :, self.kernel3:-self.kernel3, self.kernel3:-self.kernel3]
        # here dim == 3
        return x[:, :, self.kernel3:-self.kernel3, self.kernel3:-self.kernel3, self.kernel3:-self.kernel3]