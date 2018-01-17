import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='xavier'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class SkipLSTM(nn.Module):
    def __init__(self, input_size, output_size, opt):
        super(SkipLSTM, self).__init__()
        self.opt = opt

        self.lstm = nn.LSTM(
            input_size=input_size,
            num_layers=1,
            hidden_size=output_size,
            batch_first=True,
            bidirectional=True
        )

        init_weights(self, opt.init_type)

    def forward(self, x):
        tmpx = x
        x, _ = self.lstm(x)
        x = x.contiguous()
        x = torch.cat((x, tmpx), dim=2)

        return x

class TorchNetwork(nn.Module):
    def __init__(self, opt):
        super(TorchNetwork, self).__init__()
        self.opt = opt

        if opt.skip_lstm == 3:
            self.skipLSTM = nn.Sequential(
                SkipLSTM(opt.feature_size, opt.feature_size, opt),
                SkipLSTM(3 * opt.feature_size, opt.feature_size, opt),
                SkipLSTM(5 * opt.feature_size, opt.feature_size, opt),
            )
            output_size = 7 * opt.utterance_size * opt.feature_size
        elif opt.skip_lstm == 2:
            self.skipLSTM = nn.Sequential(
                SkipLSTM(opt.feature_size, opt.feature_size, opt),
                SkipLSTM(3 * opt.feature_size, opt.feature_size, opt),
            )
            output_size = 5 * opt.utterance_size * opt.feature_size
        elif opt.skip_lstm == 1:
            self.skipLSTM = nn.Sequential(
                SkipLSTM(opt.feature_size, opt.feature_size, opt),
            )
            output_size = 3 * opt.utterance_size * opt.feature_size
        else:
            output_size = opt.utterance_size * opt.feature_size

        self.output_size = output_size
        self.out_layer = nn.Sequential(
                nn.Linear(
                    in_features=output_size,
                    out_features=opt.fc_hidden_size
                ),
                nn.Linear(
                    in_features=opt.fc_hidden_size,
                    out_features=opt.class_size
                ),
            )

        init_weights(self, opt.init_type)

    def forward(self, x):
        opt = self.opt
        if opt.skip_lstm != 0:
            x = self.skipLSTM(x)

        x = x.view(-1, self.output_size)

        x = self.out_layer(x)

        return x
