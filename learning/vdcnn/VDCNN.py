# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal, constant


def get_args():
    parser = argparse.ArgumentParser("""
    Very Deep CNN with optional residual connections (https://arxiv.org/abs/1606.01781)
    """)
    parser.add_argument("--dataset", type=str, default='imdb')
    parser.add_argument("--model_folder", type=str, default="models/VDCNN/imdb")
    parser.add_argument("--depth", type=int, choices=[9, 17, 29, 49], default=9,
                        help="Depth of the network tested in the paper (9, 17, 29, 49)")
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument('--shortcut', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle train and test sets")
    parser.add_argument("--chunk_size", type=int, default=2048, help="number of examples read from disk")
    parser.add_argument("--batch_size", type=int, default=128, help="number of example read by the gpu")
    parser.add_argument("--test_batch_size", type=int, default=512,
                        help="number of example read by the gpu during test time")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=100,
                        help="Number of iterations before halving learning rate")
    parser.add_argument("--class_weights", nargs='+', type=float, default=None)
    parser.add_argument("--test_interval", type=int, default=50, help="Number of iterations between testing phases")
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--last_pooling_layer", type=str, choices=['k-max-pooling', 'max-pooling'],
                        default='k-max-pooling', help="type of last pooling layer")

    args = parser.parse_args()
    return args


def predict_from_model(generator, model, gpu=True):
    model.eval()
    y_prob = []

    for data in generator:
        tdata = [Variable(torch.from_numpy(x).long(), volatile=True) for x in data]
        if gpu:
            tdata = [x.cuda() for x in tdata]

        yhat = model(tdata[0])

        # normalizing probs
        yhat = nn.functional.softmax(yhat)

        y_prob.append(yhat)

    y_prob = torch.cat(y_prob, 0)
    y_prob = y_prob.cpu().data.numpy()

    model.train()
    return y_prob


def batchify(arrays, batch_size=128):
    assert np.std([x.shape[0] for x in arrays]) == 0

    for j in range(0, len(arrays[0]), batch_size):
        yield [x[j: j + batch_size] for x in arrays]


class BasicConvResBlock(nn.Module):

    def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=False,
                 downsample=None):
        super(BasicConvResBlock, self).__init__()

        self.downsample = downsample
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(n_filters)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out


class VDCNN(nn.Module):

    def __init__(self, n_classes=2, input_dim=16, depth=9, n_fc_neurons=2048, shortcut=False):
        super(VDCNN, self).__init__()

        layers = []
        fc_layers = []

        layers.append(nn.Conv1d(input_dim, 64, kernel_size=3, padding=1))

        if depth == 9:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
        elif depth == 17:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
        elif depth == 29:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
        elif depth == 49:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3
        else:
            raise NotImplementedError()

        layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
        for _ in range(n_conv_block_64 - 1):
            layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # l = initial length / 2

        ds = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(128))
        layers.append(
            BasicConvResBlock(input_dim=64, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_128 - 1):
            layers.append(BasicConvResBlock(input_dim=128, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # l = initial length / 4

        ds = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
        layers.append(
            BasicConvResBlock(input_dim=128, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_256 - 1):
            layers.append(BasicConvResBlock(input_dim=256, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(512))
        layers.append(
            BasicConvResBlock(input_dim=256, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_512 - 1):
            layers.append(BasicConvResBlock(input_dim=512, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut))

        last_pooling_layer = 'k-max-pooling'

        if last_pooling_layer == 'k-max-pooling':
            layers.append(nn.AdaptiveMaxPool1d(8))
            fc_layers.extend([torch.nn.Dropout()])
            fc_layers.extend([nn.Linear(8 * 512, n_fc_neurons), nn.ReLU()])
        elif last_pooling_layer == 'max-pooling':
            layers.append(nn.MaxPool1d(kernel_size=8, stride=2, padding=0))
            fc_layers.extend([torch.nn.Dropout()])
            fc_layers.extend([nn.Linear(61 * 512, n_fc_neurons), nn.ReLU()])
        else:
            raise NotImplementedError()
        fc_layers.extend([torch.nn.Dropout()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([torch.nn.Dropout()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

        fc_layers2 = []

        if last_pooling_layer == 'k-max-pooling':
            layers.append(nn.AdaptiveMaxPool1d(8))
            fc_layers2.extend([torch.nn.Dropout()])
            fc_layers2.extend([nn.Linear(8 * 512, 256), nn.ReLU()])
        elif last_pooling_layer == 'max-pooling':
            layers.append(nn.MaxPool1d(kernel_size=8, stride=2, padding=0))
            fc_layers2.extend([torch.nn.Dropout()])
            fc_layers2.extend([nn.Linear(61 * 512, 256), nn.ReLU()])
        else:
            raise NotImplementedError()
        fc_layers2.extend([torch.nn.Dropout()])
        fc_layers2.extend([nn.Linear(256, 256)])
        fc_layers2.extend([torch.nn.Dropout()])
        fc_layers2.extend([nn.Linear(256, 1)])
        fc_layers2.extend([nn.Tanh()])
        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)

        self.fc_layers2 = nn.Sequential(*fc_layers2)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal(m.weight, mode='fan_in')
                if m.bias is not None:
                    constant(m.bias, 0)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        priors = self.fc_layers(out)
        scalar = self.fc_layers2(out)
        return priors, scalar
