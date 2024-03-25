"""
EECS 445 - Introduction to Machine Learning
Winter 2024 - Project 2

Challenge_Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge_source import Source
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class challenge_source(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        # TODO: define each layer
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5,5), stride=(2,2), padding=2)
        self.n1 = nn.BatchNorm2d(16) # normalize
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(5,5), stride=(2,2), padding=2)
        self.n2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 8, kernel_size=(5,5), stride=(2,2), padding=2)
        self.n3 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8*2*2,8)
        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.15)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""

        torch.manual_seed(445)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc_1]
        C_in_fc = self.fc1.weight.size(1)
        nn.init.normal_(self.fc1.weight, 0.0, 1 / sqrt(C_in_fc))
        nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """
        N, C, H, W = x.shape
        ## TODO: forward pass
        z1 = self.n1(self.conv1(x))
        h1 = F.relu(z1)
        h1 = self.d1(h1)
        p2 = self.pool(h1)
        z3 = self.n2(self.conv2(p2))
        h3 = F.relu(z3)
        h3 = self.d2(h3)
        p4 = self.pool(h3)
        z5 = self.n3(self.conv3(p4))
        h5 = F.relu(z5)
        h5 = self.d2(h5)
        z6 = h5.reshape(N, 32)
        z6 = self.fc1(z6)
        return z6