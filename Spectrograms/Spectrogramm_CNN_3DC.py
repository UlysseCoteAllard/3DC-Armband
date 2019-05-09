import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, number_of_class):
        super(Net, self).__init__()
        self._conv1 = nn.Conv2d(4, 32, kernel_size=3)
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(.5)

        self._conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self._pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self._batch_norm2 = nn.BatchNorm2d(64)
        self._prelu2 = nn.PReLU(64)
        self._dropout2 = nn.Dropout2d(.5)

        self._conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self._batch_norm3 = nn.BatchNorm2d(128)
        self._prelu3 = nn.PReLU(128)
        self._dropout3 = nn.Dropout2d(.5)

        self._global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self._output = nn.Linear(128, number_of_class)

        self.initialize_weights()

        print(self)

        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        pool1 = self._pool1(conv1)
        conv2 = self._dropout2(self._prelu2(self._batch_norm2(self._conv2(pool1))))
        pool2 = self._pool2(conv2)
        conv3 = self._dropout3(self._prelu3(self._batch_norm3(self._conv3(pool2))))

        global_average_pool = self._global_average_pooling(conv3)
        flatten_tensor = global_average_pool.view(-1, 128)
        output = self._output(flatten_tensor)
        return output
