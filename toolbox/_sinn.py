#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch import nn


class SINN(nn.Module):
    def __init__(self, observed_dimension, hidden_size, num_layers, output_size, dropout_p=0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=observed_dimension,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.readout = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, state=None):
        x, (h, c) = self.lstm(self.dropout(x), state)
        return self.readout(x), (h, c)
