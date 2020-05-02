import torch
import torch.nn as nn


class BiLSTM_CRF(nn.Module):
    def __init__(self):
        super(BiLSTM_CRF, self).__init__()

    def forward(self):
        