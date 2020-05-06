import torch.nn as nn


class BiLSTM_CRF(nn.module):

    def __init__(self, a):
        super(BiLSTM_CRF, self).__init__()
