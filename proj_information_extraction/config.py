import torch


class Config(object):
    def __init__(self):
        self.bert_model_dir = "bert/chinese_L-12_H-768_A-12/"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

