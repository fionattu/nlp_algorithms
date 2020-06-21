import torch
from transformers import BertTokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


class Config(object):
    def __init__(self):
        self.chinese_bert_path = "bert/chinese_L-12_H-768_A-12/"
        self.tokenizer = BertTokenizer.from_pretrained(self.chinese_bert_path)
        self.train_path = "data/word2tag.txt"
        self.use_gpu = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_tags = 11
        self.n_epochs = 1
        self.hidden_dim = 768
        self.max_len = 200
        self.tag2id = {'': 0,
                       'B_ns': 1,
                       'B_nr': 2,
                       'B_nt': 3,
                       'M_nt': 4,
                       'M_nr': 5,
                       'M_ns': 6,
                       'E_nt': 7,
                       'E_nr': 8,
                       'E_ns': 9,
                       'o': 0}
        self.lr = 0.001
        self.batch_size = 16
