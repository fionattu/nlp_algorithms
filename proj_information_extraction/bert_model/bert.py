"""
BertForTokenClassification with the classifier layer fine-tuned
"""
import torch.nn as nn
from transformers import BertForTokenClassification, BertTokenizer
import logging
import torch


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class Config(object):
    def __init__(self):
        self.chinese_bert_path = "../pretrained_models/chinese_L-12_H-768_A-12/"
        self.tokenizer = BertTokenizer.from_pretrained(self.chinese_bert_path)
        self.train = ['../data/MSRA/train_inputs.txt', '../data/MSRA/train_labels.txt']
        self.valid = ['../data/MSRA/valid_inputs.txt', '../data/MSRA/valid_labels.txt']
        self.use_gpu = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fine_tune = True
        self.n_epochs = 20
        self.hidden_dim = 768
        self.max_len = 150
        self.lr = 0.001
        self.batch_size = 32
        self.num_workers = 4
        self.eval_freq = 1
        self.f1_conv = 0.01
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
        self.id2tag = {0: 'o',
                       1: 'B_ns',
                       2: 'B_nr',
                       3: 'B_nt',
                       4: 'M_nt',
                       5: 'M_nr',
                       6: 'M_ns',
                       7: 'E_nt',
                       8: 'E_nr',
                       9: 'E_ns'}
        self.n_tags = len(self.id2tag)
        self.model_save_path = "../checkpoints/bert_{}.pt".format(self.n_epochs)


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        # https://github.com/huggingface/transformers/issues/2110
        # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin
        # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json
        self.model = BertForTokenClassification.from_pretrained(config.chinese_bert_path, num_labels=config.n_tags)
        self.model.to(config.device)

    def forward(self, input_ids, tag_ids, att_masks):
        loss, outputs = self.model(input_ids, attention_mask=att_masks, labels=tag_ids)
        return loss, outputs
