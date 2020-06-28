import logging
import os

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from bilstm_crf_model.bilstm_crf import BiLSTM_CRF


os.environ['CUDA_VISIBLE_DEVICES'] = '6'
torch.set_num_threads(10)

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
        self.n_epochs = 2000  # 10000 for bilstm
        self.embedding_dim = 768
        self.hidden_dim = 600
        self.max_len = 100  # 50 bilstm
        self.start_tag = '<START>'
        self.end_tag = '<END>'
        self.lr = 0.001
        self.batch_size = 300  # 200 for bilstm
        self.num_workers = 4
        self.eval_freq = 5  #
        self.f1_conv = 0.01  #
        self.model_save_path = "../checkpoints/bert_bilstm_crf_{}.pt".format(self.n_epochs)
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
                       'o': 0,
                       self.start_tag: 10,
                       self.end_tag: 11}
        self.id2tag = {0: 'o',
                       1: 'B_ns',
                       2: 'B_nr',
                       3: 'B_nt',
                       4: 'M_nt',
                       5: 'M_nr',
                       6: 'M_ns',
                       7: 'E_nt',
                       8: 'E_nr',
                       9: 'E_ns',
                       10: self.start_tag,
                       11: self.end_tag}
        self.n_tags = len(self.id2tag)


class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(config.chinese_bert_path, output_hidden_states=False)
        self.bilstm_crf = BiLSTM_CRF(config)

    def nll(self, input_ids, att_masks, tag_ids, lengths):
        embeddings, _ = self.bert(input_ids, attention_mask=att_masks)
        # param outputs: [batch_size, seq_len, n_embedding]
        # neg_log_likelihood(self, X, tags, masks, length):
        loss = self.bilstm_crf.neg_log_likelihood(embeddings, tag_ids, att_masks, lengths)
        return loss

    def forward(self, input_ids, att_masks, lengths):
        # forward(self, X, masks, length):
        embeddings, _ = self.bert(input_ids, attention_mask=att_masks)
        scores, sequences = self.bilstm_crf(embeddings, att_masks, lengths)
        return scores, sequences
