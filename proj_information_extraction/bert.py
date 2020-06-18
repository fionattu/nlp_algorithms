import torch.nn as nn
from transformers import BertModel, BertForTokenClassification


class Bert(nn.Module):
    def __init__(self, config):
        # self.model = BertModel.from_pretrained(bert_model_dir)
        # self.linear = nn.Linear(n_hidden, n_tags)
        self.model = BertForTokenClassification.from_pretrained(config.bert_model_dir, num_labels=config.n_tags)

    def forward(self, X):
        input_ids, tag_ids, att_masks = X[0], X[1], X[2]
        loss = self.model(input_ids, attention_mask=att_masks, labels=tag_ids)[0]
        return loss
