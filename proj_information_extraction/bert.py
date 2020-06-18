import torch.nn as nn
from transformers import BertModel, BertForTokenClassification
import logging
logging.basicConfig(level=logging.INFO)


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        # self.linear = nn.Linear(n_hidden, n_tags)
        # https://github.com/huggingface/transformers/issues/2110
        # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin
        # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json
        self.model = BertForTokenClassification.from_pretrained(config.chinese_bert_path, num_labels=config.n_tags)
        print()

    def forward(self, input_ids, tag_ids, att_masks):
        loss = self.model(input_ids, attention_mask=att_masks, labels=tag_ids)[0]
        return loss
