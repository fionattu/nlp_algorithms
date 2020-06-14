import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Bert(nn.Module):
    def __init__(self, bert_model_dir, n_hidden, n_tags):
        self.model = BertModel.from_pretrained(bert_model_dir)
        # self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
        self.linear = nn.Linear(n_hidden, n_tags)

    def forward(self, X):
        # tokens = self.tokenizer(self.tokenizer.decode(self.tokenizer.encode(sentences)))
        # inputs = self.tokenizer.encode(sentences, return_tensors='pt')
        sentences, masks = X[0], X[1]
        last_hidden_states = self.model(sentences, attention_mask=masks, output_all_encoded_layers=False)[1]
