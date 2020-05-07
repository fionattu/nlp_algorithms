import torch.nn as nn
import torch
from torch.autograd import Variable


class BiLSTM_CRF(nn.module):
    def __init__(self, n_voc, n_embedding, n_hidden, n_tags):
        super(BiLSTM_CRF, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        # self.n_voc = n_voc
        self.n_tag = n_tags
        self.embedings = nn.Embedding(n_voc, n_embedding)
        self.lstm = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, bidirectional=True)
        self.hidden2tag = nn.Linear(n_hidden * 2, n_tags, bias=True)
        self.transitions = nn.Parameter(n_tags, n_tags)

    def _init_embedding(self, embeddings):
        pass
    
    def _lstm_features(self, X): # X = [batch_size, seq_len, n_embedding]
        X = X.transpose(0, 1) # X = [seq_len, batch_size, n_embedding]
        hidden_state = Variable(torch.zeros(1 * 2, list(X.size())[1], self.n_hidden))
        cell_state = Variable(torch.zeros(1 * 2, list(X.size())[1], self.n_hidden))
        lstm_out, (_,_) = self.lstm(X, (hidden_state, cell_state))
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

