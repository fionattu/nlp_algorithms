import torch.nn as nn
import torch
from torch.autograd import Variable


START_TAG = '<START>'
END_TAG = '<END>'


class BiLSTM_CRF(nn.Module):
    def __init__(self, n_embedding, n_hidden, tag2id):
        super(BiLSTM_CRF, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_tags = len(tag2id)
        self.tag2id = tag2id

        self.lstm = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, bidirectional=True)
        self.hidden2tag = nn.Linear(n_hidden * 2, self.n_tags, bias=True)

        self.transitions = nn.Parameter(torch.randn(self.n_tags, self.n_tags))
        self.transitions.data[:, tag2id[START_TAG]] = -10000
        self.transitions.data[tag2id[END_TAG], :] = -10000

    def _lstm_features(self, X):  # X = [batch_size, seq_len, n_embedding] consider padding input
        X = X.transpose(0, 1)  # X = [seq_len, batch_size, n_embedding]
        hidden_state = Variable(torch.zeros(1 * 2, list(X.size())[1], self.n_hidden))
        cell_state = Variable(torch.zeros(1 * 2, list(X.size())[1], self.n_hidden))
        lstm_out, (_, _) = self.lstm(X, (hidden_state, cell_state))  # [seq_len, batch, num_directions * hidden_size]
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def _compute_sentence_score(self, emissions, tags, masks):
        # emissions = [seq_len, batch_size, n_tags] consider padding input
        # tags = [batch_size, seq_len]
        # masks = [batch_size]
        emissions = emissions.transpose(0, 1)  # emissions = [batch_size, seq_len, n_tags]
        batch_size, seq_len = tags.shape
        scores = torch.zeros(batch_size)
        first_tags = tags[:, 0]
        t_scores = self.transitions[self.tag2id[START_TAG], first_tags]
        # emissions[:, 0] = [batch_size, n_tags] pytorch remove 0 dimension automatically
        e_scores = (emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1))).squeeze()
        scores += (e_scores + t_scores)

        for i in range(1, seq_len):
            is_valid = masks[:, i]
            t_scores = self.transitions[tags[:, i - 1], tags[:, i]]
            e_scores = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze()
            scores += (e_scores + t_scores) * is_valid

        last_valid_idx = torch.sum(masks.type(torch.IntTensor), 1) - 1  # dim=1: sum across columns
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()
        scores += self.transitions[last_tags, self.tag2id[END_TAG]]
        return scores

    def neg_log_likelihood(self, X, tags, masks):
        emissions = self._lstm_features(X)
        sentence_score = self._compute_sentence_score(emissions, tags, masks)
        return 0




