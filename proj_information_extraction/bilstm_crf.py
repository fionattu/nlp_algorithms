import torch.nn as nn
import torch
from torch.autograd import Variable


# START_TAG = '<START>'
# END_TAG = '<END>'


class BiLSTM_CRF(nn.Module):
    def __init__(self, n_embedding, n_hidden, tag2id, start_tag, end_tag):
        super(BiLSTM_CRF, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_tags = len(tag2id)
        self.tag2id = tag2id

        self.START_TAG = start_tag
        self.END_TAG = end_tag

        self.lstm = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, bidirectional=True)
        self.hidden2tag = nn.Linear(n_hidden * 2, self.n_tags, bias=True)

        self.transitions = nn.Parameter(torch.randn(self.n_tags, self.n_tags))
        self.transitions.data[:, tag2id[self.START_TAG]] = -10000
        self.transitions.data[tag2id[self.END_TAG], :] = -10000

    def _lstm_features(self, X):  # X =  consider padding input
        """Get lstm features after the fully-connected layer

        :param X: [batch_size, seq_len, n_embedding]
        :return: [seq_len, batch_size, n_tags]
        """
        X = X.transpose(0, 1)  # X = [seq_len, batch_size, n_embedding]
        hidden_state = Variable(torch.zeros(1 * 2, list(X.size())[1], self.n_hidden))
        cell_state = Variable(torch.zeros(1 * 2, list(X.size())[1], self.n_hidden))
        lstm_out, (_, _) = self.lstm(X, (hidden_state, cell_state))  # [seq_len, batch, num_directions * hidden_size]
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def _compute_sentence_score(self, emissions, tags, masks):
        """Compute score of real path

        :param emissions: [seq_len, batch_size, n_tags]
        :param tags: [batch_size, seq_len]
        :param masks: [batch_size, seq_len]
        :return: [batch_size]
        """
        emissions = emissions.transpose(0, 1)  # emissions = [batch_size, seq_len, n_tags]
        batch_size, seq_len = tags.shape
        scores = torch.zeros(batch_size)
        first_tags = tags[:, 0]
        t_scores = self.transitions[self.tag2id[self.START_TAG], first_tags]
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
        scores += self.transitions[last_tags, self.tag2id[self.END_TAG]]
        return scores

    def _compute_log_partition(self, emissions, masks):
        """Compute the log-scores of all paths

        :param emissions: [seq_len, batch_size, n_tags]
        :param masks: [batch_size, seq_len]
        :return: [batch_size]
        """
        seq_len, batch_size, n_tags = emissions.shape

        # [batch_size, seq_len, n_tags]
        emissions = emissions.transpose(0, 1)

        # [batch_size, n_tags]
        alphas = self.transitions[self.tag2id[self.START_TAG], :] + emissions[:, 0]

        for i in range(1, seq_len):
            # [batch_size, n_tags] -> [batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1)

            # [n_tags, n_tags] -> [1, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)

            # [batch_size, n_tags] -> [batch_size, 1, n_tags]
            a_scores = alphas.unsqueeze(1)

            # [batch_size, n_tags, n_tags]
            scores = e_scores + t_scores + a_scores

            # dim=1 to sum across rows of [n_tags, n_tags]
            # [batch_size, n_tags]
            new_alphas = torch.logsumexp(scores, dim=1)

            # [batch_size] -> [batch_size, 1]
            is_valid = masks[:, i].unsqueeze(1)

            # if pad, use old alphas
            # [batch_size, n_tags]
            alphas = new_alphas * is_valid + alphas * (1 - is_valid)

        alphas += self.transitions[:, self.tag2id[self.END_TAG]]

        return torch.logsumexp(alphas, dim=1)

    def neg_log_likelihood(self, X, tags, masks):
        """Return NLL as the loss"""
        emissions = self._lstm_features(X)
        sentence_score = self._compute_sentence_score(emissions, tags, masks)
        log_partition = self._compute_log_partition(emissions, masks)
        return log_partition - sentence_score




