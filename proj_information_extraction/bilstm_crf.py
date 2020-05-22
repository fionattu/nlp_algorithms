import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM_CRF(nn.Module):
    def __init__(self, n_embedding, n_hidden, tag2id, start_tag, end_tag):
        super(BiLSTM_CRF, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_tags = len(tag2id)
        self.tag2id = tag2id

        self.START_TAG = start_tag
        self.END_TAG = end_tag

        self.lstm = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(n_hidden * 2, self.n_tags, bias=True)

        self.transitions = nn.Parameter(torch.randn(self.n_tags, self.n_tags))
        self.transitions.data[:, tag2id[self.START_TAG]] = -10000
        self.transitions.data[tag2id[self.END_TAG], :] = -10000

    def _lstm_features(self, X, length):
        """Get lstm features after the fully-connected layer

        :param X: [batch_size, seq_len, n_embedding]
        :param length: [batch_size]
        :return: [seq_len, batch_size, n_tags]
        """

        batch_size = X.shape[0]

        # [seq_len, batch_size, n_embedding]
        X = X.transpose(0, 1)
        X = pack_padded_sequence(X, length)

        hidden_state = Variable(torch.zeros(1 * 2, batch_size, self.n_hidden))
        cell_state = Variable(torch.zeros(1 * 2, batch_size, self.n_hidden))

        # [seq_len, batch, num_directions * hidden_size]
        lstm_out, (_, _) = self.lstm(X, (hidden_state, cell_state))
        outputs, length = pad_packed_sequence(lstm_out)
        emissions = self.hidden2tag(outputs)
        return emissions

    def _compute_sentence_score(self, emissions, tags, masks):
        """Compute score of real path

        :param emissions: [seq_len, batch_size, n_tags]
        :param tags: [batch_size, seq_len]
        :param masks: [batch_size, seq_len]
        :return: [batch_size]
        """
        # emissions = [batch_size, seq_len, n_tags]
        emissions = emissions.transpose(0, 1)
        batch_size, seq_len, _ = emissions.shape
        scores = torch.zeros(batch_size)
        first_tags = tags[:, 0]
        t_scores = self.transitions[self.tag2id[self.START_TAG], first_tags]
        # emissions[:, 0] = [batch_size, n_tags] as pytorch remove 1-dimension automatically
        e_scores = (emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1))).squeeze()
        scores += (e_scores + t_scores)

        for i in range(1, seq_len):
            is_valid = masks[:, i]
            t_scores = self.transitions[tags[:, i - 1], tags[:, i]]
            e_scores = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze()
            scores += (e_scores + t_scores) * is_valid

        # dim=1: sum across columns
        last_valid_idx = torch.sum(masks.type(torch.IntTensor), 1) - 1
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

        scores = alphas + self.transitions[:, self.tag2id[self.END_TAG]]

        return torch.logsumexp(scores, dim=1)

    def neg_log_likelihood(self, X, tags, masks, length):
        """Return NLL as the loss"""
        emissions = self._lstm_features(X, length)
        sentence_score = self._compute_sentence_score(emissions, tags, masks)
        log_partition = self._compute_log_partition(emissions, masks)
        return torch.sum(log_partition - sentence_score)  # loss can only be one-dim tensor

    def _find_best_sequence(self, batch, back_track, end_tag):
        """
        :param batch: batch_index
        :param back_track: list of length seq_len, each of [batch_size, n_tags]
        :param end_tag: best end tag of current batch
        :return: sequence of best tags
        """
        seq_len = len(back_track)

        best_sequence = [end_tag]
        prev_tag = end_tag
        for i in reversed(range(1, seq_len)):
            prev_tag = back_track[i][batch, prev_tag].item()
            best_sequence.append(prev_tag)

        best_sequence.reverse()
        return best_sequence

    def _viterbi_decode(self, emissions, masks):
        """
        :param emissions: [seq_len, batch_size, nb_labels]
        :param masks: [batch_size, seq_len]
        :return:
        """
        seq_len, batch_size, n_tags = emissions.shape
        emissions = emissions.transpose(0, 1)

        # [batch_size, n_tags]
        alphas = emissions[:, 0] + self.transitions[self.tag2id[self.START_TAG], :]
        back_track_tags = []

        for i in range(1, seq_len):
            e_scores = emissions[:, i].unsqueeze(1)
            t_scores = self.transitions.unsqueeze(0)
            a_scores = alphas.unsqueeze(2)

            scores = e_scores + t_scores + a_scores

            # max_scores = [batch_size, n_tags]
            # max_scores_tags = [batch_size, n_tags]
            # max_score_tags find the best tags of last time step given the tags of current step
            max_scores, max_score_tags = torch.max(scores, dim=1)

            is_valid = masks[:, i].unsqueeze(1)
            alphas = max_scores * is_valid + alphas * (1 - is_valid)
            back_track_tags.append(max_score_tags)

        scores = alphas + self.transitions[:, self.tag2id[self.END_TAG]]

        # max_end_scores = [batch_size]
        # max_scores_tags = [batch_size]
        max_end_scores, max_end_score_tags = torch.max(scores, dim=1)

        best_sequences = []

        # [batch_size]
        valid_end_index = torch.sum(masks, dim=1) - 1
        for i in range(batch_size):
            valid_len_val = valid_end_index[i].item()

            # [valid_len_val, batch_size, n_tags]
            back_tracks = back_track_tags[:valid_len_val]

            end_tag = max_end_score_tags[i].item()

            best_sequence = self._find_best_sequence(i, back_tracks, end_tag)

            best_sequences.append(best_sequence)

        return max_end_scores, best_sequences

    def forward(self, X, masks, length):
        """Predict the best sequence after training"""

        # [seq_len, batch_size, nb_labels]
        emissions = self._lstm_features(X, length)
        scores, sequences = self._viterbi_decode(emissions, masks)
        return scores, sequences
