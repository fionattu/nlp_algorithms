import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

sentence = (
    'Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation'
)

voc = list(set(sentence.split()))
voc.append('pad')
word_dict = {w: i for i, w in enumerate(voc)}
idx_word_dict = {i: w for i, w in enumerate(voc)}
voc_size = len(voc)
voc_class = voc_size - 1
max_len = len(sentence.split())
n_hidden = 5
dtype = torch.FloatTensor


def create_all_batches(sentence):
    input_batch = []
    target_batch = []
    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_dict[w] for w in words[:(i + 1)]]
        input = input + [word_dict['pad']] * (max_len - len(input))
        target = word_dict[words[i + 1]]
        input_batch.append(np.eye(voc_size)[input])
        target_batch.append(target)
    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=voc_size, hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Parameter(torch.randn(n_hidden * 2, voc_size).type(dtype))
        self.b = nn.Parameter(torch.randn(voc_size).type(dtype))

    def forward(self, X): # X=[batch_size, seq_len, input_size]
        X = X.transpose(0, 1) # [seq_len, batch_size, input_size]

        hidden_state = Variable(torch.zeros(1 * 2, list(X.size())[1], n_hidden)) # [n_layers(=1) * n_directions(=2), batch_size, hidden_size]
        cell_state = Variable(torch.zeros(1 * 2, list(X.size())[1], n_hidden)) # [n_layers(=1) * n_directions(=2), batch_size, hidden_size]

        output, (_, _) = self.lstm(X, (hidden_state, cell_state)) # output=[seq_len, batch_size, hidden_size * 2]
        output = output[-1] # get the last word hidden state, [batch_size, hidden_size * 2]
        return torch.mm(output, self.W) + self.b


input_batch, target_batch = create_all_batches(sentence)
model = BiLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 100 == 0:
        print('Epoch: ', '%04d' % (epoch+1), 'cost = ', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()