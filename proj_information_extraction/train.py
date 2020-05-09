import pickle
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from bilstm_crf import BiLSTM_CRF

torch.manual_seed(1)
pkl_fname = "data/msra_ner.pkl"
batch_size = 10  # depends on memory
n_epoches = 100
embedding_dim = 100
hidden_dim = 5
dtype = torch.FloatTensor

with open(pkl_fname, 'rb') as f:
    word2id = pickle.load(f)
    id2word = pickle.load(f)

    tag2id = pickle.load(f)
    id2tag = pickle.load(f)

    x_train = pickle.load(f)
    y_train = pickle.load(f)

    x_test = pickle.load(f)
    y_test = pickle.load(f)

    x_valid = pickle.load(f)
    y_valid = pickle.load(f)

print("word2id size: {}, id2word size: {}".format(len(word2id), len(id2word)))
print("word 'pad' id: {}, 'unk' id: {}".format(word2id['pad'], word2id['unk']))
assert len(word2id) == len(id2word)
assert len(tag2id) == len(id2tag)
assert len(x_train) == len(y_train)
assert len(x_test) == len(y_test)
assert len(x_valid) == len(y_valid)

word2embeds = {}
with open('dataset/vec.txt', 'r') as f:
    for line in f:
        char = line[0]
        embedding = line.replace(line[0], '').strip().split()
        word2embeds[char] = embedding
f.close()


embeddings = np.zeros((len(word2id), embedding_dim))
for word, id in word2id.items():
    if word in word2embeds:
        embeddings[id] = word2embeds[word]
    else:
        print('no embedding: {}'.format(word))


START_TAG = '<START>'
END_TAG = '<END>'
tag2id[START_TAG] = len(tag2id)
tag2id[END_TAG] = len(tag2id)


def random_batch(embeddings, x_train, y_train, batch_size):
    batch_inputs = []
    batch_output = []
    random_indices = np.random.choice(range(len(x_train)), batch_size, replace=False)

    for i in random_indices:
        batch_inputs.append(embeddings[x_train[i]])
        batch_output.append(y_train[i])

    return batch_inputs, batch_output


model = BiLSTM_CRF(embedding_dim, hidden_dim, len(id2tag))  # tag_length should not include start/end tags
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(n_epoches):
    optimizer.zero_grad()

    batch_inputs, batch_outputs = random_batch(embeddings, x_train, y_train, batch_size)
    batch_inputs = Variable(torch.FloatTensor(batch_inputs))
    batch_output = Variable(torch.Tensor(batch_outputs))

    outputs = model.lstm_features(batch_inputs)
    loss = criterion(outputs, batch_output)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch: {:04d}'.format(epoch))