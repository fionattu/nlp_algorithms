import pickle
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from bilstm_crf import BiLSTM_CRF

torch.manual_seed(1)
pkl_fname = "data/msra_ner.pkl"
# depends on memory
batch_size = 50
n_epoches = 2000
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

# oov use random embedding "for each" but not zeros
embeddings = np.random.normal(0, 0.1, (len(word2id), embedding_dim))
for word, id in word2id.items():
    if word in word2embeds:
        embeddings[id] = word2embeds[word]
    else:
        print('no embedding: {}， use random embedding'.format(word))

START_TAG = '<START>'
END_TAG = '<END>'
# PAD_TAG = '<PAD>'
tag2id[START_TAG] = len(tag2id)
tag2id[END_TAG] = len(tag2id)


def random_batch(embeddings, x_train, y_train, batch_size):
    batch_inputs = []
    batch_outputs = []
    masks = []
    length = []

    random_indices = np.random.choice(range(len(x_train)), batch_size, replace=False)
    random_indices = np.sort(random_indices)
    for i in random_indices:
        batch_inputs.append(embeddings[x_train[i]])
        masks.append(np.where(np.array(x_train[i]) > 0, 1, 0))
        length.append(masks[-1].sum())
        batch_outputs.append(y_train[i])

    return batch_inputs, batch_outputs, masks, length


# tag_length should not include start/end tags
model = BiLSTM_CRF(embedding_dim, hidden_dim, tag2id, START_TAG, END_TAG)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(n_epoches):
    # or model.zero_grad() since all model parameters are in optimizer
    optimizer.zero_grad()

    batch_inputs, batch_outputs, masks, length = random_batch(embeddings, x_train, y_train, batch_size)
    batch_inputs = Variable(torch.FloatTensor(batch_inputs))
    batch_output = Variable(torch.LongTensor(batch_outputs))
    masks = Variable(torch.IntTensor(masks))
    length = torch.LongTensor(length)

    loss = model.neg_log_likelihood(batch_inputs, batch_output, masks, length)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch: {:04d}, loss: {:.4f}'.format(epoch, loss))


# predictions after training
batch_inputs, batch_outputs, masks, length = random_batch(embeddings, x_train, y_train, batch_size)
batch_inputs = Variable(torch.FloatTensor(batch_inputs))
batch_output = Variable(torch.LongTensor(batch_outputs))
masks = Variable(torch.IntTensor(masks))
length = torch.LongTensor(length)
scores, sequences = model(batch_inputs, masks, length)

