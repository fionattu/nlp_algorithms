import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

sentences = ["i like dog", "i like cat", "i like animal",
             "dog cat animal", "apple cat dog like", "dog fish milk like",
             "dog cat eyes like", "i like apple", "apple i hate",
             "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join([sentences]).split()
word_list = list(set(word_sequence))
word_dict = {w: i for i, w in enumerate(word_list)}
print(word_dict)

# Creating n-grams with a sliding window, n=3
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    center_w = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
    for context_w in context:
        skip_grams.append([center_w, context_w])


def random_batch(n_grams, batch_size):
    batch_inputs = []
    batch_labels = []
    random_indices = np.random.choice(range(len(n_grams)), batch_size, replace=False)

    for i in random_indices:
        batch_inputs.append(np.eye(voc_size)[n_grams[i][0]])  # one-hot of center_w
        batch_labels.append(n_grams[i][1])  # index of context_w

    return batch_inputs, batch_labels


# Word2Vec Parameter
batch_size = 20  # To show 2 dim embedding graph
embedding_size = 2  # To show 2 dim embedding graph
voc_size = len(word_list)
dtype = torch.FloatTensor


# Model
class Word2Vec(nn.Module):

    def __init__(self):
        super(Word2Vec, self).__init__()

        # W for hidden layer, WT for output layer, not transpose relation
        self.W = nn.Parameter(-2 * torch.rand(voc_size, embedding_size) + 1).type(dtype)
        self.WT = nn.Parameter(-2 * torch.rand(embedding_size, voc_size) + 1).type(dtype)

    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = torch.matmul(X, self.W)  # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.WT)  # output_layer : [batch_size, voc_size]
        return output_layer


model = Word2Vec()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5000):
    batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

    batch_inputs = Variable(torch.Tensor(batch_inputs))
    batch_labels = Variable(torch.LongTensor(batch_labels))

    optimizer.zero_grad()
    output = model(batch_inputs)

    # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, batch_labels)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

