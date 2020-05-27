import pickle
import numpy as np
import torch
import math
import time
from torch import optim
from torch.autograd import Variable

from bilstm_crf import BiLSTM_CRF
from constants import Consts

torch.manual_seed(1)
pkl_fname = "data/msra_ner.pkl"
model_fname = "model/bilstmcrf_{}.pt".format(Consts.N_EPOCH)

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

    embeddings = pickle.load(f)

print("word2id size: {}, id2word size: {}".format(len(word2id), len(id2word)))
print("word 'pad' id: {}, 'unk' id: {}".format(word2id['pad'], word2id['unk']))
assert len(word2id) == len(id2word)
assert len(tag2id) == len(id2tag)
assert len(x_train) == len(y_train)
assert len(x_test) == len(y_test)
assert len(x_valid) == len(y_valid)


def random_batch(embeddings, x, y, batch_size, random=True):
    batch_inputs = []
    batch_ids = []
    batch_outputs = []
    masks = []
    length = []

    if random:
        random_indices = np.random.choice(range(len(x)), batch_size, replace=False)
        random_indices = np.sort(random_indices)
    else:
        random_indices = range(batch_size)

    for i in random_indices:
        batch_ids.append(x[i])
        batch_inputs.append(embeddings[x[i]])
        masks.append(np.where(np.array(x[i]) > 0, 1, 0))
        length.append(masks[-1].sum())
        batch_outputs.append(y[i])

    batch_inputs = Variable(torch.FloatTensor(batch_inputs))
    batch_outputs = Variable(torch.LongTensor(batch_outputs))
    masks = Variable(torch.IntTensor(masks))
    length = torch.LongTensor(length)

    return batch_ids, batch_inputs, batch_outputs, masks, length


def retrieve_entity(x, y, masks, id2tag, id2word):
    entity, result = [], []
    # batch
    for b in range(len(x)):
        # seq
        valid_len = masks[b].sum().item()
        for i in range(valid_len):
            word_idx, tag_idx = x[b][i], y[b][i]
            word, tag = id2word[word_idx], id2tag[tag_idx]
            if word_idx == 0 or tag_idx == 0:
                continue
            if tag[0] == 'B':
                entity = ['{}/{}'.format(word, tag)]
            elif tag[0] == 'M' and len(entity) > 0 and entity[-1].split('/')[1][1:] == tag[1:]:
                entity.append('{}/{}'.format(word, tag))
            elif tag[0] == 'E' and len(entity) > 0 and entity[-1].split('/')[1][1:] == tag[1:]:
                entity.append('{}/{}'.format(word, tag))
                entity.append(str(i))
                result.append(entity)
                entity = []
            else:
                if len(entity) == 1:
                    entity.append(str(i - 1))
                    result.append(entity)
                    entity = []
    return result


def get_f1(model, test=True):
    if test:
        model = BiLSTM_CRF(Consts.EMBEDDING_DIM, Consts.HIDDEN_DIM, tag2id, Consts.START_TAG, Consts.END_TAG)
        model.load_state_dict(torch.load(model_fname))
        model.eval()
        x, y = x_test, y_test
    else:
        x, y = x_valid, y_valid

    n_batch = math.ceil(len(x) / Consts.BATCH_SIZE)
    entity_pred, entity_true = [], []

    for i in range(n_batch):
        start = i * Consts.BATCH_SIZE
        end = (i + 1) * Consts.BATCH_SIZE if i != (n_batch - 1) else len(x)
        batch_ids, batch_inputs, batch_outputs, masks, length = random_batch(embeddings,
                                                                             x[start:end],
                                                                             y[start:end],
                                                                             end - start,
                                                                             False)
        scores, sequences = model(batch_inputs, masks, length)
        entity_pred += retrieve_entity(batch_ids, sequences, masks, id2tag, id2word)
        entity_true += retrieve_entity(batch_ids, batch_outputs.numpy(), masks, id2tag, id2word)

    union = [i for i in entity_pred if i in entity_true]
    precision = float(len(union)) / len(entity_pred)
    recall = float(len(union)) / len(entity_true)
    f1_score = 2 * precision * recall / (precision + recall) if len(union) != 0 else 0.0

    return entity_pred, f1_score, precision, recall


def train():
    # tag_length should not include start/end tags
    model = BiLSTM_CRF(Consts.EMBEDDING_DIM, Consts.HIDDEN_DIM, tag2id, Consts.START_TAG, Consts.END_TAG)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # f1 score of validation dataset
    valid_f1 = -1000
    stop = False
    start_t = time.time()
    for epoch in range(Consts.N_EPOCH):
        # or model.zero_grad() since all model parameters are in optimizer
        if stop:
            break
        optimizer.zero_grad()

        _, batch_inputs, batch_outputs, masks, length = random_batch(embeddings, x_train, y_train, Consts.BATCH_SIZE)

        loss = model.neg_log_likelihood(batch_inputs, batch_outputs, masks, length)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 300 == 0:
            print('Epoch: {:04d}, loss: {:.4f}, seconds: {:.4f}'.format(epoch, loss, time.time() - start_t))
            entities, new_valid_f1, prec, recall = get_f1(model, test=False)
            print('[Validation]f1 score from {:.6f} to {:.6f}'.format(valid_f1, new_valid_f1))
            print('[Validation]precision: {}, recall: {}\n'.format(prec, recall))
            if epoch > 3000 and (abs(new_valid_f1 - valid_f1) < 0.001 or new_valid_f1 < valid_f1):
                stop = True
            if new_valid_f1 > valid_f1:
                valid_f1 = new_valid_f1

    torch.save(model.state_dict(), model_fname)


if __name__ == '__main__':
    train()
    entities, new_valid_f1, prec, recall = get_f1("", test=True)
    print("[Test]f1 score: {:.6f}, precision: {}, recall: {}".format(new_valid_f1, prec, recall))

