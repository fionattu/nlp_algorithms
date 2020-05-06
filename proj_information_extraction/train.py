import pickle

pkl_fname = "data/msra_ner.pkl"
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

START_TAG = '<START>'
END_TAG = '<END>'
tag2id[START_TAG] = len(tag2id)
tag2id[END_TAG] = len(tag2id)
