# -*-coding:utf-8 -*-
import re
import pickle
import random
import numpy as np
from constants import Consts

msra_dpath = "../dataset/MSRA/"
train_fname = "train1.txt"
word2tag_fname = "word2tag.txt"
pkl_fname = "msra_ner.pkl"
embedding_fname = '../dataset/vec.txt'

# use BME to tag character
# nr/人名 ns/处所 nt/团体
line_cnt = 0
with open(word2tag_fname, 'w') as out:
    with open(msra_dpath + train_fname, 'r') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                w, label = word.split('/')
                w = w.strip()
                if label == 'o' and len(w) > 0:
                    for cha in w:
                        out.write("{}/{} ".format(cha, label))
                else:
                    if len(w) == 1:
                        out.write("{}/B_{} ".format(w, label))
                    elif len(w) > 1:
                        for cha in w:
                            if cha == w[0]:
                                out.write("{}/B_{} ".format(cha, label))
                            elif cha == w[-1]:
                                out.write("{}/E_{} ".format(cha, label))
                            else:
                                out.write("{}/M_{} ".format(cha, label))
            out.write('\n')
            line_cnt += 1
            if line_cnt % 1000 == 0:
                print("finish processing line: {}".format(line_cnt))
    f.close()
out.close()

tag2id = {'': 0,
          'B_ns': 1,
          'B_nr': 2,
          'B_nt': 3,
          'M_nt': 4,
          'M_nr': 5,
          'M_ns': 6,
          'E_nt': 7,
          'E_nr': 8,
          'E_ns': 9,
          'o': 0}

id2tag = {0: '',
          1: 'B_ns',
          2: 'B_nr',
          3: 'B_nt',
          4: 'M_nt',
          5: 'M_nr',
          6: 'M_ns',
          7: 'E_nt',
          8: 'E_nr',
          9: 'E_ns',
          10: 'o'}

inputs_dict = {}

# tag to id
with open('word2tag.txt', 'r') as f:
    for line in f:
        line = re.split('[，。；！：？、‘’“”]/[o]', line.strip())
        for sentence in line:
            sen_input, sen_label = [], []
            words = sentence.strip().split()
            has_ner = False
            for w in words:
                word, label = w.split('/')
                sen_input.append(word)
                sen_label.append(tag2id[label])
                if not has_ner and label != 'o':
                    has_ner = True
            # only train sentence with named entity
            if has_ner:
                inputs_dict[' '.join(sen_input)] = sen_label
                # inputs.append(sen_input)
                # labels.append(sen_label)
f.close()
print("finished converting tag to id")

# word to id
word_corpus = list(set([word for sentence in inputs_dict for word in sentence]))
word2id = {w: i + 1 for i, w in enumerate(word_corpus)}
word2id['unk'] = len(word2id) + 1
word2id['pad'] = 0
id2word = {i + 1: w for i, w in enumerate(word_corpus)}
id2word[len(id2word) + 1] = 'unk'
id2word[0] = 'pad'

tag2id[Consts.START_TAG] = len(tag2id)
tag2id[Consts.END_TAG] = len(tag2id)
id2tag[len(id2tag)] = Consts.START_TAG
id2tag[len(id2tag)] = Consts.END_TAG


def sentence_padding(sentence):
    """把 sentences 转为 word_ids 形式，并自动补全位 Consts.max_len 长度。"""
    ids = [word2id[w] for w in sentence]
    if len(ids) >= Consts.MAX_LEN:  # 长则弃掉
        return ids[:Consts.MAX_LEN]
    ids.extend([0] * (Consts.MAX_LEN - len(ids)))  # 短则补全
    return ids


def tag_padding(tags):
    """把 tags 转为 id 形式， 并自动补全位 Consts.max_len 长度。"""
    if len(tags) >= Consts.MAX_LEN:  # 长则弃掉
        return tags[:Consts.MAX_LEN]
    tags.extend([0] * (Consts.MAX_LEN - len(tags)))  # 短则补全
    return tags


def train_test_split(inputs, labels, test_percentage):
    test_size = int(len(inputs) * test_percentage)
    test_index = [random.randint(0, test_size) for _ in range(test_size)]
    x_train, x_test, y_train, y_test = [], [], [], []
    for i, v in enumerate(inputs):
        if i in test_index:
            x_test.append(v)
            y_test.append(labels[i])
        else:
            x_train.append(v)
            y_train.append(labels[i])
    return x_train, x_test, y_train, y_test


inputs_keys = list(inputs_dict.keys())
inputs_keys.sort(key=len, reverse=True)
sorted_labels = [inputs_dict[x] for x in inputs_keys]
sorted_inputs = [x.split() for x in inputs_keys]
inputs = [sentence_padding(sentence) for sentence in sorted_inputs]
labels = [tag_padding(label) for label in sorted_labels]
print("finished converting word to id")

# split train, test, validation
x_train, x_test, y_train, y_test = train_test_split(inputs, labels, 0.2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, 0.1)
print("finished splitting train, test, validation")

# load char-embedding
word2embeds = {}
with open(embedding_fname, 'r') as f:
    for line in f:
        char = line[0]
        embedding = line.replace(line[0], '').strip().split()
        word2embeds[char] = embedding
f.close()

# build word2vec
# oov use random embedding "for each" but not zeros
embedding_dim = 100
embeddings = np.random.normal(0, 0.1, (len(word2id), embedding_dim))
for word, id in word2id.items():
    if word in word2embeds:
        embeddings[id] = word2embeds[word]
    else:
        print('no embedding: {}， use random embedding'.format(word))

# pickle serialization use binary protocol by default
with open(pkl_fname, 'wb') as outp:
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
    pickle.dump(x_train, outp)
    pickle.dump(y_train, outp)
    pickle.dump(x_test, outp)
    pickle.dump(y_test, outp)
    pickle.dump(x_valid, outp)
    pickle.dump(y_valid, outp)
    pickle.dump(embeddings, outp)
print("finished saving to {}".format(pkl_fname))