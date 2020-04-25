import os
import json
import re
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors


def read_file(path):
    with open(path, 'r') as f:
        text = f.read()
    return text


def process_text(text):
    text = text.lower()
    text = re.sub('[^a-z^0-9^\u4e00-\u9fa5]', '', text)
    text = re.sub('[0-9]', '0', text)
    return text


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        self.epoch += 1


sentences = []
json_data_path = "/Users/fiona/Desktop/Github/IE_week2/dl_information_extract/dataset/json/"
corpus_path = "/Users/fiona/Desktop/Github/IE_week2/dl_information_extract/dataset/w2v/corpus_1.txt"
embedding_path = "/Users/fiona/Desktop/Github/IE_week2/dl_information_extract/dataset/w2v/sg_ns_100.txt"
# pretrained chinese vectors
key_vectors_path = "/Users/fiona/Desktop/Github/IE_week2/dl_information_extract/dataset/w2v/vec.txt"
json_files = [os.path.join(json_data_path, file) for file in os.listdir(json_data_path)]
for file in json_files:
    text = read_file(file)
    contexts = json.loads(text)
    for context in contexts:
        text = process_text(context['text'])
        sentences.append(' '.join(text))
print(len(sentences), sentences[0])

with open(corpus_path, 'w') as f:
    f.write('\n'.join(sentences))

with open(corpus_path, 'r') as f:
    sentences = [line.split(' ') for line in f.read().split('\n')]
epoch_logger = EpochLogger()
model = Word2Vec(sentences=sentences, min_count=1, size=100, sg=1, iter=3, callbacks=[epoch_logger])
model.wv.save_word2vec_format(embedding_path, binary=False)

for word in ['病', '痛', '肚']:
    sim_words = model.wv.most_similar(word)
    print(word, sim_words)

word_vectors = KeyedVectors.load_word2vec_format(key_vectors_path, binary=False)
for word in ['病', '痛', '肚']:
    sim_words_pretained = word_vectors.most_similar(word)
    print(word, sim_words_pretained)