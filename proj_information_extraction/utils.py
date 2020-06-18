import re
import torch

from bert import Bert
from config import Config
import numpy as np
import torch.optim as optim

from transformers import BertTokenizer


class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.chinese_bert_path)

    def load_sentences_tags(self):
        sentences, tags = [], []
        with open(self.config.data_path, 'r') as data:
            for line in data:
                line = re.split('[，。；！：？、‘’“”]/[o]', line.strip())
                for sentence in line:
                    sen_input, sen_label = [], []
                    words = sentence.strip().split()
                    has_ner = False
                    for w in words:
                        word, label = w.split('/')
                        sen_input.append(word)
                        sen_label.append(label)
                        if not has_ner and label != 'o':
                            has_ner = True
                    # only train sentences with named entities
                    if has_ner:
                        sentences.append(sen_input)
                        tags.append(sen_label)
        return sentences, tags

    def get_batch_iterator(self):

        sentences, tags = self.load_sentences_tags()
        pad_id = self.config.tokenizer.convert_tokens_to_ids('[PAD]')
        n_batch = len(sentences) // self.config.batch_size

        for b in range(n_batch):
            start = b * self.config.batch_size
            end = (b + 1) * self.config.batch_size if b != (n_batch - 1) else len(sentences)

            # max len of sentence in the batch
            max_len = min(self.config.max_len, max([len(s) for s in sentences[start:end]]))

            # init inputs, tags, attention_masks in a batch
            shape = (end - start, max_len)
            input_ids, tag_ids, att_masks = pad_id * np.ones(shape), np.zeros(shape), np.zeros(shape)

            for i, sen in sentences[start:end]:
                # convert tokens to ids
                tokens = self.config.tokenizer.tokenize(''.join(sen))
                token_ids = self.config.tokenizer.convert_tokens_to_ids(tokens)
                token_ids_sp = self.config.tokenizer.build_inputs_with_special_tokens(token_ids)
                n_valid = len(token_ids_sp)

                # pad with '[PAD]' for inputs and '' for tags
                input_ids[i][:n_valid] = token_ids_sp
                tag_ids[i][:n_valid] = [self.config.tag2id[tag] for tag in tags[start:end][i]]

                # attention masks
                att_masks[i][:n_valid] = np.ones(n_valid)

                # convert to torch long tensor
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                tag_ids = torch.tensor(tag_ids, dtype=torch.int)
                att_masks = torch.tensor(att_masks, dtype=int)

            yield input_ids, tag_ids, att_masks


def train(config):
    model = Bert(config)
    model.train()
    optimizer = optim.Adam(model.parameters(), config.lr)
    data_iterator = DataLoader(config).get_batch_iterator()

    for epoch in range(config.n_epochs):

        input_ids, tag_ids, att_masks = next(data_iterator)

        while input_ids:

            optimizer.zero_grad()

            loss = model(input_ids, tag_ids, att_masks)

            loss.backward()

            optimizer.step()

            print("loss: {}".format(loss))


if __name__ == '__main__':
    train(Config())
