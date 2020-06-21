import re
import torch
import logging

from bert import Bert
from config import Config
import numpy as np
import torch.optim as optim
logging.basicConfig(level=logging.INFO)


class DataLoader(object):
    def __init__(self, config):
        self.config = config

    def load_sentences_tags(self):
        sentences, tags = [], []
        with open(self.config.train_path, 'r', encoding='UTF-8') as data:
            for line in data:
                line = re.split('[，。；！：？、‘’“”]/[o]', line.strip())
                for sentence in line:
                    sen_input, sen_label = [], []
                    words = sentence.strip().split()
                    has_ner = False
                    aldigits_cache = ''
                    for w in words:
                        word, label = w.split('/')
                        if bool(re.search(r'[\da-zA-Z]', word)):
                            aldigits_cache += word
                        else:
                            if len(aldigits_cache) != 0:
                                sen_input.append('1')
                                sen_label.append('o')
                                aldigits_cache = ''
                            sen_input.append(word)
                            sen_label.append(label)
                        if not has_ner and label != 'o':
                            has_ner = True
                    if len(aldigits_cache) != 0:
                        sen_input.append('1')
                        sen_label.append('o')
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

            # init inputs, tags, attention_masks in a
            # pad with '[PAD]' for inputs and '' for tags
            shape_inputs = (end - start, max_len)
            input_ids, att_masks, tag_ids = pad_id * np.ones(shape_inputs), np.zeros(shape_inputs), \
                                            np.zeros(shape_inputs)

            for i, sen in enumerate(sentences[start:end]):
                if len(sen) > max_len:
                    sen, tags[start:end][i] = sen[:max_len], tags[start:end][i][:max_len]
                # convert tokens to ids
                tokens = self.config.tokenizer.tokenize(''.join(sen))
                token_ids = self.config.tokenizer.convert_tokens_to_ids(tokens)

                # masks
                n_valid = len(token_ids)
                tag_id_l = [self.config.tag2id[tag] for tag in tags[start:end][i]]

                if n_valid == len(tag_id_l):
                    input_ids[i][:n_valid] = token_ids
                    tag_ids[i][:n_valid] = tag_id_l

                    # attention masks
                    att_masks[i][:n_valid] = np.ones(n_valid)

            # convert to torch long tensor
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            tag_ids = torch.tensor(tag_ids, dtype=torch.long)  # should be long
            att_masks = torch.tensor(att_masks, dtype=torch.int)

            yield input_ids, tag_ids, att_masks


def train(config):
    model = Bert(config)
    model.train()
    optimizer = optim.Adam(model.parameters(), config.lr)
    data_iterator = DataLoader(config).get_batch_iterator()

    for epoch in range(config.n_epochs):
        logging.info('epoch: {} starts'.format(epoch))

        for input_ids, tag_ids, att_masks in data_iterator:

            optimizer.zero_grad()

            if config.use_gpu:
                input_ids = input_ids.to(config.device)
                tag_ids = tag_ids.to(config.device)
                att_masks = att_masks.to(config.device)

            loss = model(input_ids, tag_ids, att_masks)

            loss.backward()

            optimizer.step()

        logging.info('epoch: {} loss: {}'.format(epoch, loss))


if __name__ == '__main__':
    train(Config())
