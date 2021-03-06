from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import collections
import numpy as np
import torch


def read_file(path, desc):
    result = []
    line_counter = len(open(path).readlines())
    with open(path, 'r', encoding='utf-8') as f:
        for _ in tqdm(f, desc=desc, total=line_counter):
            result.append(_.replace('\n', '').split(','))
    f.close()
    return result


def collate_fn(batch):
    """
    :param batch: list of batch_size, each is a tuple returned by __getitem__()
    :return: batch sorted by length
    """
    # tuple=[input_ids, tag_ids, att_masks]
    input_ids, tag_ids, masks, lengths = [], [], [], []
    len_masks = {torch.sum(t[2]).item(): i for i, t in enumerate(batch)}
    sorted_dict = collections.OrderedDict(sorted(len_masks.items(), reverse=True))
    for k, v in sorted_dict.items():
        if k > 0:
            input_ids.append(batch[v][0].numpy())
            tag_ids.append(batch[v][1].numpy())
            masks.append(batch[v][2].numpy())
            lengths.append(k)

    # convert to torch long tensor
    input_ids = torch.LongTensor(input_ids)
    tag_ids = torch.LongTensor(tag_ids)  # should be long
    att_masks = torch.IntTensor(masks)
    lengths = torch.LongTensor(lengths)

    return input_ids, tag_ids, att_masks, lengths


class BertDataset(Dataset):
    def __init__(self, config, path):
        self.inputs = read_file(path[0], 'Loading inputs')
        self.tags = read_file(path[1], 'Loading labels')
        self.tokenizer = config.tokenizer
        self.max_len = config.max_len
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.tag2id = config.tag2id
        self.use_gpu = config.use_gpu
        self.device = config.device

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        sentence, tag = self.inputs[item], self.tags[item]
        if len(sentence) > self.max_len:
            sentence, tag = sentence[:self.max_len], tag[:self.max_len]
        # init inputs, tags, attention_masks
        # pad with '[PAD]' for inputs and '' for tags
        input_ids, att_masks, tag_ids = self.pad_id * np.ones(self.max_len), \
                                        np.zeros(self.max_len), \
                                        np.zeros(self.max_len)

        tokens = self.tokenizer.tokenize(''.join(sentence))
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # masks
        n_valid = len(token_ids)
        tag_id_l = [self.tag2id[t] for t in tag]

        if n_valid == len(tag_id_l):
            input_ids[:n_valid] = token_ids
            tag_ids[:n_valid] = tag_id_l

            # attention masks
            att_masks[:n_valid] = np.ones(n_valid)
        else:
            input_ids, att_masks, tag_ids = self.pad_id * np.ones(self.max_len), \
                                            np.zeros(self.max_len), \
                                            np.zeros(self.max_len)

        # convert to torch long tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        tag_ids = torch.tensor(tag_ids, dtype=torch.long)  # should be long
        att_masks = torch.tensor(att_masks, dtype=torch.int)

        return input_ids, tag_ids, att_masks
