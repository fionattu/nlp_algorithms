import re
from config import Config
import numpy as np


def create_data(config):
    sentences, tags = [], []
    with open(config.data_path, 'r') as data:
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


def load_sentences_tags(sentences_path, tags_path, config):
    sentences, tags = create_data(config)
    input_ids_list, tag_ids_list, att_masks_list = [], [], []
    pad_id = config.tokenizer.convert_tokens_to_ids('[PAD]')
    max_len = min(config.max_len, max([len(s) for s in sentences]))
    for i, sen in enumerate(sentences):
        tokens = config.tokenizer.tokenize(''.join(sen))
        token_ids = config.tokenizer.convert_tokens_to_ids(tokens)
        token_ids_sp = config.tokenizer.build_inputs_with_special_tokens(token_ids)

        # pad with '[PAD]' for inputs and '' for tags
        input_ids, tag_ids = [pad_id] * max_len, [0] * max_len
        input_ids[:len(token_ids_sp)] = token_ids_sp
        tag_ids[:len(token_ids_sp)] = [config.tag2id[tag] for tag in tag[i]]

        # attention masks
        att_masks = [0] * max_len
        att_masks[:len(token_ids_sp)] = np.ones(len(token_ids_sp))

        input_ids_list.append(input_ids)
        tag_ids_list.append(tag_ids)
        att_masks_list.append(att_masks)

    data = {'inputs_ids': input_ids_list, 'tag_ids': tag_ids_list, 'att_masks': att_masks_list}
    return data


config = Config()
# create_data(config)
load_sentences_tags("", "", config)
