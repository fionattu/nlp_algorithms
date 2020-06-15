import re
from config import Config


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
                # only train sentence with named entity
                if has_ner:
                    sentences.append(sen_input)
                    tags.append(sen_label)
    return sentences, tags


def load_sentences_tags(sentences_path, tags_path, config):
    sentences, tags = create_data(config)
    sentence_list, tag_list = [], []
    for i, sen in enumerate(sentences):
        tokens = config.tokenizer.tokenize(''.join(sen))
        sentence_list.append(config.tokenizer.convert_tokens_to_ids(tokens))
        tags.append([config.tag2id[tag] for tag in tags[i]])

    return sentences, tags

config = Config()
# create_data(config)
load_sentences_tags("", "", config)
